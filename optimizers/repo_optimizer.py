from typing import Optional, List, Dict, Union
import os
import openai
import anthropic
from google import generativeai as genai
from google.generativeai import types
from collections import deque
import networkx as nx
from analyzers.code_analyzer import NodeKind, RepoAnalyzer, is_kind
from analyzers.stack_analyzer import StackAnalyzer, FunctionStats
from dotenv import load_dotenv

from obfuscators.code_obfuscator import CodeObfuscator
from utils.string_utils import _edit_distance, edit_distance
from difflib import SequenceMatcher
load_dotenv()

class RepoOptimizer:
    def __init__(self, model: str, provider: str, repo_analyzer: RepoAnalyzer):
        self.model = model
        self.provider = provider.lower()
        self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        self.repo_analyzer = repo_analyzer
        self.stack_analyzer = StackAnalyzer(str(repo_analyzer.perfstacks_dir))

        # Initialize the appropriate client
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)

        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)

        elif self.provider == "google":
            # Set API key securely and directly
            if not self.api_key:
                raise ValueError("Missing API key for Google provider.")
            
            genai.configure(api_key=self.api_key)
            self.client = genai  # Keep the module reference for later GenerativeModel use


        else:
            raise ValueError(f"Unsupported provider: {provider!r}. "
                             "Use one of ['openai', 'anthropic', 'google'].")

    def prompt(self,
        input_prompt: str,
        context_window: Optional[List[str]] = None,
        system_message: Optional[str] = "You are an expert C++ performance optimization engineer.",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_schema: Optional[Dict] = None) -> Union[str, Dict]:
        """
        Send a prompt to the LLM and get a response.

        Args:
            input_prompt: The prompt to send
            context_window: Optional list of context messages
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            response_schema: Optional JSON schema to constrain the output

        Returns:
            Either a string response (if no schema) or a dictionary (if schema provided)
        """

        if self.provider in ("openai", "anthropic"):
            # Build the messages array
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if context_window:
                messages.extend({"role": "user", "content": m} for m in context_window)
            messages.append({"role": "user", "content": input_prompt})

            # === OPENAI branch ===
            if self.provider == "openai":
                if response_schema:
                    # 1) Wrap your schema in a single function definition
                    functions = [{
                        "name": "optimization_response",
                        "description": "Structured C++ optimization output",
                        "parameters": response_schema,
                    }]

                    # 2) Ask the model to call that function
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        functions=functions,
                        function_call={"name": "optimization_response"},
                    )

                    # 3) Pull the JSON arguments out of the function_call
                    func_call = resp.choices[0].message.function_call
                    import json
                    return json.loads(func_call.arguments)
                else:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content

            else:  # anthropic
                if response_schema:
                    tools = [{
                        "name": "json_output",
                        "description": "Respond with a JSON object.",
                        "input_schema": response_schema,
                    }]
                    resp = self.client.messages.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice={"type": "tool", "name": "json_output"},
                    )
                    if resp.content[0].type == "tool_use":
                        return resp.content[0].input
                    else:
                        return resp.content[0].text
                else:
                    resp = self.client.messages.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return resp.content[0].text

        elif self.provider == "google":
            # Gemini branch (unchanged from earlier)
            model = self.client.GenerativeModel(self.model)
            full_prompt = ""
            if system_message:
                full_prompt += f"{system_message}\n\n"
            if context_window:
                full_prompt += "\n".join(context_window) + "\n\n"
            full_prompt += input_prompt

            if response_schema:
                generation_config = self.client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )
                resp = model.generate_content(full_prompt, generation_config=generation_config)
                print("--- RAW RESPONSE ---")
                print(resp.text)
                print("--- END RAW RESPONSE ---")
                import json
                import re

                try:
                    return json.loads(resp.text)
                except json.JSONDecodeError:
                    print("Bad JSON, trying to fix it...")
                    # Try a simple fix: trim to the first full JSON object
                    match = re.search(r'\{.*\}', resp.text, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Still bad JSON after repair attempt: {e}")
                    else:
                        raise ValueError(f"Cannot find JSON object in response: {resp.text}")
            else:
                generation_config = self.client.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                resp = model.generate_content(full_prompt, generation_config=generation_config)
                return resp.text

        else:
            raise RuntimeError(f"Invalid provider: {self.provider!r}")

    def _optimize_node(
        self,
        node: str,
        obfuscation_tier: int = 0,
        system_message: Optional[str] = None,
        callee_depth: int = 0,
        node_filter: Optional[callable] = None,   # expects lambda (node_id, attrs) -> bool
        edge_filter: Optional[callable] = None,   # expects lambda ((u,v), attrs) -> bool
        caller_depth: int = 0,
    ) -> Dict:
        """
        Optimize a code node from a graph.  Gathers up to `context_depth` hops
        of neighbors (filtered by node_filter/edge_filter), obfuscates them,
        sends to the LLM, then deobfuscates the diff.
        """
        # 1) Grab the code for the target node
        labels = self.repo_analyzer.nodes[node]
        code = labels.get("code", "").strip()
        if not code:
            raise ValueError(f"Node {node!r} has no code")

        # 2) BFS out to collect context nodes up to context_depth
        lengths = nx.single_source_shortest_path_length(self.repo_analyzer, node, cutoff=callee_depth)
        # drop the root, then apply node_filter
        callee_nodes = [
            n for n in lengths
            if n != node
            and (node_filter is None or node_filter(n, self.repo_analyzer.nodes[n]))
        ]

        # 3) (Optional) apply edge_filter: ensure at least one valid path exists
        if edge_filter is not None:
            def has_valid_path(target):
                for path in nx.all_simple_paths(self.repo_analyzer, source=node, target=target, cutoff=callee_depth):
                    # check every edge in this path
                    if all(
                        edge_filter((u, v), self.repo_analyzer.edges[u, v])
                        for u, v in zip(path, path[1:])
                    ):
                        return True
                return False
            callee_nodes = [n for n in callee_nodes if has_valid_path(n)]
        
        # 3.5) add caller nodes using BFS on the predecessors (ensuring nodes and edge filters are applied)
        # We can't use the same function as above, because we need to check the predecessors
        caller_nodes = []
        bfs = deque([(node, caller_depth)])
        visited = set()
        while bfs:
            current_node, depth = bfs.popleft()
            if depth == 0:
                continue
            for pred in self.repo_analyzer.predecessors(current_node):
                if pred not in visited and (node_filter is None or node_filter(pred, self.repo_analyzer.nodes[pred])):
                    visited.add(pred)
                    caller_nodes.append(pred)
                    bfs.append((pred, depth - 1))
        

        # 4) Prepare obfuscator over the target + contexts
        ob_nodes = [self.repo_analyzer.nodes[n] for n in [node] + callee_nodes + caller_nodes]
        obfuscator = CodeObfuscator(
            tier=obfuscation_tier,
            nodes=ob_nodes,
            project_root=self.repo_analyzer.repo_path,
        )

        # 5) Obfuscate each code snippet and build context_window
        context_window = []

        # Makefile FLAGS add to context window
        makefile_flags = self.repo_analyzer.nodes["FLAGS"]["code"].strip()
        if makefile_flags:
            context_window.append(
                f"📄 Makefile FLAGS\n```Makefile\n{makefile_flags}\n```"
            )

        # Callees
        for n in callee_nodes:
            attrs = self.repo_analyzer.nodes[n]
            snippet = attrs.get("code", "").strip()
            if snippet:
                ob_snippet = obfuscator.obfuscate(snippet)
                context_window.append(
                    f"📄 Callee — {obfuscator.obfuscate(n)}\n```cpp\n{ob_snippet}\n```"
                )

        # Callers
        for n in caller_nodes:
            attrs = self.repo_analyzer.nodes[n]
            snippet = attrs.get("code", "").strip()
            if snippet:
                ob_snippet = obfuscator.obfuscate(snippet)
                context_window.append(
                    f"📄 Caller — {obfuscator.obfuscate(n)}\n```cpp\n{ob_snippet}\n```"
                )

        ob_code = obfuscator.obfuscate(code)
        obfuscated_node = obfuscator.obfuscate(node)

        # Determine if function is "long"
        is_long_function = len(code.splitlines()) > 90

        # 6) Build the structured prompt (unified-diff focus)
        prompt = (
            f"You are an expert C++ performance engineer. Optimize the following C++ function `{obfuscated_node}` for performance.\n\n"
            +
            (
                'You have access to:\n'
                '- Callee context\n'
                '- Caller context\n\n'
            )
            +
            "Please note:\n"
            f"- Callees and Callers are NOT exclusive to `{obfuscated_node}` and might be used by other functions\n"
            f"{'- The code is fully correct and compilable as is' if (obfuscation_tier == 0) else '- The code has been obfuscated but is still correct and compilable as is'}\n\n"
            f"{'⚙️ Return strictly as unified diffs.' if is_long_function else '✍️ Return the changed functions as a drop-in replacement. Do not return a diff; Do NOT skip lines; return the full rewritten function.'}\n\n"
            "```cpp\n"
            f"{ob_code}\n"
            "```\n\n"
            'Your response is machine-processed, so include every detail verbosely'
        )

        # 7) Send to the LLM
        raw_resp = self.prompt(
            input_prompt=prompt,
            context_window=context_window,
            system_message=system_message,
            response_schema = {
            "type": "object",
            "required": ["insights", "optimizations"],
            "properties": {
                "insights": {
                    "type": "string",
                    "description": "Outline of your plan: what to change and why"
                },
                "optimizations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "function_name",
                            "brief_description",
                            "new_imports",
                            "new_objects",
                            "rewritten_function"
                        ],
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": "The name of the function you changed"
                            },
                            "brief_description": {
                                "type": "string",
                                "description": "Brief description of change"
                            },
                            "new_imports": {
                                "type": "string",
                                "description": "New imports or '// NO CHANGES NEEDED'"
                            },
                            "new_objects": {
                                "type": "string",
                                "description": "New functions/structs/classes/variables or '// NO CHANGES NEEDED'"
                            },
                            "rewritten_function": {
                                "type": "string",
                                "description": (
                                    "the full rewritten function drop-in replacement"
                                )
                            }
                        }
                    }
                },
                "makefile_flags": {
                    "type": "string",
                    "description": (
                        "New Makefile FLAGS, e.g. 'FLAGS := <new build flags>' "
                        "or '// NO CHANGES NEEDED'"
                    )
                }
            }
        }
        )

        if isinstance(raw_resp, str):
            raise ValueError(f"LLM returned a string instead of JSON: {raw_resp}")
        # De-obfuscate the response
        deobfuscated_response = {
            "insights": raw_resp["insights"],
            "optimizations": [
                {
                    "function_name": obfuscator.deobfuscate(opt["function_name"]),
                    "brief_description": obfuscator.deobfuscate(opt["brief_description"]),
                    "new_imports": opt["new_imports"],
                    "new_objects": opt["new_objects"],
                    "rewritten_function": obfuscator.deobfuscate(opt["rewritten_function"]),
                }
                for opt in raw_resp["optimizations"]
            ],
            "makefile_flags": raw_resp.get("makefile_flags", "// NO CHANGES NEEDED")
        }
        return deobfuscated_response, caller_nodes + callee_nodes + [node]
    
    def optimize_repo(self):
        # 1) Use StackAnalyzer to get the bottleneck call trace
        stats = self.stack_analyzer.get_top_functions(1)[0]
        for bottleneck_name in list(reversed(stats.call_chain))[:3]:
            print(f"Optimizing {bottleneck_name}")
            bottleneck_nodes = self.repo_analyzer.symbol_to_signatures[bottleneck_name]
            if len(bottleneck_nodes) != 1:
                continue
            bottleneck_node = bottleneck_nodes[0]
            # 2) Optimize the bottleneck function
            response, nodes = self._optimize_node(
                node=bottleneck_node,
                obfuscation_tier=0,
                callee_depth=1,
                caller_depth=1,
                node_filter=lambda n, attrs: is_kind(attrs.get("kind", NodeKind.UNKNOWN), NodeKind.IN_CODEBASE),
            )
            # 3) Write the response functions to the repo
            optimizations = response['optimizations']
            for opt in optimizations:
                function_name = opt['function_name']
                # Choose the node with the highest similarity ratio using SequenceMatcher
                highest_similarity = 0.0
                best_node = None
                for node in nodes:
                    if not is_kind(self.repo_analyzer.nodes[node].get("kind", NodeKind.UNKNOWN), NodeKind.FUNCTION):
                        continue
                    if not is_kind(self.repo_analyzer.nodes[node].get("kind", NodeKind.UNKNOWN), NodeKind.IN_CODEBASE):
                        continue
                    similarity = SequenceMatcher(None, function_name.replace(' ', '').split('(')[0], node.replace(' ', '').split('(')[0]).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_node = node
                print(f"Best match for {function_name} is {best_node} with similarity {highest_similarity:.2f}")
                rewritten_function = opt['rewritten_function']
                new_imports = opt['new_imports']
                new_objects = opt['new_objects']
                # Write the new function to the repo
                fn = self.repo_analyzer.write_function(
                    function=best_node,
                    code=rewritten_function
                )
                # Add the new imports statements to the file
                if new_imports != "// NO CHANGES NEEDED":
                    self.repo_analyzer.add_imports(
                        fn=fn,
                        imports=new_imports
                    )
                # Add the new objects to a file next to the function
                if new_objects != "// NO CHANGES NEEDED":
                    self.repo_analyzer.add_objects(
                        fn=fn,
                        objects=new_objects
                    )
                # Write the new Makefile flags to the repo
                if response['makefile_flags'] != "// NO CHANGES NEEDED":
                    self.repo_analyzer.write_to_makefile(
                        field='FLAGS',
                        value=response['makefile_flags']
                    )
