from typing import Optional, List, Dict, Union
import os
import openai
import anthropic
from google import generativeai as genai
from google.generativeai import types
from collections import deque
import networkx as nx
from analyzers.repo_analyzer import NodeKind, RepoAnalyzer, is_kind
from analyzers.stack_analyzer import StackAnalyzer, FunctionStats
from utils import git_utils
from dotenv import load_dotenv
import asyncio
from verifiers.base_verifier import PerformanceVerifier
import logging
from utils.string_utils import word_list
from obfuscators.code_obfuscator import CodeObfuscator
from difflib import SequenceMatcher
load_dotenv()
logger = logging.getLogger(__name__)

ai_counter = 0

class RepoOptimizer:
    def __init__(self, model: str, provider: str, repo_analyzer: RepoAnalyzer, pv: PerformanceVerifier):
        self.model = model
        self.provider = provider.lower()
        self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        self.repo_analyzer = repo_analyzer
        self.stack_analyzer = StackAnalyzer(str(repo_analyzer.perfstacks_dir))
        self.pv = pv
        self._configure(model, provider)
    
    def _configure(self, model: str, provider: str):
        """
        Configure the LLM client based on the provider.
        """
        self.model = model
        if provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif provider == "google":
            genai.configure(api_key=self.api_key)
            self.client = genai
        else:
            raise ValueError(f"Unsupported provider: {provider!r}. "
                             "Use one of ['openai', 'anthropic', 'google'].")

    def prompt(self,
        input_prompt: str,
        context_window: Optional[List[str]] = None,
        system_message: Optional[str] = "You are an expert C++ performance optimization engineer.",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_schema: Optional[Dict] = None,
        ) -> Union[str, Dict]:
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

    async def prompt_async(
        self,
        input_prompt: str,
        context_window: Optional[List[str]] = None,
        system_message: Optional[str] = "You are an expert C++ performance optimization engineer.",
        temperature: float = 0.9,
        max_tokens: int = 4096,
        response_schema: Optional[Dict] = None,
    ) -> Union[str, Dict]:
        """
        Async version of prompt: Send a prompt to the LLM and get a response.
        """
        if self.provider in ("openai", "anthropic"):
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            if context_window:
                messages.extend({"role": "user", "content": m} for m in context_window)
            messages.append({"role": "user", "content": input_prompt})

            if self.provider == "openai":
                if response_schema:
                    functions = [{
                        "name": "optimization_response",
                        "description": "Structured C++ optimization output",
                        "parameters": response_schema,
                    }]
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        functions=functions,
                        function_call={"name": "optimization_response"},
                    )
                    import json
                    return json.loads(resp.choices[0].message.function_call.arguments)
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
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(full_prompt, generation_config=generation_config)
                )
                import json
                import re
                try:
                    return json.loads(resp.text)
                except json.JSONDecodeError:
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
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(full_prompt, generation_config=generation_config)
                )
                return resp.text

        else:
            raise RuntimeError(f"Invalid provider: {self.provider!r}")

    async def optimize_function(
        self,
        node: str,
        obfuscation_tier: int = 0,
        system_message: Optional[str] = None,
        callee_depth: int = 0,
        node_filter: Optional[callable] = None,   # expects lambda (node_id, attrs) -> bool
        edge_filter: Optional[callable] = None,   # expects lambda ((u,v), attrs) -> bool
        caller_depth: int = 0,
        samples: int = 3,
        include_clang_remarks: bool = False,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict:
        global ai_counter
        """
        Optimize a code node from a graph.  Gathers up to `context_depth` hops
        of neighbors (filtered by node_filter/edge_filter), obfuscates them,
        sends to the LLM, then deobfuscates the diff.
        """
        if model and provider:
            self._configure(model, provider)
        # 1) Grab the code for the target node
        code = self.repo_analyzer.get_code(node)
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
            logger.info(f"Makefile FLAGS: {makefile_flags}")

        # Callees
        for n in callee_nodes:
            attrs = self.repo_analyzer.nodes[n]
            snippet = attrs.get("code", "").strip()
            if snippet:
                ob_snippet = obfuscator.obfuscate(snippet)
                context_window.append(
                    f"📄 Callee — {obfuscator.obfuscate(n)}\n```cpp\n{ob_snippet}\n```"
                )
                # logger.debug(f"Callee {n}: {ob_snippet}")

        # Callers
        for n in caller_nodes:
            attrs = self.repo_analyzer.nodes[n]
            snippet = attrs.get("code", "").strip()
            if snippet:
                ob_snippet = obfuscator.obfuscate(snippet)
                context_window.append(
                    f"📄 Caller — {obfuscator.obfuscate(n)}\n```cpp\n{ob_snippet}\n```"
                )
                # logger.debug(f"Caller {n}: {ob_snippet}")

        ob_code = obfuscator.obfuscate(code)
        # Attach clang remarks to obfuscated code
        remarks = self.repo_analyzer.nodes[node].get("missed_optimizations", {})
        # remarks[line_number] = [list of remarks for that line]
        ob_code_list = ob_code.splitlines()
        if include_clang_remarks:
            for line_number, remarks_list in remarks.items():
                if line_number in obfuscator.obfuscation_map:
                    obfuscated_line = obfuscator.obfuscation_map[line_number]
                    remarks_str = " | ".join(remarks_list)
                    ob_code_list[obfuscated_line] += f" // {remarks_str}"
        ob_code = "\n".join(ob_code_list)
        # logger.info(f"Obfuscated code: {ob_code}")
        obfuscated_node = obfuscator.obfuscate(node)

        # Determine if function is "long"
        is_long_function = False

        # 6) Build the structured prompt (unified-diff focus)
        prompt = (
            f"You are an expert C++ performance engineer. Optimize the following C++ function `{obfuscated_node}` for performance.\n\n"
            +
            (
               """CRITICAL REQUIREMENTS:\n
1. DO NOT modify the function signature (parameter types, return type, name) - it MUST remain EXACTLY as provided\n
2. DO NOT inline the function or merge it with others\n
3. ONLY modify the internal implementation while preserving exact behavior\n
4. If no valuable optimizations exist, return the original function unmodified\n
5. Your code MUST compile without errors\n""" 
            )
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
            f"{'⚙️ Return strictly as unified diffs.' if is_long_function else '✍️ Return the changed functions as a drop-in replacement. Do NOT return a diff; Do NOT skip lines; Do NOT change the signature; return the full rewritten function.'}\n\n"
            "```cpp\n"
            f"{ob_code}\n"
            "```\n\n"
            'Your response is machine-processed, so include every detail verbosely. CRITICALLY: Do NOT alter the function name or signature!!! If an optimization is not possible given these constraints, simply return an empty list of optimizations\n\n'
            +
            (
                "Your code should look something like:\n"
                f"{obfuscated_node.splitlines()[0]} // <only change the body>\n"
            )
        )
        response_schema = {
            "type": "object",
            "required": ["insights", "optimizations"],
            "properties": {
                "branch_name": {
                    "type": "string",
                    "description": (
                        "Name of the branch to create for this optimization"
                    )
                },
                "commit_message": {
                    "type": "string",
                    "description": (
                        "Commit message for the optimization"
                    )
                },
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
                            # "new_objects",
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
                            # "new_objects": {
                            #     "type": "string",
                            #     "description": "New functions/structs/classes/variables or '// NO CHANGES NEEDED'"
                            # },
                            "rewritten_function": {
                                "type": "string",
                                "description": (
                                    f"{ob_code.splitlines()[0]}\n    <only change the body>\n" + "}\n or '// NO CHANGES NEEDED'"
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

        # 7) Send to the LLM
        tasks = [
            self.prompt_async(
            input_prompt=prompt,
            context_window=context_window,
            system_message=system_message,
            response_schema=response_schema,
            )
            for _ in range(samples)
        ]
        responses = await asyncio.gather(*tasks)
        # TODO: actually run all responses
        # Use the first non-string response as the raw response
        raw_resp = next((resp for resp in responses if not isinstance(resp, str)), None)
        logger.debug(f"LLM responses: {raw_resp}")
        if raw_resp is None:
            raise ValueError(f"All LLM responses were strings: {responses}")

        if isinstance(raw_resp, str):
            raise ValueError(f"LLM returned a string instead of JSON: {raw_resp}")
        # De-obfuscate the response
        response = {
            "branch_name": raw_resp.get("branch_name", f"{node}"),
            "commit_message": raw_resp.get("commit_message", 0),
            "insights": raw_resp.get("insights", 0),
            "optimizations": [
                {
                    "function_name": obfuscator.deobfuscate(opt["function_name"]),
                    "brief_description": obfuscator.deobfuscate(opt["brief_description"]),
                    "new_imports": opt["new_imports"],
                    # "new_objects": opt["new_objects"],
                    "rewritten_function": obfuscator.deobfuscate(opt["rewritten_function"]),
                }
                for opt in raw_resp["optimizations"]
            ],
            "makefile_flags": raw_resp.get("makefile_flags", "// NO CHANGES NEEDED")
        }
        used_nodes = caller_nodes + callee_nodes + [node]
        optimizations = response['optimizations']
        if len(optimizations) == 0 or 'branch_name' not in response or response['branch_name'].startswith('//'):
            logger.info(f"Model gave no optimizations, skipping {node}")
            return {"model_optimized": False}
        for opt in optimizations:
            function_name = opt['function_name']
            # Choose the node with the highest similarity ratio using SequenceMatcher
            highest_similarity = 0.0
            best_node = None
            for node in used_nodes:
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
            # new_objects = opt['new_objects']
            branch_name = f"ai-{ai_counter}/{response["branch_name"]}"

            if not git_utils.create_branch(branch_name=branch_name, codebase_dir=self.repo_analyzer.repo_path):
                print(f"FATAL: Could not create branch {branch_name}, skipping...")
                return
            
            with git_utils.temp_checkout(branch_name=branch_name, codebase_dir=self.repo_analyzer.repo_path, quiet=False):
                # Write the new function to the repo
                if rewritten_function != "// NO CHANGES NEEDED":
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
                # if new_objects != "// NO CHANGES NEEDED":
                #     self.repo_analyzer.add_objects(
                #         fn=fn,
                #         objects=new_objects
                #     )
                # Write the new Makefile flags to the repo
                if response['makefile_flags'] != "// NO CHANGES NEEDED":
                    append = False
                    value = ""
                    if '+=' in response['makefile_flags']:
                        append = True
                        value = response['makefile_flags'].split('+=')[1].strip()
                    elif ':=' in response['makefile_flags']:
                        value = response['makefile_flags'].split(':=')[1].strip()
                    else:
                        value = ''
                        append = True
                    self.repo_analyzer.write_to_makefile(
                        field='FLAGS',
                        value=value,
                        append=append,
                    )
                git_utils.stage_file(self.repo_analyzer.nodes[best_node]['file'], codebase_dir=self.repo_analyzer.repo_path)
                git_utils.commit_changes(response["commit_message"], codebase_dir=self.repo_analyzer.repo_path)
        # 8) Verify the performance of the optimization
        perf, tests_pass = self.pv.get_performance(branch=branch_name)
        if perf is None:
            print(f"Could not get performance for branch {branch_name}, likely due to a failed build.")
        ai_counter += 1
        return {
            **perf,
            "tests_pass": tests_pass,
            "branch_name": response['branch_name'],
            "insights": response['insights'],
            "model_optimized": True
        }
        
    
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
            self.optimize_function(
                node=bottleneck_node,
                obfuscation_tier=0,
                callee_depth=1,
                caller_depth=1,
                node_filter=lambda n, attrs: is_kind(attrs.get("kind", NodeKind.UNKNOWN), NodeKind.IN_CODEBASE),
            )            
