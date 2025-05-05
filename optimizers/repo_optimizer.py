import json
from typing import Optional, List, Dict, Tuple, Union
import os
import re
from llms.llm import UniversalLLM
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
    """
    Optimizes C++ code using LLMs.
    """
    def __init__(self, repo_analyzer: RepoAnalyzer):
        self.repo_analyzer = repo_analyzer

    async def build_optimization_prompt_callable(
        self,
        node: str,
        obfuscation_tier: int = 0,
        system_message: Optional[str] = None,
        callee_depth: int = 0,
        node_filter: Optional[callable] = None,   # expects lambda (node_id, attrs) -> bool
        edge_filter: Optional[callable] = None,   # expects lambda ((u,v), attrs) -> bool
        caller_depth: int = 0,
        include_clang_remarks: bool = False,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        llm = UniversalLLM(model, provider)
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
        context_window = "\n".join(context_window)

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

        with open('/home/ayuram/pesquared/prompts/singlefunction_imports_makeflags.txt', 'r') as f:
            prompt = f.read().format(**{
                'function': obfuscated_node,
                'code': ob_code,
            })

        used_nodes = [node] + callee_nodes + caller_nodes
         # 7) Send to the LLM
        return await llm.prompt_async(
                input_prompt=prompt,
                context_window=context_window,
                system_message=system_message,
            ), obfuscator, used_nodes, obfuscation_tier, callee_depth, include_clang_remarks
    
    def implement_optimization(self, raw_resp, obfuscator: CodeObfuscator, used_nodes, obfuscation_tier, callee_depth, include_clang_remarks):
        """
        Implement the optimization by applying the changes to the codebase.
        """
        global ai_counter
        # De-obfuscate the response
        logger.info(f"Raw response: {raw_resp} from {used_nodes[0]} by {self.model}")
        if 'optimizations' not in raw_resp:
            logger.info(f"Model gave no optimizations, skipping {used_nodes[0]}")
            return {"model_optimized": False}
        opt = raw_resp.get("optimizations", None)
        if not isinstance(opt, dict):
            # try to parse it
            try:
                opt = json.loads(opt)
            except:
                logger.error(f"Failed to parse optimization response: {opt}")
                return {"model_optimized": False}
        response = {
            "branch_name": raw_resp.get("branch_name", f"{used_nodes[0]}"),
            "commit_message": raw_resp.get("commit_message", 0),
            "insights": raw_resp.get("insights", 0),
            "optimizations":
                {
                    "function_name": obfuscator.deobfuscate(opt.get('function_name', used_nodes[0])),
                    "brief_description": obfuscator.deobfuscate(opt["brief_description"]),
                    "new_imports": opt.get("new_imports", "// NO CHANGES NEEDED"),
                    # "new_objects": opt["new_objects"],
                    "new_function": obfuscator.deobfuscate(opt["new_function"]),
                },
            "makefile_flags": raw_resp.get("makefile_flags", "// NO CHANGES NEEDED"),
            "obfuscation_tier": obfuscation_tier,
            "callee_depth": callee_depth,
            "include_clang_remarks": include_clang_remarks,
        }
        if  "branch_name" not in response or response["branch_name"].startswith('//'):
            logger.info(f"Model gave no optimizations, skipping {used_nodes[0]}")
            return {"model_optimized": False}

        logger.info(response)
        opt = response['optimizations']
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
        new_function = opt['new_function']
        new_imports = opt['new_imports']
        # new_objects = opt['new_objects']
        branch_name = f"ai-{ai_counter}/{response["branch_name"]}"
        ai_counter += 1

        if not git_utils.create_branch(branch_name=branch_name, codebase_dir=self.repo_analyzer.repo_path):
            print(f"FATAL: Could not create branch {branch_name}, skipping...")
            return
        
        with git_utils.temp_checkout(branch_name=branch_name, codebase_dir=self.repo_analyzer.repo_path, quiet=False):
            # Write the new function to the repo
            if new_function != "// NO CHANGES NEEDED":
                fn = self.repo_analyzer.write_function(
                    function=best_node,
                    code=new_function
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
        return {
            "branch_name": branch_name,
            "insights": response['insights'],
            "model_optimized": True,
            'model': self.model,
            'provider': self.provider,
            'obfuscation_tier': obfuscation_tier,
            'callee_depth': callee_depth,
            'include_clang_remarks': include_clang_remarks,
        }
        
        

    async def optimize_function(
        self,
        node: str,
        obfuscation_tier: int = 0,
        system_message: Optional[str] = None,
        callee_depth: int = 0,
        node_filter: Optional[callable] = None,   # expects lambda (node_id, attrs) -> bool
        edge_filter: Optional[callable] = None,   # expects lambda ((u,v), attrs) -> bool
        caller_depth: int = 0,
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
        raw_resp, obfuscator, used_nodes = await self.build_optimization_prompt_callable(
            node=node,
            obfuscation_tier=obfuscation_tier,
            system_message=system_message,
            callee_depth=callee_depth,
            node_filter=node_filter,
            edge_filter=edge_filter,
            caller_depth=caller_depth,
            include_clang_remarks=include_clang_remarks,
            model=model,
            provider=provider
        )
        return self.implement_optimization(raw_resp, obfuscator, used_nodes)
        
        
    
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
