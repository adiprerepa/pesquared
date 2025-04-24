import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Set
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from clang import cindex
from utils.cursor_utils import loc_info, CTOR_KINDS, FUNC_KINDS, TYPE_KINDS, TEMPLATE_PARAM_KINDS, MEMBER_KINDS, GLOBAL_KINDS

_CLANG_INITIALIZED = False

from enum import IntFlag

class NodeKind(IntFlag):
    # Base categories
    UNKNOWN = 0
    FUNCTION = 1 << 0
    VARIABLE = 1 << 1
    TYPE = 1 << 2

    # Subcategories embed their base bit
    CONSTRUCTOR = FUNCTION | (1 << 3)
    DESTRUCTOR = FUNCTION | (1 << 4)
    METHOD = FUNCTION | (1 << 5)
    FREE_FUNCTION = FUNCTION | (1 << 6)

    MEMBER_VAR = VARIABLE | (1 << 7)
    GLOBAL_VAR = VARIABLE | (1 << 8)
    LOCAL_VAR = VARIABLE | (1 << 9)

    CLASS = TYPE | (1 << 10)
    STRUCT = TYPE | (1 << 11)
    ENUM = TYPE | (1 << 12)
    TYPEDEF = TYPE | (1 << 13)
    TEMPLATE = TYPE | (1 << 14)

    # Origin modifiers
    IN_CODEBASE = 1 << 24
    LIBRARY = 1 << 25


class EdgeType(IntFlag):
    # Base categories
    UNKNOWN = 0
    CALL = 1 << 0       # 1 - Function/method calls
    REFERENCE = 1 << 1  # 2 - References to entities
    DEPENDENCY = 1 << 2 # 4 - Type dependencies
    
    # Call subcategories
    FUNCTION_CALL = CALL | (1 << 4)      # 17
    METHOD_CALL = CALL | (1 << 5)        # 33
    CONSTRUCTOR_CALL = CALL | (1 << 6)   # 65
    DESTRUCTOR_CALL = CALL | (1 << 7)    # 129
    
    # Reference subcategories
    FUNCTION_REF = REFERENCE | (1 << 8)  # 258
    MEMBER_REF = REFERENCE | (1 << 9)    # 514
    GLOBAL_REF = REFERENCE | (1 << 10)   # 1026
    TYPE_REF = DEPENDENCY | (1 << 11)    # 2052
    NAMESPACE_REF = REFERENCE | (1 << 12) # 4098
    
    # Additional properties
    DIRECT = 1 << 24    # Direct reference in source
    INDIRECT = 1 << 25  # Indirect reference (e.g., via typedef)

def is_kind(node, kind):
    return (node & kind) == kind

class CodeAnalyzer:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()
        self.graph = nx.DiGraph()
        self._file_cache: Dict[str, List[str]] = {}
        self._init_clang()
        self.index = cindex.Index.create()
        self._build_graph()

    # ──────────────────── Clang init ───────────────────────────────────

    def _init_clang(self) -> None:
        global _CLANG_INITIALIZED
        if _CLANG_INITIALIZED:
            return
        for p in [
            "/usr/lib/llvm-18/lib",
            "/usr/lib/llvm-17/lib",
            "/usr/lib/llvm-16/lib",
            "/usr/lib/llvm-15/lib",
            "/usr/lib64/llvm",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/llvm/lib",
        ]:
            lib = Path(p) / "libclang.so"
            if lib.exists():
                try:
                    cindex.Config.set_library_file(str(lib))
                    cindex.Config.set_compatibility_check(False)
                    _ = cindex.Index.create()
                    _CLANG_INITIALIZED = True
                    print(f"✅ libclang loaded from: {lib}")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load {lib}: {e}")
        raise RuntimeError("libclang.so not found - install clang (libclang-dev)")

    # ──────────────────── Helpers ─────────────────────────────────────

    def base_name_variants(self, name: str) -> set:
        variants = set()
        if not name:
            return variants
        variants.add(name)
        name_no_ret = name.split()[-1]
        variants.add(name_no_ret)
        no_args = re.sub(r'\(.*\)', '', name_no_ret)
        variants.add(no_args)
        base = no_args.split("::")[-1]
        variants.add(base)
        no_template = re.sub(r'<.*?>', '', base)
        variants.add(no_template)
        return {v.strip() for v in variants if v.strip()}

    def is_mangled(self, name: str) -> bool:
        return name.startswith("_Z")

    def demangle(self, name: str) -> str:
        if not self.is_mangled(name):
            return name
        try:
            out = subprocess.run(["c++filt", name], capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except FileNotFoundError:
            pass
        return name

    def fq_name(self, cur: cindex.Cursor) -> str:
        parts = []
        c = cur
        while c and c.kind != cindex.CursorKind.TRANSLATION_UNIT:
            if c.kind in (
                cindex.CursorKind.NAMESPACE,
                cindex.CursorKind.CLASS_DECL,
                cindex.CursorKind.STRUCT_DECL,
                cindex.CursorKind.CLASS_TEMPLATE,
            ):
                parts.append(c.spelling)
            c = c.semantic_parent
        parts.reverse()
        if cur.spelling:
            parts.append(cur.spelling)
        n = "::".join(parts)
        return self.demangle(n) if self.is_mangled(n) else n

    def read_extent(self, cur: cindex.Cursor) -> str:
        ext = cur.extent
        if not ext or not ext.start.file:
            return ""
        fn = ext.start.file.name
        if fn not in self._file_cache:
            try:
                with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
                    self._file_cache[fn] = f.readlines()
            except:
                self._file_cache[fn] = []
        
        lines = self._file_cache[fn]
        start_line = ext.start.line - 1  # 0-based index
        end_line = ext.end.line - 1      # 0-based index
        start_col = ext.start.column - 1 # 0-based index
        end_col = ext.end.column - 1     # 0-based index
        
        # Ensure we don't go out of bounds
        if start_line >= len(lines) or start_line < 0:
            return ""
        
        if start_line == end_line:
            # Single line case
            line = lines[start_line]
            return line[start_col:end_col+1] if end_col < len(line) else line[start_col:]
        else:
            # Multi-line case
            result = []
            # First line from start column to end of line
            if start_line < len(lines):
                result.append(lines[start_line][start_col:])
            
            # Middle lines (complete)
            for l in range(start_line + 1, min(end_line, len(lines))):
                result.append(lines[l])
            
            # Last line from start to end column
            if end_line < len(lines):
                line = lines[end_line]
                result.append(line[:end_col+1] if end_col < len(line) else line)
            
            return ''.join(result)

    def outline_ast(self, cur: cindex.Cursor, depth=0) -> List[str]:
        lines = ["  " * depth + cur.kind.name]
        for c in cur.get_children():
            lines.extend(self.outline_ast(c, depth + 1))
        return lines

    # ─── Helper to build fully-qualified name with signature for ctors/dtors ───
    def _signature(self, cur: cindex.Cursor) -> str:
        # Build namespace/class qualifier chain
        parts = []
        p = cur.semantic_parent
        while p and p.kind != cindex.CursorKind.TRANSLATION_UNIT:
            if p.kind in (
                cindex.CursorKind.NAMESPACE,
                cindex.CursorKind.CLASS_DECL,
                cindex.CursorKind.STRUCT_DECL,
                cindex.CursorKind.CLASS_TEMPLATE,
            ):
                parts.append(p.spelling)
            p = p.semantic_parent
        parts.reverse()
        prefix = "::".join(parts)

        # Destructor: no args
        if cur.kind == cindex.CursorKind.DESTRUCTOR:
            base = cur.semantic_parent.spelling
            name = f"{prefix}::~{base}()"
        else:
            # Constructor: include parameter types
            base = cur.spelling  # class name
            params = ", ".join(arg.type.spelling for arg in cur.get_arguments())
            name = f"{prefix}::{base}({params})"
        return name
                

    # ─── Updated collect_nodes ───
    def collect_nodes(self, cursor: cindex.Cursor) -> None:
        def in_proj(c: cindex.Cursor) -> bool:
            return bool(c.location.file) and Path(c.location.file.name).resolve().is_relative_to(self.repo_path)
        
        def ensure_node(name: str, kind: NodeKind, cur: Optional[cindex.Cursor]):
            if name not in self.graph or len(self.graph.nodes[name]['ast']) < len("\n".join(self.outline_ast(cur))):
                self.graph.add_node(name, kind=kind, code="", ast="", cursor=None, **(loc_info(cur) if cur else {}))
                if cur:
                    if self.graph.nodes[name]["code"] == "":
                        self.graph.nodes[name]["code"] = self.read_extent(cur)
                    if self.graph.nodes[name]["ast"] == "":
                        try:
                            self.graph.nodes[name]["ast"] = "\n".join(self.outline_ast(cur))
                        except Exception:
                            self.graph.nodes[name]["ast"] = "<AST outline unavailable>"
                    if self.graph.nodes[name]["cursor"] is None:
                        self.graph.nodes[name]["cursor"] = cur

        # --- functions, methods, templates, constructors, destructors ---
        node_kind = NodeKind.UNKNOWN
        if cursor.kind in FUNC_KINDS:    
            # if cursor.is_definition()        
            if cursor.kind in CTOR_KINDS:
                fq = self._signature(cursor)
                if cursor.kind == cindex.CursorKind.DESTRUCTOR:
                    node_kind |= NodeKind.DESTRUCTOR
                else:
                    node_kind |= NodeKind.CONSTRUCTOR
            else:
                fq = self.fq_name(cursor)
                node_kind |= NodeKind.METHOD if '::' in fq else NodeKind.FREE_FUNCTION
            node_kind |= NodeKind.IN_CODEBASE if in_proj(cursor) else NodeKind.LIBRARY
            ensure_node(fq, node_kind, cursor)
            # else:
            #     if cursor.kind in CTOR_KINDS:
            #         fq = self._signature(cursor)
            #     else:
            #         fq = self.fq_name(cursor)
            #     if fq not in self.graph:
            #         ensure_node(fq, "function", cursor)
        
        # --- member variables ---
        elif cursor.kind in MEMBER_KINDS and in_proj(cursor):
            fq = self.fq_name(cursor)
            node_kind |= NodeKind.MEMBER_VAR
            node_kind |= NodeKind.IN_CODEBASE if in_proj(cursor) else NodeKind.LIBRARY
            ensure_node(fq, node_kind, cursor)

        # --- global variables ---
        elif cursor.kind in GLOBAL_KINDS and cursor.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT and in_proj(cursor):
            fq = self.fq_name(cursor)
            node_kind = NodeKind.GLOBAL_VAR
            node_kind |= NodeKind.IN_CODEBASE if in_proj(cursor) else NodeKind.LIBRARY
            ensure_node(fq, node_kind, cursor)

        # --- types ---
        elif cursor.kind in TYPE_KINDS and in_proj(cursor):
            fq = self.fq_name(cursor)
            node_kind = NodeKind.TYPE
            node_kind |= NodeKind.IN_CODEBASE if in_proj(cursor) else NodeKind.LIBRARY
            ensure_node(fq, node_kind, cursor)

        # recurse
        for child in cursor.get_children():
            self.collect_nodes(child)

    def add_edge_strict(self, u: str, v: str, edge_type: EdgeType = EdgeType.UNKNOWN, **kwargs) -> None:
        if u == v:
            return
        if self.graph.has_edge(u, v):
            # If edge exists, update the edge type by combining with any existing type
            current_type = self.graph.edges[u, v].get("type", EdgeType.UNKNOWN)
            self.graph.edges[u, v]["type"] = current_type | edge_type
            return
        if not self.graph.has_node(u) or not self.graph.has_node(v):
            return
        else:
            self.graph.add_edge(u, v, type=edge_type, **kwargs)


    def collect_edges(self, cursor: cindex.Cursor, ctx_func: Optional[str]) -> None:
        # Update context if we're inside a known function or template
        if cursor.kind in FUNC_KINDS and cursor.is_definition():
            if cursor.kind in CTOR_KINDS:
                ctx_func = self._signature(cursor)
            else:
                ctx_func = self.fq_name(cursor)

        if cursor.kind in TYPE_KINDS and cursor.is_definition():
            ctx_func = self.fq_name(cursor)
        
        if cursor.kind in MEMBER_KINDS and cursor.is_definition():
            ctx_func = self.fq_name(cursor)

        if cursor.kind in GLOBAL_KINDS and cursor.is_definition() and cursor.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT:
            ctx_func = self.fq_name(cursor)

        # --- Function call detection
        if cursor.kind == cindex.CursorKind.CALL_EXPR and ctx_func:
            callee_name = None
            raw_spelling = ""

            # Direct reference to the function being called
            if cursor.referenced and cursor.referenced.kind in FUNC_KINDS:
                if cursor.referenced.kind in CTOR_KINDS:
                    callee_name = self._signature(cursor.referenced)
                else:
                    callee_name = self.fq_name(cursor.referenced)
                raw_spelling = cursor.referenced.spelling or ""

            # Fallback: search for the function name in children
            if not callee_name:
                for child in cursor.get_children():
                    if (
                        child.kind == cindex.CursorKind.DECL_REF_EXPR and
                        child.spelling and
                        (not child.referenced or child.referenced.kind in FUNC_KINDS)
                    ):
                        raw_spelling = child.spelling
                        if child.referenced and child.referenced.kind in FUNC_KINDS:
                            callee_name = self.fq_name(child.referenced)
                        else:
                            variants = self.base_name_variants(child.spelling)
                            matches = [
                                n for n, d in self.graph.nodes(data=True)
                                if d.get("kind") & NodeKind.FUNCTION and variants & self.base_name_variants(n)
                            ]
                            if len(matches) == 1:
                                callee_name = matches[0]
                            elif len(matches) > 1:
                                print(f"⚠️ Ambiguous call to '{child.spelling}'; candidates: {matches}")
                            else:
                                print(f"⚠️ Could not resolve call to '{child.spelling}' from '{ctx_func}'")
                        break

            # Record call edge
            if callee_name and ctx_func != callee_name:
                loc = cursor.location
                self.add_edge_strict(ctx_func, callee_name, 
                     edge_type=EdgeType.FUNCTION_CALL, 
                     location=(str(loc.file), loc.line), 
                     raw=raw_spelling)

        # --- references ---
        if cursor.kind == cindex.CursorKind.DECL_REF_EXPR and ctx_func:
            ref = cursor.referenced
            if ref:
                loc = cursor.location
                if ref.kind in FUNC_KINDS:
                    # function ref edge
                    callee = self._signature(ref) if ref.kind in CTOR_KINDS else self.fq_name(ref)
                    if callee != ctx_func:
                        self.add_edge_strict(ctx_func, callee, 
                                edge_type=EdgeType.REFERENCE, 
                                location=(str(loc.file), loc.line), 
                                raw=cursor.spelling)
                elif ref.kind in GLOBAL_KINDS:
                    # global variable ref edge
                    callee = self.fq_name(ref)
                    if callee != ctx_func:
                        self.add_edge_strict(ctx_func, callee, 
                                            edge_type=EdgeType.GLOBAL_REF, 
                                            location=(str(loc.file), loc.line), 
                                            raw=cursor.spelling)
        if cursor.kind == cindex.CursorKind.MEMBER_REF_EXPR and ctx_func:
                ref = cursor.referenced
                if ref:
                    loc = cursor.location
                    name = self.fq_name(ref)
                    self.add_edge_strict(ctx_func, name, 
                                        edge_type=EdgeType.MEMBER_REF, 
                                        location=(str(loc.file), loc.line), 
                                        raw=cursor.spelling)
        if cursor.kind == cindex.CursorKind.NAMESPACE_REF and ctx_func:
            ref = cursor.referenced
            if ref:
                loc = cursor.location
                name = self.fq_name(ref)
                self.add_edge_strict(ctx_func, name, 
                                    edge_type=EdgeType.GLOBAL_REF, 
                                    location=(str(loc.file), loc.line), 
                                    raw=cursor.spelling)


        # --- Type reference tracking
        if cursor.kind == cindex.CursorKind.TYPE_REF and ctx_func:
            decl = cursor.referenced or cursor.get_definition() or cursor.get_type().get_declaration()
            if not decl:
                return
            if decl.kind in TEMPLATE_PARAM_KINDS:
                return

            type_name = self.fq_name(decl)
            loc = cursor.location
            if type_name != ctx_func:
                self.add_edge_strict(ctx_func, type_name, edge_type=EdgeType.TYPE_REF, location=(str(loc.file), loc.line), raw=cursor.spelling)

        # Recurse into children
        for child in cursor.get_children():
            self.collect_edges(child, ctx_func)


    # ──────────────────── Parsing helpers ──────────────────────────────

    COMMON_COMPILER_ARGS = [
        '-x', 'c++', '-std=c++17',
        '-I/usr/include', '-I/usr/local/include',
        '-I.', '-Isrc', '-Iinclude', '-I..',
        '-fparse-all-comments',
        '-Wno-unknown-warning-option',
        '-ferror-limit=0',
        '-D__clang_analyzer__',
        '-ftemplate-depth=1024',
        '-finstantiate-templates',
    ]

    def make_include_args(self, p: Path) -> List[str]:
        args = []
        src_dir = str(p.parent)
        args.append(f'-I{src_dir}')
        parent = p.parent.parent
        if parent.exists():
            args.append(f'-I{parent}')
            for sub in ("include","src","lib"):
                d = parent / sub
                if d.is_dir():
                    args.append(f'-I{d}')
        return args

    def parse_file(self, index: cindex.Index, p: Path) -> Optional[cindex.TranslationUnit]:
        try:
            args = self.COMMON_COMPILER_ARGS + self.make_include_args(p)
            return index.parse(str(p), args=args,
                            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        except Exception as e:
            print(f"Error parsing {p}: {e}")
            return None

    # ──────────────────── File discovery ───────────────────────────────

    def generate_uber_file(self, output_file: Path = None) -> Path:
        if output_file is None:
            output_file = self.repo_path / "__uber.cpp"

        includes = []
        for ext in (".hpp", ".h", ".cpp", ".cxx", ".cc"):
            for file in self.repo_path.rglob(f"*{ext}"):
                if "test" in file.parts:  # optional: skip tests
                    continue
                rel_path = file.relative_to(self.repo_path)
                includes.append(f'#include "{rel_path}"')

        content = "// AUTO-GENERATED UBER FILE\n" + "\n".join(sorted(includes)) + "\n"
        output_file.write_text(content)
        print(f"✅ Uber file generated at {output_file}")
        return output_file

    def parse_uber_file(self, uber_file: Path) -> Optional[cindex.TranslationUnit]:
        args = self.COMMON_COMPILER_ARGS + self.make_include_args(uber_file)
        try:
            return self.index.parse(str(uber_file), args=args,
                            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        except Exception as e:
            print(f"❌ Failed to parse uber file: {e}")
            return None


    def cpp_files(self) -> List[Path]:
        return [Path(dp) / f for dp, _, files in os.walk(self.repo_path) for f in files if f.endswith((".cpp", ".cc", ".cxx", ".c", ".hpp", ".h"))]

    # ──────────────────── Build graph ─────────────────────────────────

    def _close_graph(self) -> None:
        # if we have 2 nodes `name` and `name::name`, merge into `name::name`
        for n in list(self.graph.nodes):
            if "::" in n:
                split = n.split("::")
                # Check if all elements of split are the same
                if len(set(split)) == 1:
                    base = split[-1]
                    if base in self.graph.nodes:
                        print(f"Merging node {base} into {n}")
                        self.graph = nx.contracted_nodes(self.graph, n, base, self_loops=False)
        # make all types who point to variables, point to the variables' succesors
        for n in list(self.graph.nodes):
            if self.graph.nodes[n]["kind"] & NodeKind.TYPE:
                for succ in list(self.graph.successors(n)):
                    if is_kind(self.graph.nodes[succ]["kind"], NodeKind.VARIABLE):
                        for succ_succ in list(self.graph.successors(succ)):
                            print(f"Adding edge from {n} to {succ_succ}")
                            self.add_edge_strict(n, succ_succ, edge_type=EdgeType.DEPENDENCY)

    def _build_graph(self) -> nx.DiGraph:
        self.graph = nx.DiGraph()    
        uber_file = self.generate_uber_file()
        tu = self.parse_uber_file(uber_file)    
        self.collect_nodes(tu.cursor)
        print(f"Found {len(self.graph.nodes)} nodes")
        self.collect_edges(tu.cursor, None)
        print(f"Found {len(self.graph.edges)} edges")
        # Remove the uber file node
        if uber_file.name in self.graph:
            self.graph.remove_node(uber_file.name)
        # Delete the uber file
        try:
            os.remove(uber_file)
        except Exception as e:
            print(f"❌ Failed to remove uber file: {e}")
        # if we have 2 nodes `name` and `name::name`, merge into `name::name`
        self._close_graph()
    
    def induce_subgraph(
        self,
        node_filter: Optional[callable] = None,
        edge_filter: Optional[callable] = None
    ) -> nx.DiGraph:
        """
        Induces a subgraph based on user-defined filters for nodes and edges.
    
        Args:
            node_filter (callable, optional): A function that takes a node and its attributes
                and returns True if the node should be included in the subgraph.
                Example: lambda node, attrs: attrs["kind"] & NodeKind.FUNCTION and attrs["kind"] & NodeKind.IN_CODEBASE
            edge_filter (callable, optional): A function that takes two nodes (u, v) and edge attributes
                and returns True if the edge should be included in the subgraph.
                Example: lambda u, v, attrs: attrs["type"] & EdgeType.FUNCTION_CALL
    
        Returns:
            nx.DiGraph: The induced subgraph containing only the filtered nodes and edges.
        """
        # Filter nodes
        if node_filter:
            filtered_nodes = [
                node for node, attrs in self.graph.nodes(data=True) if node_filter(node, attrs)
            ]
        else:
            filtered_nodes = list(self.graph.nodes)
    
        # Create a subgraph with the filtered nodes
        induced_graph = self.graph.subgraph(filtered_nodes).copy()
    
        # Filter edges
        if edge_filter:
            edges_to_remove = [
                (u, v) for u, v, attrs in induced_graph.edges(data=True) if not edge_filter(u, v, attrs)
            ]
            induced_graph.remove_edges_from(edges_to_remove)
    
        return induced_graph

    def plot(self, horizontal=False, node_filter: Optional[callable] = None, edge_filter: Optional[callable] = None):
        filtered_graph = self.induce_subgraph(node_filter, edge_filter)

        # --- Level Computation ---
        levels = {}

        if nx.is_directed_acyclic_graph(filtered_graph):
            # Regular topological sort
            for node in nx.topological_sort(filtered_graph):
                preds = list(filtered_graph.predecessors(node))
                levels[node] = 0 if not preds else 1 + max(levels[p] for p in preds)
        else:
            print("⚠️ Graph is not a DAG. Using SCC condensation for level estimation.")
            cycle = list(nx.simple_cycles(filtered_graph))
            if cycle:
                print(f"⚠️ Found cycles: {cycle}")
            sccs = list(nx.strongly_connected_components(filtered_graph))
            scc_map = {node: idx for idx, comp in enumerate(sccs) for node in comp}
            condensed = nx.condensation(filtered_graph, sccs)

            scc_levels = {}
            for scc_node in nx.topological_sort(condensed):
                preds = list(condensed.predecessors(scc_node))
                scc_levels[scc_node] = 0 if not preds else 1 + max(scc_levels[p] for p in preds)

            # Map back to node-level
            for node in filtered_graph.nodes:
                scc_index = scc_map[node]
                levels[node] = scc_levels[scc_index]

        # --- Plotting ---
        level_nodes = defaultdict(list)
        for node, lvl in levels.items():
            level_nodes[lvl].append(node)

        pos = {}
        max_level = max(level_nodes.keys()) if level_nodes else 0
        max_nodes_per_level = max(len(nodes) for nodes in level_nodes.values()) if level_nodes else 0

        horizontal_spacing = 2.0
        vertical_spacing = 10

        for level, nodes in level_nodes.items():
            for i, node in enumerate(nodes):
                offset = (max_nodes_per_level - len(nodes)) / 2
                if horizontal:
                    pos[node] = (level * horizontal_spacing, -(i + offset) * vertical_spacing)
                else:
                    pos[node] = ((i + offset) * horizontal_spacing, -level * vertical_spacing)

        node_colors = []
        for node in filtered_graph.nodes:
            kind = filtered_graph.nodes[node].get("kind", NodeKind.UNKNOWN)
            if kind & NodeKind.FUNCTION:
                color = "lightblue" 
                if kind & NodeKind.CONSTRUCTOR:
                    color = "skyblue"
                elif kind & NodeKind.DESTRUCTOR:
                    color = "royalblue"
            elif kind & NodeKind.TYPE:
                color = "orange"
            elif kind & NodeKind.VARIABLE:
                color = "lightgreen"
            else:
                color = "gray"
            node_colors.append(color)

        edge_colors = []
        for u, v, data in filtered_graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            if edge_type & EdgeType.CALL:
                color = "blue"
            elif edge_type & EdgeType.REFERENCE:
                color = "green"
            elif edge_type & EdgeType.DEPENDENCY:
                color = "orange"
            else:
                color = "gray"
            edge_colors.append(color)

        plt.figure(figsize=(16, 10))
        nx.draw_networkx_edges(
            filtered_graph, pos, arrowstyle='-|>', arrowsize=10,
            edge_color=edge_colors, width=0.8
        )
        nx.draw_networkx_nodes(filtered_graph, pos, node_size=400, node_color=node_colors)
        nx.draw_networkx_labels(filtered_graph, pos, font_size=7, font_family='monospace')

        plt.title("Code Graph Ordered by Levels (SCC-aware)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()