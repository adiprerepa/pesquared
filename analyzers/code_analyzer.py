import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Set
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from clang import cindex

_CLANG_INITIALIZED = False

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

    # ──────────────────── Traversal ────────────────────────────────────

    CTOR_KINDS = {
        cindex.CursorKind.CONSTRUCTOR,
        cindex.CursorKind.DESTRUCTOR,
    }

    FUNC_KINDS = {
        cindex.CursorKind.FUNCTION_DECL,
        cindex.CursorKind.CXX_METHOD,
        cindex.CursorKind.FUNCTION_TEMPLATE,
    } | CTOR_KINDS

    TYPE_KINDS = {
        cindex.CursorKind.CLASS_DECL,
        cindex.CursorKind.STRUCT_DECL,
        cindex.CursorKind.CLASS_TEMPLATE,
        cindex.CursorKind.TYPEDEF_DECL,
        cindex.CursorKind.TYPE_ALIAS_DECL,
        cindex.CursorKind.TYPE_ALIAS_TEMPLATE_DECL,
        cindex.CursorKind.ENUM_DECL,
    }

    TEMPLATE_PARAM_KINDS = {
        cindex.CursorKind.TEMPLATE_TYPE_PARAMETER,
        cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER,
        cindex.CursorKind.TEMPLATE_TEMPLATE_PARAMETER,
    }

    MEMBER_KINDS = {cindex.CursorKind.FIELD_DECL}
    GLOBAL_KINDS = {cindex.CursorKind.VAR_DECL}


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

    def loc_info(self, cur: cindex.Cursor) -> dict:
        loc = cur.location
        return {"file": str(loc.file) if loc.file else "", "line": loc.line, "col": loc.column}
                

    # ─── Updated collect_nodes ───
    def collect_nodes(self, cursor: cindex.Cursor) -> None:
        def in_proj(c: cindex.Cursor) -> bool:
            return bool(c.location.file) and Path(c.location.file.name).resolve().is_relative_to(self.repo_path)
        
        def ensure_node(name: str, kind: str, cur: Optional[cindex.Cursor]):
            if name not in self.graph or len(self.graph.nodes[name]['ast']) < len("\n".join(self.outline_ast(cur))):
                self.graph.add_node(name, kind=kind, code="", ast="", cursor=None, **(self.loc_info(cur) if cur else {}))
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
        if cursor.kind in self.FUNC_KINDS and in_proj(cursor):            
            if cursor.is_definition():
                if cursor.kind in self.CTOR_KINDS:
                    fq = self._signature(cursor)
                else:
                    fq = self.fq_name(cursor)
                ensure_node(fq, "function", cursor)
            else:
                if cursor.kind in self.CTOR_KINDS:
                    fq = self._signature(cursor)
                else:
                    fq = self.fq_name(cursor)
                if fq not in self.graph:
                    ensure_node(fq, "function", cursor)
        
        # --- member variables ---
        if cursor.kind in self.MEMBER_KINDS and in_proj(cursor):
            fq = self.fq_name(cursor)
            ensure_node(fq, "member", cursor)

        # --- global variables ---
        if cursor.kind in self.GLOBAL_KINDS and cursor.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT and in_proj(cursor):
            fq = self.fq_name(cursor)
            ensure_node(fq, "variable", cursor)

        # --- types ---
        if cursor.kind in self.TYPE_KINDS and in_proj(cursor):
            fq = self.fq_name(cursor)
            ensure_node(fq, "type", cursor)

        # recurse
        for child in cursor.get_children():
            self.collect_nodes(child)

    def add_edge_strict(self, u: str, v: str, **kwargs) -> None:
        if u == v:
            return
        if self.graph.has_edge(u, v):
            return
        if not self.graph.has_node(u):
            return
        if not self.graph.has_node(v):
            return
        else:
            self.graph.add_edge(u, v, **kwargs)


    def collect_edges(self, cursor: cindex.Cursor, ctx_func: Optional[str]) -> None:
        def in_proj(c: cindex.Cursor) -> bool:
            return bool(c.location.file) and Path(c.location.file.name).resolve().is_relative_to(self.repo_path)

        # Update context if we're inside a known function or template
        if cursor.kind in self.FUNC_KINDS and cursor.is_definition() and in_proj(cursor):
            if cursor.kind in self.CTOR_KINDS:
                ctx_func = self._signature(cursor)
            else:
                ctx_func = self.fq_name(cursor)

        if cursor.kind in self.TYPE_KINDS and cursor.is_definition() and in_proj(cursor):
            ctx_func = self.fq_name(cursor)
        
        if cursor.kind in self.MEMBER_KINDS and cursor.is_definition() and in_proj(cursor):
            ctx_func = self.fq_name(cursor)

        if cursor.kind in self.GLOBAL_KINDS and cursor.is_definition() and cursor.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT and in_proj(cursor):
            ctx_func = self.fq_name(cursor)

        # --- Function call detection
        if cursor.kind == cindex.CursorKind.CALL_EXPR and ctx_func:
            callee_name = None
            raw_spelling = ""

            # Direct reference to the function being called
            if cursor.referenced and cursor.referenced.kind in self.FUNC_KINDS:
                if cursor.referenced.kind in self.CTOR_KINDS:
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
                        (not child.referenced or child.referenced.kind in self.FUNC_KINDS)
                    ):
                        raw_spelling = child.spelling
                        if child.referenced and child.referenced.kind in self.FUNC_KINDS:
                            callee_name = self.fq_name(child.referenced)
                        else:
                            variants = self.base_name_variants(child.spelling)
                            matches = [
                                n for n, d in self.graph.nodes(data=True)
                                if d.get("kind") == "function" and variants & self.base_name_variants(n)
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
                self.add_edge_strict(ctx_func, callee_name, type="call", location=(str(loc.file), loc.line), raw=raw_spelling)

        # --- references ---
        if cursor.kind == cindex.CursorKind.DECL_REF_EXPR and ctx_func:
            ref = cursor.referenced
            if ref and in_proj(ref):
                loc = cursor.location
                if ref.kind in self.FUNC_KINDS:
                    # function ref edge
                    callee = self._signature(ref) if ref.kind in self.CTOR_KINDS else self.fq_name(ref)
                    if callee != ctx_func:
                        self.add_edge_strict(ctx_func, callee, type="ref", location=(str(loc.file), loc.line), raw=cursor.spelling)
                elif ref.kind in self.GLOBAL_KINDS:
                    # global variable ref edge
                    callee = self.fq_name(ref)
                    if callee != ctx_func:
                        self.add_edge_strict(ctx_func, callee, type="global-ref", location=(str(loc.file), loc.line), raw=cursor.spelling)
        if cursor.kind == cindex.CursorKind.MEMBER_REF_EXPR and ctx_func:
                ref = cursor.referenced
                if ref and in_proj(ref):
                    loc = cursor.location
                    name = self.fq_name(ref)
                    self.add_edge_strict(ctx_func, name, type="member-ref", location=(str(loc.file), loc.line), raw=cursor.spelling)
        if cursor.kind == cindex.CursorKind.NAMESPACE_REF and ctx_func:
            ref = cursor.referenced
            if ref and in_proj(ref):
                loc = cursor.location
                name = self.fq_name(ref)
                self.add_edge_strict(ctx_func, name, type="global-ref", location=(str(loc.file), loc.line), raw=cursor.spelling)


        # --- Type reference tracking
        if cursor.kind == cindex.CursorKind.TYPE_REF and ctx_func and in_proj(cursor):
            decl = cursor.referenced or cursor.get_definition() or cursor.get_type().get_declaration()
            if not decl:
                return
            if decl.kind in self.TEMPLATE_PARAM_KINDS:
                print(f"⚠️ Ignoring template parameter reference: {decl.spelling} at location {self.loc_info(decl)}")
                return

            if in_proj(decl):
                type_name = self.fq_name(decl)
                loc = cursor.location
                if type_name != ctx_func:
                    self.add_edge_strict(ctx_func, type_name, type="type-ref", location=(str(loc.file), loc.line), raw=cursor.spelling)

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

    # ──────────────────── Build graph ─────────────────────────────────-

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
        for n in list(self.graph.nodes):
            if "::" in n:
                split = n.split("::")
                # Check if all elements of split are the same
                if len(set(split)) == 1:
                    base = split[-1]
                    if base in self.graph.nodes:
                        print(f"Merging node {base} into {n}")
                        self.graph = nx.contracted_nodes(self.graph, n, base, self_loops=False)

    def plot(self, horizontal=False, exclude_node_kinds=None, exclude_edge_types=None):
        exclude_node_kinds = exclude_node_kinds or set()
        exclude_edge_types = exclude_edge_types or set()

        # Filter nodes and edges
        filtered_nodes = [
            node for node in self.graph.nodes
            if self.graph.nodes[node].get("kind", "unknown") not in exclude_node_kinds
        ]
        filtered_graph = self.graph.subgraph(filtered_nodes).copy()

        edges_to_remove = [
            (u, v) for u, v, data in filtered_graph.edges(data=True)
            if data.get("type", "unknown") in exclude_edge_types
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

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
            kind = filtered_graph.nodes[node].get("kind", "unknown")
            node_colors.append(
                "lightblue" if kind == "function" else "orange" if kind == "type" else "gray"
            )

        edge_colors = []
        for u, v, data in filtered_graph.edges(data=True):
            edge_type = data.get("type", "unknown")
            edge_colors.append(
                "blue" if edge_type == "call" else "green" if edge_type == "type-ref" else "gray"
            )

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