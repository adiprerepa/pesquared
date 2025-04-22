import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import re

import networkx as nx
from clang import cindex

from collections import defaultdict
import matplotlib.pyplot as plt


class CodeAnalyzer:
    FUNC_KINDS = {
        cindex.CursorKind.FUNCTION_DECL,
        cindex.CursorKind.CXX_METHOD,
        cindex.CursorKind.FUNCTION_TEMPLATE,
    }

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

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self._init_clang()
        self.file_cache: Dict[str, List[str]] = {}
        self.index = cindex.Index.create()
        self.graph = nx.DiGraph()
        self.build()

    def _init_clang(self):
        if hasattr(self, "_clang_initialized") and self._clang_initialized:
            return
        for p in [
            "/usr/lib/llvm-18/lib", "/usr/lib/llvm-17/lib", "/usr/lib/llvm-16/lib",
            "/usr/lib/llvm-15/lib", "/usr/lib64/llvm", "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/llvm/lib",
        ]:
            lib = Path(p) / "libclang.so"
            if lib.exists():
                try:
                    cindex.Config.set_library_file(str(lib))
                    cindex.Config.set_compatibility_check(False)
                    _ = cindex.Index.create()
                    self._clang_initialized = True
                    print(f"✅ libclang loaded from: {lib}")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load {lib}: {e}")
        raise RuntimeError("libclang.so not found - install clang (libclang-dev)")

    @staticmethod
    def is_mangled(name: str) -> bool:
        return name.startswith("_Z")

    @staticmethod
    def demangle(name: str) -> str:
        if not CodeAnalyzer.is_mangled(name):
            return name
        try:
            out = subprocess.run(["c++filt", name], capture_output=True, text=True)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except FileNotFoundError:
            pass
        return name

    @staticmethod
    def base_name_variants(name: str) -> set:
        variants = {name}
        name_no_ret = name.split()[-1]
        variants.add(name_no_ret)
        no_args = re.sub(r'\(.*\)', '', name_no_ret)
        variants.add(no_args)
        base = no_args.split("::")[-1]
        variants.add(base)
        no_template = re.sub(r'<.*?>', '', base)
        variants.add(no_template)
        return {v.strip() for v in variants if v.strip()}

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
        name = "::".join(parts)
        return self.demangle(name) if self.is_mangled(name) else name

    def read_extent(self, cur: cindex.Cursor) -> str:
        ext = cur.extent
        if not ext or not ext.start.file:
            return ""
        fn = ext.start.file.name
        if fn not in self.file_cache:
            try:
                with open(fn, "r", encoding="utf-8", errors="ignore") as f:
                    self.file_cache[fn] = f.readlines()
            except:
                self.file_cache[fn] = []
        lines = self.file_cache[fn]
        return ''.join(lines[ext.start.line - 1:ext.end.line])

    def outline_ast(self, cur: cindex.Cursor, depth=0) -> List[str]:
        lines = ["  " * depth + cur.kind.name]
        for c in cur.get_children():
            lines.extend(self.outline_ast(c, depth + 1))
        return lines

    def loc_info(self, cur: cindex.Cursor) -> dict:
        loc = cur.location
        return {"file": str(loc.file) if loc.file else "", "line": loc.line, "col": loc.column}

    def make_include_args(self, path: Path) -> List[str]:
        args = []
        src_dir = path.parent
        args.append(f'-I{src_dir}')
        parent = src_dir.parent
        if parent.exists():
            args.append(f'-I{parent}')
            for sub in ("include", "src", "lib"):
                d = parent / sub
                if d.is_dir():
                    args.append(f'-I{d}')
        return args

    def parse_file(self, path: Path) -> Optional[cindex.TranslationUnit]:
        try:
            args = self.COMMON_COMPILER_ARGS + self.make_include_args(path)
            return self.index.parse(str(path), args=args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return None

    def in_proj(self, c: cindex.Cursor) -> bool:
        return bool(c.location.file) and Path(c.location.file.name).resolve().is_relative_to(self.project_root)

    def collect_nodes(self, cursor: cindex.Cursor):
        def ensure_node(name: str, kind: str, cur: Optional[cindex.Cursor]):
            if name not in self.graph:
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

        if cursor.kind in self.FUNC_KINDS and cursor.is_definition() and self.in_proj(cursor):
            fq = self.fq_name(cursor)
            ensure_node(fq, "function", cursor)

        if cursor.kind in self.TYPE_KINDS and self.in_proj(cursor):
            fq = self.fq_name(cursor)
            ensure_node(fq, "type", cursor)

        for child in cursor.get_children():
            self.collect_nodes(child)

    def collect_edges(self, cursor: cindex.Cursor, ctx_func: Optional[str]):
        if cursor.kind in self.FUNC_KINDS and cursor.is_definition() and self.in_proj(cursor):
            ctx_func = self.fq_name(cursor)

        if cursor.kind == cindex.CursorKind.CALL_EXPR and ctx_func:
            callee_name = None
            raw_spelling = ""

            if cursor.referenced and cursor.referenced.kind in self.FUNC_KINDS:
                callee_name = self.fq_name(cursor.referenced)
                raw_spelling = cursor.referenced.spelling or ""

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

            if callee_name and ctx_func != callee_name:
                loc = cursor.location
                location = (str(loc.file), loc.line)
                self.graph.add_edge(ctx_func, callee_name, type="call", location=location, raw=raw_spelling)

        if cursor.kind == cindex.CursorKind.TYPE_REF and ctx_func and self.in_proj(cursor):
            decl = cursor.referenced or cursor.get_definition() or cursor.get_type().get_declaration()
            if not decl or decl.kind in self.TEMPLATE_PARAM_KINDS:
                return
            if self.in_proj(decl):
                type_name = self.fq_name(decl)
                if type_name not in self.graph:
                    self.graph.add_node(
                        type_name, kind="type", code="", ast="", cursor=decl, **self.loc_info(decl)
                    )
                    print(f"⚠️ Missing node for type '{type_name}'")
                    print(f"Location: {self.loc_info(decl)}")
                loc = cursor.location
                self.graph.add_edge(
                    ctx_func, type_name, type="type-ref", location=(str(loc.file), loc.line)
                )

        for child in cursor.get_children():
            self.collect_edges(child, ctx_func)

    def cpp_files(self) -> List[Path]:
        return [Path(dp) / f for dp, _, files in os.walk(self.project_root)
                for f in files if f.endswith((".cpp", ".cc", ".cxx", ".c", ".hpp", ".h"))]

    def build(self) -> nx.DiGraph:
        tus = []
        for fp in sorted(self.cpp_files()):
            tu = self.parse_file(fp)
            if tu:
                tus.append(tu)
            else:
                print(f"⚠️ Failed to parse {fp}")
        for tu in tus:
            self.collect_nodes(tu.cursor)
        print(f"🔍 Found {len(self.graph.nodes)} nodes in {len(tus)} translation units")
        for tu in tus:
            self.collect_edges(tu.cursor, None)
        print(f"🔗 Found {len(self.graph.edges)} edges")
        self.clean_graph()
        return self.graph

    def clean_graph(self):
        for n in list(self.graph.nodes):
            if 'file' not in self.graph.nodes[n] or not self.graph.nodes[n]['file']:
                self.graph.remove_node(n)

    def plot_dag_by_levels(G, horizontal=False):
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Input graph must be a Directed Acyclic Graph (DAG)")

        # Compute node levels using topological sort
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else 1 + max(levels[p] for p in preds)

        # Group nodes by levels
        level_nodes = defaultdict(list)
        for node, lvl in levels.items():
            level_nodes[lvl].append(node)

        # Assign positions with increased spacing
        pos = {}
        max_level = max(level_nodes.keys()) if level_nodes else 0
        max_nodes_per_level = max(len(nodes) for nodes in level_nodes.values()) if level_nodes else 0
        
        # Define spacing factors
        horizontal_spacing = 2.0  # Increase for more horizontal space
        vertical_spacing = 1.5    # Increase for more vertical space
        
        for level, nodes in level_nodes.items():
            for i, node in enumerate(nodes):
                # Center nodes at each level horizontally
                offset = (max_nodes_per_level - len(nodes)) / 2
                if horizontal:
                    pos[node] = (level * horizontal_spacing, -(i + offset) * vertical_spacing)
                else:
                    pos[node] = ((i + offset) * horizontal_spacing, -level * vertical_spacing)

        # Define node colors based on their kind
        node_colors = []
        for node in G.nodes:
            kind = G.nodes[node].get("kind", "unknown")
            if kind == "function":
                node_colors.append("lightblue")
            elif kind == "class":
                node_colors.append("lightgreen")
            elif kind == "type":
                node_colors.append("orange")
            else:
                node_colors.append("gray")

        # Define edge colors based on their type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type", "unknown")
            if edge_type == "call":
                edge_colors.append("blue")
            elif edge_type == "type-ref":
                edge_colors.append("green")
            else:
                edge_colors.append("gray")

        # Draw
        plt.figure(figsize=(16, 10))  # Larger figure size
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            arrowstyle='-|>', 
            arrowsize=10, 
            edge_color=edge_colors,
            width=0.8
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=800,  # Smaller node size
            node_color=node_colors
        )
        
        # Draw labels with smaller font
        nx.draw_networkx_labels(
            G, pos, 
            font_size=7,  # Smaller font size
            font_family='monospace'
        )
        
        plt.title("DAG Ordered by Levels")
        plt.axis("off")
        plt.tight_layout()
