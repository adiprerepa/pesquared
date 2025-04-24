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
from utils.git_utils import clone_repo, get_current_branch
import pickle
from analyzers.stack_analyzer import StackAnalyzer, FunctionStats

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

    # Inheritance
    INHERITS = TYPE_REF | 1 << 13   # Inheritance relationship
    
    # Additional properties
    DIRECT = 1 << 24    # Direct reference in source
    INDIRECT = 1 << 25  # Indirect reference (e.g., via typedef)

def is_kind(node, kind):
    return (node & kind) == kind

class CodeAnalyzer:
    lib_subgraph : Optional[nx.DiGraph] = None
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()
        self.graph = nx.DiGraph()
        self._file_cache: Dict[str, List[str]] = {}
        self._init_clang()
        self.index = cindex.Index.create()
        self.branch = get_current_branch(self.repo_path)
        self._build_graph()
    
    def __init__(self, url: str, branch: str):
        repo = clone_repo(url, branch)
        self.repo_path = Path(repo.working_dir).resolve()
        self.graph = nx.DiGraph()
        self._file_cache: Dict[str, List[str]] = {}
        self._init_clang()
        self.index = cindex.Index.create()
        self.branch = get_current_branch(self.repo_path)
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

    # ─── Helper to build fully-qualified name with signature ───
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

        # Handle global namespace case
        if not prefix:
            prefix = ""

        # Destructor: no args
        if cur.kind == cindex.CursorKind.DESTRUCTOR:
            base = cur.semantic_parent.spelling
            name = f"{prefix}::~{base}()" if prefix else f"~{base}()"
        else:
            # Constructor: include parameter types
            base = cur.spelling  # class name
            params = ", ".join(arg.type.spelling for arg in cur.get_arguments())
            name = f"{prefix}::{base}({params})" if prefix else f"{base}({params})"
        return name
                

    # ─── Updated collect_nodes ───
    def collect_nodes(self, cursor: cindex.Cursor) -> None:
        """Collect nodes from the AST and add them to the graph."""
        if self.lib_subgraph and cursor.location.file and cursor.location.file.name in self.lib_subgraph.graph['files']:
            return
        # Process current cursor
        kind, name = self.determine_node_kind(cursor)

        # Only add nodes from our project or with valid kinds
        if kind != NodeKind.UNKNOWN and name:
            self.add_node_to_graph(name, kind, cursor)

        # Recursively process children
        for child in cursor.get_children():
            self.collect_nodes(child)

    def in_proj(self, c: cindex.Cursor) -> bool:
        """Check if a cursor is within the project."""
        return bool(c.location.file) and Path(c.location.file.name).resolve().is_relative_to(self.repo_path)

    def add_node_to_graph(self, name: str, kind: NodeKind, cursor: Optional[cindex.Cursor]) -> None:
        """Add a node to the graph."""
        # Check if node doesn't exist or if new AST is more detailed
        ast_outline = "\n".join(self.outline_ast(cursor)) if cursor else ""
        if name not in self.graph or len(self.graph.nodes[name].get('ast', '')) < len(ast_outline):
            # Basic node properties
            attrs = {"kind": kind, "code": "", "ast": "", "cursor": None}

            # Add location info if available
            if cursor:
                attrs.update(loc_info(cursor))
                attrs["code"] = self.read_extent(cursor)
                attrs["ast"] = ast_outline
                attrs["cursor"] = cursor

            self.graph.add_node(name, **attrs)

    def determine_node_kind(self, cur: cindex.Cursor) -> tuple[NodeKind, str]:
        """Determine the node kind and fully qualified name for a cursor."""
        base_kind = NodeKind.UNKNOWN
        name = ""

        # Set origin flag
        origin_flag = NodeKind.IN_CODEBASE if self.in_proj(cur) else NodeKind.LIBRARY

        # Functions, methods, constructors, destructors
        if cur.kind in FUNC_KINDS:
            if cur.kind in CTOR_KINDS:
                name = self._signature(cur)
                base_kind = NodeKind.CONSTRUCTOR if cur.kind != cindex.CursorKind.DESTRUCTOR else NodeKind.DESTRUCTOR
            else:
                name = self._signature(cur)
                base_kind = NodeKind.METHOD if '::' in name else NodeKind.FREE_FUNCTION

        # Member variables
        elif cur.kind in MEMBER_KINDS:
            name = self.fq_name(cur)
            base_kind = NodeKind.MEMBER_VAR

        # Global variables
        elif cur.kind in GLOBAL_KINDS and cur.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT:
            name = self.fq_name(cur)
            base_kind = NodeKind.GLOBAL_VAR

        # Types
        elif cur.kind in TYPE_KINDS:
            name = self.fq_name(cur)
            base_kind = NodeKind.CLASS if cur.kind in {cindex.CursorKind.CLASS_DECL,
                                                       cindex.CursorKind.CLASS_TEMPLATE} else NodeKind.STRUCT

        return base_kind | origin_flag, name

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
        # Process different edge types based on cursor kind
        if ctx_func:
            match cursor.kind:
                case cindex.CursorKind.CALL_EXPR:
                    self._process_function_call(cursor, ctx_func)
                case cindex.CursorKind.DECL_REF_EXPR | cindex.CursorKind.FIELD_DECL:
                    self._process_reference(cursor, ctx_func)
                case cindex.CursorKind.MEMBER_REF_EXPR:
                    self._process_member_reference(cursor, ctx_func)
                case cindex.CursorKind.NAMESPACE_REF:
                    self._process_namespace_reference(cursor, ctx_func)
                case cindex.CursorKind.TYPE_REF:
                    self._process_type_reference(cursor, ctx_func)
                case cindex.CursorKind.CXX_BASE_SPECIFIER:
                    self._process_inheritance(cursor, ctx_func)

        # Update function context if we're in a definition
        ctx_func = self._update_context(cursor, ctx_func)

        # Recurse into children
        for child in cursor.get_children():
            self.collect_edges(child, ctx_func)

    def _update_context(self, cursor: cindex.Cursor, ctx_func: Optional[str]) -> Optional[str]:
        """Update the current function context based on cursor position"""
        if cursor.is_definition():
            if cursor.kind in FUNC_KINDS:
                return self._signature(cursor) if cursor.kind in CTOR_KINDS else self._signature(cursor)
            elif cursor.kind in TYPE_KINDS:
                return self.fq_name(cursor)
            elif cursor.kind in MEMBER_KINDS:
                return self.fq_name(cursor)
            elif cursor.kind in GLOBAL_KINDS and cursor.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT:
                return self.fq_name(cursor)
        return ctx_func

    def _process_function_call(self, cursor: cindex.Cursor, ctx_func: str) -> None:
        """Process function call expressions"""
        callee_name = None
        raw_spelling = ""

        # Direct reference to function being called
        if cursor.referenced and cursor.referenced.kind in FUNC_KINDS:
            if cursor.referenced.kind in CTOR_KINDS:
                callee_name = self._signature(cursor.referenced)
            else:
                callee_name = self._signature(cursor.referenced)
            raw_spelling = cursor.referenced.spelling or ""
        
        # Fallback: search children for function name
        if not callee_name:
            callee_name = self._find_function_in_children(cursor, ctx_func)
            
        # Record call edge
        if callee_name and ctx_func != callee_name:
            self.add_edge_strict(
                ctx_func, callee_name, 
                edge_type=EdgeType.FUNCTION_CALL, 
                location=(str(cursor.location.file), cursor.location.line), 
                cursor = cursor.referenced
            )

    def _find_function_in_children(self, cursor: cindex.Cursor, ctx_func: str) -> Optional[str]:
        """Find function name in children of a call expression"""
        for child in cursor.get_children():
            if (child.kind == cindex.CursorKind.DECL_REF_EXPR and child.spelling and
                (not child.referenced or child.referenced.kind in FUNC_KINDS)):
                raw_spelling = child.spelling
                
                if child.referenced and child.referenced.kind in FUNC_KINDS:
                    return self.fq_name(child.referenced)
                
                # Try to resolve by name
                variants = self.base_name_variants(child.spelling)
                matches = [
                    n for n, d in self.graph.nodes(data=True)
                    if d.get("kind") & NodeKind.FUNCTION and variants & self.base_name_variants(n)
                ]
                
                if len(matches) == 1:
                    return matches[0]
                elif len(matches) > 1:
                    print(f"⚠️ Ambiguous call to '{child.spelling}'; candidates: {matches}")
                else:
                    print(f"⚠️ Could not resolve call to '{child.spelling}' from '{ctx_func}'")
                
                return None
        return None

    def _process_reference(self, cursor: cindex.Cursor, ctx_func: str) -> None:
        """Process declaration references"""
        ref = cursor.referenced
        if not ref:
            return
            
        loc = cursor.location
        
        if ref.kind in FUNC_KINDS:
            callee = self._signature(ref) if ref.kind in CTOR_KINDS else self.fq_name(ref)
            if callee != ctx_func:
                self.add_edge_strict(
                    ctx_func, callee, 
                    edge_type=EdgeType.REFERENCE, 
                    location=(str(loc.file), loc.line), 
                    cursor=cursor
                )
        elif ref.kind in GLOBAL_KINDS:
            callee = self.fq_name(ref)
            if callee != ctx_func:
                self.add_edge_strict(
                    ctx_func, callee, 
                    edge_type=EdgeType.GLOBAL_REF, 
                    location=(str(loc.file), loc.line), 
                    cursor=ref
                )
        elif ref.kind in MEMBER_KINDS:
            callee = self.fq_name(ref)
            if callee != ctx_func:
                self.add_edge_strict(
                    ctx_func, callee, 
                    edge_type=EdgeType.MEMBER_REF, 
                    location=(str(loc.file), loc.line), 
                    cursor=ref
                )

    def _process_member_reference(self, cursor: cindex.Cursor, ctx_func: str) -> None:
        """Process member references"""
        ref = cursor.referenced
        if ref:
            loc = cursor.location
            name = self.fq_name(ref)
            self.add_edge_strict(
                ctx_func, name, 
                edge_type=EdgeType.MEMBER_REF, 
                location=(str(loc.file), loc.line), 
                cursor=ref
            )

    def _process_namespace_reference(self, cursor: cindex.Cursor, ctx_func: str) -> None:
        """Process namespace references"""
        ref = cursor.referenced
        if ref:
            loc = cursor.location
            name = self.fq_name(ref)
            self.add_edge_strict(
                ctx_func, name, 
                edge_type=EdgeType.GLOBAL_REF, 
                location=(str(loc.file), loc.line), 
                cursor=ref
            )

    def _process_type_reference(self, cursor: cindex.Cursor, ctx_func: str) -> None:
        """Process type references"""
        decl = cursor.referenced or cursor.get_definition() or cursor.get_type().get_declaration()
        if not decl or decl.kind in TEMPLATE_PARAM_KINDS:
            return
            
        type_name = self.fq_name(decl)
        if type_name != ctx_func:
            self.add_edge_strict(
                ctx_func, type_name, 
                edge_type=EdgeType.TYPE_REF, 
                location=(str(cursor.location.file), cursor.location.line), 
                cursor=decl
            )
    def _process_inheritance(self, cursor: cindex.Cursor, ctx_func: str) -> None:
        """Process inheritance references"""
        if cursor.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
            base = cursor.referenced
            if base and base.kind in TYPE_KINDS:
                name = self.fq_name(base)
                self.add_edge_strict(
                    ctx_func, name, 
                    edge_type=EdgeType.INHERITS, 
                    location=(str(cursor.location.file), cursor.location.line), 
                    cursor=base
                )


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
    
    def remove_unpickleable_attrs(self, graph):
        for node in graph.nodes:
            attrs = graph.nodes[node]
            keys_to_remove = []
            for key, val in attrs.items():
                try:
                    pickle.dumps(val)  # Test if it's pickleable
                except (pickle.PicklingError, TypeError, ValueError):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del attrs[key]
        for edge in graph.edges:
            attrs = graph.edges[edge]
            keys_to_remove = []
            for key, val in attrs.items():
                try:
                    pickle.dumps(val)  # Test if it's pickleable
                except (pickle.PicklingError, TypeError, ValueError):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del attrs[key]
        # Remove unpickleable attributes from the graph itself
        attrs = graph.graph
        keys_to_remove = []
        for key, val in attrs.items():
            try:
                pickle.dumps(val)  # Test if it's pickleable
            except (pickle.PicklingError, TypeError, ValueError):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del attrs[key]
        return graph


    def _build_graph(self) -> nx.DiGraph:
        self.graph = nx.DiGraph()    
        uber_file = self.generate_uber_file()
        tu = self.parse_uber_file(uber_file)
        lib_subgraph_path = self.repo_path / "lib_subgraph.gpickle"
        # Check if we have a lib_subgraph.gpickle
        if not self.lib_subgraph and lib_subgraph_path.exists():
            with open(lib_subgraph_path, "rb") as f:
                self.lib_subgraph = pickle.load(f)
                self.lib_subgraph.graph['files'] = set([attr.get('file', 'xyz') for n, attr in self.lib_subgraph.nodes(data=True)])
            print(f"✅ Library subgraph loaded from {lib_subgraph_path}")
        self.collect_nodes(tu.cursor)
        if self.lib_subgraph:
            # Merge the library subgraph into the main graph
            self.graph = nx.compose(self.graph, self.lib_subgraph)
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
        self._close_graph()
        if not self.lib_subgraph:
            # store the subgraph of library nodes on disk
            self.lib_subgraph = self.induce_subgraph(
                node_filter=lambda n, attrs: is_kind(attrs.get("kind", NodeKind.UNKNOWN), NodeKind.LIBRARY),
            )
            # Save the subgraph to disk
            with open(lib_subgraph_path, "wb") as f:
                pickle.dump(self.remove_unpickleable_attrs(self.lib_subgraph), f)
            print(f"✅ Library subgraph saved to {lib_subgraph_path}")
    
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
            edge_filter (callable, optional): A function that takes two nodes (u, v) and edge attributes
                and returns True if the edge should be included in the subgraph.
    
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