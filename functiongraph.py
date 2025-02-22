import networkx as nx
import os
import sys
from clang import cindex

class FunctionGraph(nx.DiGraph):
    def __init__(self, path):
        """
        Constructs a directed graph where nodes are function names and edges (a -> b)
        indicate that function 'a' calls function 'b'.
        Nodes are labeled as '{file_path}/Class::Function' or '{file_path}/Function'.
        """
        import extract_deps
        self.path = path
        self.index = cindex.Index.create()
        # mapping: fully qualified name -> list of node labels (each corresponding to one definition)
        self.definitions = {}
        super().__init__()
        self._build_graph()
    
    def _build_graph(self):
        # Extensions to consider as source files.
        source_extensions = ('.cpp', '.cc', '.cxx', '.h', '.hpp')
        # List to collect tuples of (node_label, function cursor)
        function_nodes = []
        
        # Walk the codebase directory.
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(source_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        # Parse the file using Clang. Add any compiler args you need.
                        tu = self.index.parse(file_path, args=['-std=c++17', '-I' + root])
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
                        continue
                    
                    # Extract function definitions from the translation unit.
                    self._extract_function_definitions(tu.cursor, file_path, function_nodes)
        
        # Add all found functions as nodes and record them by their fully qualified name.
        for node_label, cursor in function_nodes:
            function_body = self._get_function_body(cursor)
            self.add_node(node_label, body=function_body)
            fq_name = self.get_fully_qualified_name(cursor)
            if fq_name not in self.definitions:
                self.definitions[fq_name] = []
            self.definitions[fq_name].append(node_label)
        
        # Now, for each function, inspect its body for function calls and add edges accordingly.
        for node_label, cursor in function_nodes:
            called_functions = self._collect_called_functions(cursor)
            for called_fq_name in called_functions:
                # If a called function has been defined in the codebase (possibly in multiple files),
                # create an edge from the caller to each definition.
                if called_fq_name in self.definitions:
                    for target_node in self.definitions[called_fq_name]:
                        self.add_edge(node_label, target_node)
                # Uncomment the following to include external (undefined) functions:
                # else:
                #     ext_node = f"external/{called_fq_name}"
                #     self.add_node(ext_node)
                #     self.add_edge(node_label, ext_node)
    
    def _extract_function_definitions(self, cursor, file_path, function_nodes):
        """
        Recursively traverse the AST and collect function definitions (including methods).
        Only functions defined in the current file are collected.
        """
        from clang.cindex import CursorKind
        # Look for function or method declarations that are also definitions.
        if cursor.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD) and cursor.is_definition():
            # Ensure the function is defined in the file we are currently processing.
            if cursor.location and cursor.location.file:
                if os.path.abspath(cursor.location.file.name) == os.path.abspath(file_path):
                    fq_name = self.get_fully_qualified_name(cursor)
                    node_label = f"{file_path}/{fq_name}"
                    function_nodes.append((node_label, cursor))
        # Recurse into children.
        for child in cursor.get_children():
            self._extract_function_definitions(child, file_path, function_nodes)
    
    def _collect_called_functions(self, cursor):
        """
        Recursively collect fully qualified names of functions called within this function's AST.
        Returns a set of fully qualified names.
        """
        from clang.cindex import CursorKind
        called = set()
        # if its a function call, record the fully qualified name
        if cursor.kind == CursorKind.CALL_EXPR:
            if cursor.referenced is not None:
                fq_name = self.get_fully_qualified_name(cursor.referenced)
                if fq_name:
                    called.add(fq_name)
                else:
                    print(f"Warning: Unresolved function call at {cursor.location}")
        for child in cursor.get_children():
            called.update(self._collect_called_functions(child))
        return called

    def _get_function_body(self, cursor):
        """
        Get the source code of the function body.
        """
        extent = cursor.extent
        with open(cursor.location.file.name, 'r') as f:
            lines = f.readlines()
        return ''.join(lines[extent.start.line - 1:extent.end.line])

    @staticmethod
    def get_fully_qualified_name(cursor):
        """
        Get the fully qualified name of a function/method including namespaces and classes.
        """
        if not cursor:
            return ""
        components = []
        cur = cursor
        # Traverse up the AST, gathering namespaces and class/struct declarations.
        while cur and cur.kind.name != 'TRANSLATION_UNIT':
            if cur.kind.name in ['NAMESPACE', 'CLASS_DECL', 'STRUCT_DECL']:
                if cur.spelling:
                    components.insert(0, cur.spelling)
            cur = cur.semantic_parent
        # Append the function/method name.
        if cursor.spelling:
            components.append(cursor.spelling)
        return "::".join(components)

# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print("Usage: python function_graph.py <codebase_directory>")
#         sys.exit(1)
    
#     codebase_dir = sys.argv[1]
#     graph = FunctionGraph(codebase_dir)
#     print("Graph nodes:")
#     for node in graph.nodes:
#         print(node)
#     print("\nGraph edges:")
#     for edge in graph.edges:
#         print(edge)
