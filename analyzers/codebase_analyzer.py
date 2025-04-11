import os
import sys
import networkx as nx
from typing import Dict, Set, List, Tuple, Optional
from clang import cindex
from .dependency_extractor import (
    setup_clang, find_function_in_cursor, collect_dependencies,
    demangle_name, get_fully_qualified_name, is_mangled_name
)

class CodebaseAnalyzer:
    def __init__(self):
        # Initialize clang
        self.clang_initialized = setup_clang()
        if not self.clang_initialized:
            raise RuntimeError("Failed to initialize clang library")
        self.index = cindex.Index.create()
        
    def analyze_codebase(self, codebase_dir: str) -> nx.DiGraph:
        """
        Analyze a codebase and return a directed graph of function dependencies.
        
        Args:
            codebase_dir: Directory containing the codebase to analyze
            
        Returns:
            nx.DiGraph: A directed graph where nodes are functions (identified by mangled name)
                        and edges represent function calls
        """
        # Data structures to track functions and their dependencies
        function_graph = nx.DiGraph()
        processed_files = set()
        
        # Scan codebase for all files
        files_to_process = self._find_source_files(codebase_dir)
        print(f"Found {len(files_to_process)} source files to analyze")
        total_files = len(files_to_process)
        
        print(f"Found {total_files} source files to analyze")
        
        # Parse each file and collect functions and dependencies
        for i, file_path in enumerate(files_to_process, 1):
            if i % 10 == 0 or i == total_files:
                print(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
            
            self._process_file(file_path, function_graph)
        
        print(f"Analysis complete: {len(function_graph.nodes())} functions, {len(function_graph.edges())} dependencies")
        return function_graph
        
    def _find_source_files(self, dir_path: str) -> List[str]:
        """Find all relevant source files in the codebase."""
        files = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith(('.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx')):
                    files.append(os.path.join(root, filename))
        return files
    
    def _process_file(self, file_path: str, function_graph: nx.DiGraph):
        """Process a single file and add its functions and dependencies to the graph."""
        try:
            # Parse the file using clang
            tu = self._parse_file(file_path)
            if not tu:
                return
            
            # Find all function declarations and definitions
            functions = self._find_all_functions(tu.cursor)
            
            # For each function, collect its dependencies and add to the graph
            for func in functions:
                # Get the function's mangled name (unique identifier)
                mangled_name = get_fully_qualified_name(func)
                
                # Get a readable name for display (demangled)
                readable_name = demangle_name(mangled_name) if is_mangled_name(mangled_name) else mangled_name
                
                # Add the function to the graph
                function_graph.add_node(
                    mangled_name,  # Use mangled name as node ID
                    name=readable_name,
                    location=f"{func.location.file}:{func.location.line}" if func.location.file else "unknown"
                )
                
                # Collect dependencies (function calls)
                deps = collect_dependencies(func)
                
                # Add edges for each function call
                for called_func in deps["functions"]:
                    called_name = called_func["name"]
                    # Add edge from this function to the called function
                    function_graph.add_edge(
                        mangled_name,  # Source: current function
                        called_name,   # Target: called function
                        signature=called_func.get("signature", "")
                    )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _parse_file(self, file_path: str) -> Optional[cindex.TranslationUnit]:
        """Parse a source file using clang."""
        try:
            compiler_args = [
                '-x', 'c++',       # Force C++ mode
                '-std=c++17',      # Use C++17
                '-I/usr/include',  # System includes
                '-I/usr/local/include',
                '-I.',             # Current directory
                '-Isrc',           # Common source directory
                '-Iinclude',       # Another common source directory 
                '-I..',            # Parent directory
                '-fparse-all-comments',  # Parse all comments
                '-Wno-unknown-warning-option',
                '-ferror-limit=0', 
                '-ftemplate-depth=1024',
            ]
            
            # Add the source directory and its parent to include paths
            src_dir = os.path.dirname(file_path)
            if src_dir:
                compiler_args.extend([f'-I{src_dir}'])
                parent_dir = os.path.dirname(src_dir)
                if parent_dir:
                    compiler_args.extend([f'-I{parent_dir}'])
                    
                    # Add potential include directories
                    potential_dirs = ['include', 'src', 'lib']
                    for p_dir in potential_dirs:
                        full_path = os.path.join(parent_dir, p_dir)
                        if os.path.isdir(full_path):
                            compiler_args.extend([f'-I{full_path}'])
            
            # Parse the file
            return self.index.parse(
                file_path,
                args=compiler_args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD | 
                       cindex.TranslationUnit.PARSE_INCOMPLETE
            )
        except Exception as e:
            print(f"Failed to parse {file_path}: {e}")
            return None
    
    def _find_all_functions(self, cursor) -> List:
        """Find all function declarations and definitions in a translation unit."""
        functions = []
        
        def visit(cursor):
            if cursor.kind in (
                cindex.CursorKind.FUNCTION_DECL,
                cindex.CursorKind.CXX_METHOD,
                cindex.CursorKind.FUNCTION_TEMPLATE
            ) and cursor.is_definition():
                functions.append(cursor)
            
            for child in cursor.get_children():
                visit(child)
        
        visit(cursor)
        return functions

    def visualize_graph(self, graph: nx.DiGraph, output_path: str = None):
        """
        Visualize the function dependency graph.
        
        Args:
            graph: The directed graph to visualize
            output_path: Path to save the visualization (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(graph)
            
            # Draw nodes with labels based on readable names
            node_labels = {node: graph.nodes[node].get('name', node) for node in graph.nodes()}
            nx.draw_networkx_nodes(graph, pos, node_size=100, alpha=0.8)
            nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)
            nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            
            plt.title(f"Function Dependency Graph - {len(graph.nodes())} functions")
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except ImportError:
            print("Visualization requires matplotlib. Install using: pip install matplotlib")
            
    def export_graph(self, graph: nx.DiGraph, output_path: str):
        """
        Export the function dependency graph to a file format.
        
        Args:
            graph: The directed graph to export
            output_path: Path to save the exported graph
        """
        # Determine export format based on file extension
        extension = os.path.splitext(output_path)[1].lower()
        
        if extension == '.gexf':
            nx.write_gexf(graph, output_path)
        elif extension == '.graphml':
            nx.write_graphml(graph, output_path)
        elif extension == '.gml':
            nx.write_gml(graph, output_path)
        elif extension == '.json':
            # Custom JSON export with readable names
            import json
            
            data = {
                'nodes': [],
                'links': []
            }
            
            # Add nodes
            for node in graph.nodes():
                node_data = graph.nodes[node]
                data['nodes'].append({
                    'id': node,
                    'name': node_data.get('name', node),
                    'location': node_data.get('location', 'unknown')
                })
            
            # Add edges
            for source, target in graph.edges():
                edge_data = graph.edges[(source, target)]
                data['links'].append({
                    'source': source,
                    'target': target,
                    'signature': edge_data.get('signature', '')
                })
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Default to adjacency list
            with open(output_path, 'w') as f:
                for node in graph.nodes():
                    f.write(f"{graph.nodes[node].get('name', node)}:\n")
                    for _, neighbor in graph.out_edges(node):
                        f.write(f"  -> {graph.nodes[neighbor].get('name', neighbor)}\n")

def analyze_codebase(codebase_dir: str) -> nx.DiGraph:
    """
    Analyze a codebase and return a directed graph of function dependencies.
    
    Args:
        codebase_dir: Directory containing the codebase to analyze
        
    Returns:
        nx.DiGraph: A directed graph where nodes are functions (identified by mangled name)
                    and edges represent function calls
    """
    analyzer = CodebaseAnalyzer()
    return analyzer.analyze_codebase(codebase_dir)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python codebase_analyzer.py <codebase_directory> [output_file]")
        sys.exit(1)
    
    codebase_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = CodebaseAnalyzer()
    graph = analyzer.analyze_codebase(codebase_dir)
    
    print(f"Analysis complete: {len(graph.nodes())} functions, {len(graph.edges())} dependencies")
    
    if output_file:
        analyzer.export_graph(graph, output_file)
        print(f"Graph exported to {output_file}")
    else:
        analyzer.visualize_graph(graph)
