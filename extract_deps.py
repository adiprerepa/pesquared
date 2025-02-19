import os
import sys
from clang import cindex
import tiktoken

_CLANG_INITIALIZED = False

def setup_clang():
    """Setup clang library path and configuration."""
    global _CLANG_INITIALIZED
    
    if _CLANG_INITIALIZED:
        return True
        
    print("Setting up clang library...")
    
    # Common paths for libclang on Linux
    possible_paths = [
        "/usr/lib/llvm-15/lib",  # Ubuntu/Debian with LLVM 15
        "/usr/lib/llvm-14/lib",  # Ubuntu/Debian with LLVM 14
        "/usr/lib/llvm-13/lib",  # Ubuntu/Debian with LLVM 13
        "/usr/lib/llvm/lib",     # Generic path
        "/usr/lib64/llvm",       # Some Linux distributions
        "/usr/lib/x86_64-linux-gnu",  # Another common location
    ]
    
    # Try to find libclang
    for path in possible_paths:
        print(f"Checking {path}...")
        if os.path.exists(os.path.join(path, "libclang.so")):
            print(f"Found libclang.so in {path}")
            try:
                cindex.Config.set_library_path(path)
                cindex.Config.set_compatibility_check(False)
                # Test if it works
                index = cindex.Index.create()
                _CLANG_INITIALIZED = True
                print("Successfully initialized clang")
                return True
            except Exception as e:
                print(f"Error initializing clang with {path}: {str(e)}")
                continue
    
    paths_checked = "\n".join(f"- {p}" for p in possible_paths)
    raise RuntimeError(
        f"Could not find working libclang.so. Checked following paths:\n{paths_checked}\n"
        "Please install clang/llvm development packages.\n"
        "On Ubuntu/Debian: sudo apt-get install libclang-dev"
    )

# Initialize clang when module is imported
setup_clang()

def find_function_in_cursor(cursor, target_name):
    """
    Recursively traverse the AST to find functions or methods whose spelling matches the target.
    Handles both standalone functions and class methods.
    """
    matches = []
    
    # Handle class methods and standalone functions
    if cursor.kind in (cindex.CursorKind.FUNCTION_DECL, cindex.CursorKind.CXX_METHOD):
        # Get fully qualified name for the function
        qualified_name = get_fully_qualified_name(cursor)
        
        # Check for exact matches only
        if (qualified_name == target_name or  # Exact match with qualified name
            ("::" not in target_name and cursor.spelling == target_name)):  # Exact match with just function name
            matches.append(cursor)
    
    # Also check class declarations for methods
    elif cursor.kind == cindex.CursorKind.CLASS_DECL:
        class_name = cursor.spelling
        # If we're looking for a method of this class
        if "::" in target_name and target_name.startswith(f"{class_name}::"):
            method_name = target_name.split("::")[-1]
            # Search through class methods
            for child in cursor.get_children():
                if child.kind == cindex.CursorKind.CXX_METHOD and child.spelling == method_name:
                    matches.append(child)
    
    # Recurse through children
    for child in cursor.get_children():
        matches.extend(find_function_in_cursor(child, target_name))
    
    return matches

def should_replace_type(current_file, new_file, current_type, new_type):
    """
    Determine if we should replace the current type definition with the new one.
    """
    # If current is header and new is implementation, prefer implementation
    if (new_file.endswith(('.cpp', '.cc', '.cxx')) and 
        current_file.endswith(('.h', '.hpp', '.hxx'))):
        return True
    # If both are headers or both are implementations, prefer the one with content
    if current_file.endswith(('.h', '.hpp', '.hxx')) == new_file.endswith(('.h', '.hpp', '.hxx')):
        return len(new_type["body"].strip()) > len(current_type["body"].strip())
    return False

def get_fully_qualified_name(cursor):
    """Get the fully qualified name including namespaces and class names."""
    if cursor is None:
        return ""
    
    # Build namespace/class list from parent cursors
    components = []
    current = cursor
    
    while current and current.kind != cindex.CursorKind.TRANSLATION_UNIT:
        if current.kind in (cindex.CursorKind.NAMESPACE, 
                          cindex.CursorKind.CLASS_DECL,
                          cindex.CursorKind.STRUCT_DECL):
            components.insert(0, current.spelling)
        current = current.semantic_parent
    
    # Add the function/method name
    components.append(cursor.spelling)
    
    # Combine with :: separator
    return "::".join(components)

def collect_dependencies(cursor, dependencies=None):
    """
    Walk the AST subtree of a function to collect its dependencies.
    """
    if dependencies is None:
        dependencies = {"functions": [], "types": {}}

    def is_implementation_file(file_path):
        return file_path.endswith(('.cpp', '.cc', '.cxx'))

    for child in cursor.get_children():
        # If it's a call expression, record detailed info about the function being called.
        if child.kind == cindex.CursorKind.CALL_EXPR:
            referenced = child.referenced
            if referenced is not None:
                qualified_name = get_fully_qualified_name(referenced)
                func_info = {
                    "name": qualified_name,
                    "signature": referenced.type.spelling,
                    "location": (
                        f"{referenced.location.file}:{referenced.location.line}:"
                        f"{referenced.location.column}"
                        if referenced.location.file else "unknown"
                    )
                }
                dependencies["functions"].append(func_info)

        # If it's a type reference, and that type is a struct or class, record details.
        if child.kind == cindex.CursorKind.TYPE_REF:
            referenced = child.referenced
            if referenced is not None and referenced.kind in (
                cindex.CursorKind.STRUCT_DECL,
                cindex.CursorKind.CLASS_DECL
            ):
                qualified_name = get_fully_qualified_name(referenced)
                extent = referenced.extent
                body = ""
                if extent and extent.start.file:
                    try:
                        with open(extent.start.file.name, 'r') as f:
                            source_lines = f.readlines()
                            body = ''.join(source_lines[extent.start.line-1:extent.end.line])
                    except Exception as e:
                        body = f"Error reading source: {str(e)}"
                
                type_info = {
                    "name": qualified_name,
                    "location": (
                        f"{referenced.location.file}:{referenced.location.line}:"
                        f"{referenced.location.column}"
                        if referenced.location.file else "unknown"
                    ),
                    "body": body,
                    "file": str(referenced.location.file) if referenced.location.file else ""
                }
                
                # Use the qualified name as key for deduplication
                current_type = dependencies["types"].get(qualified_name)
                if current_type is None:
                    dependencies["types"][qualified_name] = type_info
                else:
                    current_file = current_type["file"]
                    new_file = type_info["file"]
                    if should_replace_type(current_file, new_file, current_type, type_info):
                        dependencies["types"][qualified_name] = type_info

        collect_dependencies(child, dependencies)
    return dependencies


def parse_file(index, file_path, target_name):
    """
    Parse a single file and return any matching function declarations along with the translation unit.
    """
    try:
        # Add compiler arguments for modern C++
        compiler_args = [
            '-x', 'c++',              # Force C++ mode
            '-std=c++17',             # Use C++17
            '-I/usr/include',         # System includes
            '-I/usr/local/include',
            '-I.',                    # Current directory
            '-Isrc',                  # Common source directory
            '-I..',                   # Parent directory
            '-fparse-all-comments',   # Parse all comments
            '-Wno-unknown-warning-option',  # Ignore unknown warnings
            '-ferror-limit=0',        # Don't stop on errors
        ]
        
        # Add the source directory and its parent to include paths
        src_dir = os.path.dirname(file_path)
        if src_dir:
            compiler_args.extend([f'-I{src_dir}'])
            parent_dir = os.path.dirname(src_dir)
            if parent_dir:
                compiler_args.extend([f'-I{parent_dir}'])
        
        translation_unit = index.parse(
            file_path,
            args=compiler_args,
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None

    functions = find_function_in_cursor(translation_unit.cursor, target_name)
    return functions, translation_unit

def scan_codebase(codebase_dir, target_name):
    index = cindex.Index.create()
    found_functions = []
    seen_locations = set()  # To track seen (file, line) pairs
    translation_units = {}

    for root, dirs, files in os.walk(codebase_dir):
        for file in files:
            if file.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
                file_path = os.path.join(root, file)
                functions, tu = parse_file(index, file_path, target_name)
                if functions:
                    for func in functions:
                        loc = func.location
                        key = (loc.file.name if loc.file else None, loc.line)
                        if key not in seen_locations:
                            seen_locations.add(key)
                            found_functions.append(func)
                    translation_units[file_path] = tu

    return found_functions, translation_units

def calculate_tokens_and_price(text):
    """
    Calculate the number of tokens and estimated price for the given text.
    Uses cl100k_base encoding (GPT-4) and current pricing of $2.50 per 1M tokens.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = len(enc.encode(text))
    price = (tokens / 1_000_000) * 2.50  # $2.50 per 1M tokens
    return tokens, price

class DependencyExtractorConfig:
    def __init__(self, 
                 compiler_args=None,
                 file_extensions=None,
                 include_function_locations=False,
                 include_type_locations=False,
                 clang_lib_path="/usr/lib/llvm-15/lib"):
        """
        Configuration for the dependency extractor.
        
        Args:
            compiler_args: List of compiler arguments (e.g. ['-std=c++11'])
            file_extensions: List of file extensions to scan (e.g. ['.cpp', '.h'])
            include_function_locations: Whether to include source locations for function calls
            include_type_locations: Whether to include source locations for type definitions
            clang_lib_path: Path to the clang library
        """
        self.compiler_args = compiler_args or ['-std=c++11']
        self.file_extensions = file_extensions or ['.cpp', '.cc', '.cxx', '.h', '.hpp']
        self.include_function_locations = include_function_locations
        self.include_type_locations = include_type_locations
        self.clang_lib_path = clang_lib_path

class DependencyAnalysis:
    def __init__(self, types, functions, token_count=None, token_price=None, locations=None):
        """
        Results of the dependency analysis.
        
        Args:
            types: Dictionary of type definitions
            functions: List of analyzed functions with their dependencies
            token_count: Number of tokens in the output (if calculated)
            token_price: Estimated price for the tokens (if calculated)
        """
        self.types = types
        self.functions = functions
        self.token_count = token_count
        self.token_price = token_price
        self.locations = locations or []

    def __str__(self):
        return f"DependencyAnalysis(types={len(self.types)}, functions={len(self.functions)}, token_count={self.token_count}, token_price={self.token_price})"

def extract_dependencies(codebase_dir: str, 
                        function_name: str, 
                        config=None) -> 'DependencyAnalysis':
    """
    Analyze dependencies of a function in a codebase.
    
    Args:
        codebase_dir: Directory containing the codebase to analyze
        function_name: Name of the function to analyze
        config: DependencyExtractorConfig object (optional)
        
    Returns:
        DependencyAnalysis object containing the results
        
    Raises:
        FileNotFoundError: If codebase_dir doesn't exist
        RuntimeError: If function is not found or other analysis errors
    """
    if not os.path.isdir(codebase_dir):
        raise FileNotFoundError(f"Directory not found: {codebase_dir}")

    # Ensure clang is initialized
    if not _CLANG_INITIALIZED and not setup_clang():
        raise RuntimeError("Failed to initialize clang library")

    config = config or DependencyExtractorConfig()
    
    # Find and analyze functions
    found_functions, _ = scan_codebase(codebase_dir, function_name)
    if not found_functions:
        raise RuntimeError(f"Function '{function_name}' not found in codebase")
    
    locations = []
    for func in found_functions:
        if func.location.file:
            locations.append(f"{func.location.file}:{func.location.line}")
    if locations:
        print(f"Found function '{function_name}' at: {', '.join(locations)}")

    # Collect all unique type definitions
    all_types = {}
    function_analyses = []
    
    # Analyze each function instance
    for func in found_functions:
        deps = collect_dependencies(func)
        
        # Collect types
        for type_info in deps["types"].values():
            current = all_types.get(type_info["name"])
            if current is None:
                all_types[type_info["name"]] = type_info
            else:
                current_file = current["file"]
                new_file = type_info["file"]
                if should_replace_type(current_file, new_file, current, type_info):
                    all_types[type_info["name"]] = type_info
        
        # Create function analysis
        func_analysis = {
            "name": get_fully_qualified_name(func),
            "location": (f"{func.location.file}:{func.location.line}:{func.location.column}" 
                        if func.location.file else "unknown"),
            "body": "",
            "dependencies": {
                "functions": [],
                "types": sorted(deps["types"].keys())
            }
        }
        
        # Get function body
        extent = func.extent
        if extent and extent.start.file:
            with open(extent.start.file.name, 'r') as f:
                source_lines = f.readlines()
                func_analysis["body"] = ''.join(source_lines[extent.start.line-1:extent.end.line])
        
        # Add function calls
        for f in deps["functions"]:
            call_info = {
                "name": f["name"],
                "signature": f["signature"]
            }
            if config.include_function_locations:
                call_info["location"] = f["location"]
            func_analysis["dependencies"]["functions"].append(call_info)
        
        function_analyses.append(func_analysis)

    # Calculate tokens if requested
    token_count = None
    token_price = None
    if config.include_function_locations or config.include_type_locations:
        output = format_analysis_output(function_analyses, all_types, config)
        token_count, token_price = calculate_tokens_and_price(output)

    return DependencyAnalysis(
        types=all_types,
        functions=function_analyses,
        token_count=token_count,
        token_price=token_price,
        locations=locations
    )

def format_analysis_output(functions, types, config):
    """Format the analysis results as a string."""
    output = []
    
    # Format type definitions
    if types:
        output.append("Type Definitions:")
        output.append("-" * 60)
        for t in sorted(types.values(), key=lambda x: x["name"]):
            output.append(f"Type: {t['name']}")
            if config.include_type_locations:
                output.append(f"Declared at: {t['location']}")
            output.append("Definition:")
            for line in t["body"].splitlines():
                output.append(f"    {line}")
            output.append("-" * 60)
        output.append("")

    # Format function analyses
    for func in functions:
        output.append("-" * 60)
        output.append(f"Function: {func['name']}")
        output.append(f"Defined at: {func['location']}")
        
        output.append("\nFunction body:")
        for line in func["body"].splitlines():
            output.append(f"    {line}")
        
        output.append("\nDependencies:")
        
        if func["dependencies"]["functions"]:
            output.append("  Function calls:")
            for f in func["dependencies"]["functions"]:
                output.append(f"    - Name: {f['name']}")
                output.append(f"      Signature: {f['signature']}")
                if config.include_function_locations and "location" in f:
                    output.append(f"      Defined at: {f['location']}")
        else:
            output.append("  No function calls found.")

        if func["dependencies"]["types"]:
            output.append("  Types referenced:")
            for type_name in func["dependencies"]["types"]:
                output.append(f"    - {type_name}")
        else:
            output.append("  No type references found.")
        output.append("-" * 60 + "\n")
    
    return "\n".join(output)

def main():
    """Command-line interface for the dependency extractor."""
    if len(sys.argv) < 3:
        print("Usage: python extract_deps.py <codebase_directory> <function_name>")
        sys.exit(1)

    try:
        analysis = extract_dependencies(sys.argv[1], sys.argv[2])
        print(format_analysis_output(
            analysis.functions, 
            analysis.types, 
            DependencyExtractorConfig()
        ))
        if analysis.token_count is not None:
            print(f"\nToken count: {analysis.token_count:,}")
            print(f"Estimated price: ${analysis.token_price:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
