import os
import sys
import subprocess
import re
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
            "/usr/lib/llvm-18/lib",
            "/usr/lib/llvm-17/lib",
            "/usr/lib/llvm-16/lib",
            "/usr/lib/llvm-15/lib",
            "/usr/lib64/llvm",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/llvm/lib",
        ]
    
    # Try to find libclang
    for path in possible_paths:
        print(f"Checking {path}...")
        joined = os.path.join(path, "libclang.so")
        if os.path.exists(joined):
            print(f"Found libclang.so in {path}")
            try:
                cindex.Config.set_library_file(joined)
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

def is_mangled_name(name):
    """
    Check if a function name appears to be in C++ mangled format.
    C++ mangled names typically start with _Z, followed by a number and name components.
    """
    return name.startswith('_Z')

def extract_components_from_mangled(mangled_name):
    """
    Extract components from a mangled name when c++filt fails.
    This is a fallback mechanism to handle cases when c++filt cannot demangle a name.
    
    Examples:
    _ZN7genetic6detail13evaluate_nodeERKNS_4nodeEPKfmmS5_ -> genetic::detail::evaluate_node
    _ZN7genetic5stackIPfi20E4pushEf -> genetic::stack::push
    
    Note: This is a simplified approach and won't handle all mangling cases perfectly,
    but it's good enough for basic function name matching.
    
    Returns the best effort demangled name.
    """
    if not mangled_name.startswith('_Z'):
        return mangled_name
    
    # Special case handling for template classes with nested push method
    # _ZN7genetic5stackIPfi20E4pushEf -> genetic::stack::push
    # This regular expression tries to identify the pattern of mangled template classes
    template_class_method = re.match(r'_ZN(\d+)([a-zA-Z0-9_]+)(\d+)([a-zA-Z0-9_]+)(?:I.*?E)(\d+)([a-zA-Z0-9_]+)', mangled_name)
    if template_class_method:
        namespace_len = int(template_class_method.group(1))
        namespace = template_class_method.group(2)[:namespace_len]
        class_len = int(template_class_method.group(3))
        class_name = template_class_method.group(4)[:class_len]
        method_len = int(template_class_method.group(5))
        method_name = template_class_method.group(6)[:method_len]
        return f"{namespace}::{class_name}::{method_name}"
    
    # For non-template cases or simpler cases, use the standard approach
    # Remove _Z prefix
    name = mangled_name[2:]
    
    parts = []
    i = 0
    while i < len(name):
        if name[i] == 'N':
            # Nested name, skip the N
            i += 1
        elif name[i].isdigit():
            # Length-prefixed identifier
            length = 0
            while i < len(name) and name[i].isdigit():
                length = length * 10 + int(name[i])
                i += 1
            if i + length <= len(name):
                parts.append(name[i:i+length])
                i += length
            else:
                # Invalid format, just return original
                return mangled_name
        elif name[i] == 'E':
            # End of nested name
            break
        else:
            # Skip over template parameter section (everything between I and E)
            if name[i] == 'I':
                template_depth = 1
                i += 1
                while i < len(name) and template_depth > 0:
                    if name[i] == 'I':
                        template_depth += 1
                    elif name[i] == 'E':
                        template_depth -= 1
                    i += 1
            else:
                # Unknown component, just continue
                i += 1
            
    if parts:
        return '::'.join(parts)
    return mangled_name


def demangle_name(mangled_name, include_parameters=True):
    """
    Demangle a C++ mangled name using c++filt.
    Returns the demangled name if successful, or the original name if demangling fails.
    Also cleans up the demangled name to make it compatible with Clang's expectations.
    
    Args:
        mangled_name: The mangled C++ name to demangle.
        include_parameters: If True, retains parameter details in the demangled name.
                            If False, strips parameter details.
    """
    try:
        result = subprocess.run(['c++filt', mangled_name], capture_output=True, text=True)
        if result.returncode == 0:
            demangled = result.stdout.strip()

            # If c++filt couldn't demangle it (returns the same string), try our fallback
            if demangled == mangled_name and is_mangled_name(mangled_name):
                print(f"c++filt failed to demangle {mangled_name}, using fallback")
                return extract_components_from_mangled(mangled_name)

            # Optionally remove parameter details if include_parameters is False
            if not include_parameters:
                # First find the template part if it exists
                template_match = re.match(r'(.*?)(<.*?>)(.*?)(\()', demangled)
                if template_match:
                    # For template functions, combine the parts up to the opening parenthesis
                    return (template_match.group(1) + template_match.group(2) + template_match.group(3)).strip()
                else:
                    # For regular functions, just remove everything from the first opening parenthesis
                    match = re.match(r'([^(]+)(\(|$)', demangled)
                    if match:
                        return match.group(1).strip()

            return demangled
    except (subprocess.SubprocessError, FileNotFoundError):
        # If c++filt isn't available, try our fallback for mangled names
        if is_mangled_name(mangled_name):
            return extract_components_from_mangled(mangled_name)

    return mangled_name


def normalize_template_name(name):
    """
    Preserve mangled names as-is, since they're now the default.
    This function is kept for backward compatibility but now acts as a pass-through.
    """
    # Simply return the name as-is, preserving mangling
    return name

def find_function_in_cursor(cursor, target_name):
    """
    Recursively traverse the AST to find functions or methods whose spelling matches the target.
    Handles both standalone functions and class methods, including mangled names.
    """
    # print(f"find_function_in_cursor({cursor}, {target_name})")
    matches = []
    # print(f"Checking cursor: {cursor.spelling} ({cursor.kind}), target: {target_name}")
    
    # With mangled names as default, we keep the target name as-is
    normalized_target = target_name
    
    # Check if target is a mangled name
    is_target_mangled = is_mangled_name(target_name)
    
    # Get just the method name (after the last ::)
    method_name = target_name.split("::")[-1] if "::" in target_name else target_name
    normalized_method = method_name
    
    # For backward compatibility, also check the base name without template params
    base_method_name = method_name.split("<")[0] if "<" in method_name else method_name
    # cursor_mangled_name = cursor.mangled_name if hasattr(cursor, 'mangled_name') else None
    # print(f"cursor_mangled_name: {cursor_mangled_name}")
    # Handle class methods and standalone functions (including function templates)
    if cursor.kind in (cindex.CursorKind.FUNCTION_DECL, cindex.CursorKind.CXX_METHOD, cindex.CursorKind.FUNCTION_TEMPLATE):
        # Get fully qualified name for the function
        qualified_name = get_fully_qualified_name(cursor)
        
        # Get cursor's mangled name if available
        cursor_mangled_name = cursor.mangled_name if hasattr(cursor, 'mangled_name') else None
        # print(f"cursor_mangled_name: {cursor.__dict__}, qualified_name: {qualified_name}")
        
        # Direct mangled name check if target is mangled
        if is_target_mangled and cursor_mangled_name and cursor_mangled_name == target_name:
            matches.append(cursor)
            return matches
            
        # Standard checks if not found by direct mangled name comparison
        if (qualified_name == target_name or  # Exact match with qualified name
            ("::" not in target_name and cursor.spelling == method_name) or  # Match with just function name
            ("::" not in target_name and cursor.spelling == base_method_name)):  # Match with base method name (for compatibility)
            matches.append(cursor)
    
    # Check class templates and regular class declarations for methods
    elif cursor.kind in (cindex.CursorKind.CLASS_DECL, cindex.CursorKind.CLASS_TEMPLATE):
        class_name = cursor.spelling
        class_qualified_name = get_fully_qualified_name(cursor)
        
        # If target_name starts with this class's name (preserve mangling)
        if ("::" in target_name and 
            (target_name.startswith(f"{class_qualified_name}::") or
             # Handle case where the class is part of a template instance
             (("<" in target_name) and 
              (target_name.split("<")[0].endswith(class_name) or 
               target_name.startswith(f"{class_name}<"))))):
            
            # Search through class methods
            for child in cursor.get_children():
                # Get child's mangled name if available
                child_mangled_name = child.mangled_name if hasattr(child, 'mangled_name') else None
                
                # Direct mangled name check if target is mangled
                if is_target_mangled and child_mangled_name and child_mangled_name == target_name:
                    matches.append(child)
                    continue
                    
                # Compare the method name (preserving mangling)
                if child.kind == cindex.CursorKind.CXX_METHOD and (
                   child.spelling == method_name or
                   child.spelling == base_method_name):
                    matches.append(child)
    
    # Check namespace declarations to help narrow down the search
    elif cursor.kind == cindex.CursorKind.NAMESPACE:
        # print(f"Checking namespace: {cursor.spelling} (children names: {[c.spelling for c in cursor.get_children()]})")
        namespace_name = cursor.spelling
        if not is_target_mangled and target_name.startswith(f"{namespace_name}::"):
            # This namespace might contain our target, prioritize searching its children
            for child in cursor.get_children():
                matches.extend(find_function_in_cursor(child, target_name))
    
    # Recurse through children if we haven't found a match yet
    if not matches:
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
    """
    Get the fully qualified name including namespaces and class names.
    Handles templates appropriately.
    """
    if cursor is None:
        return ""
    
    # Build namespace/class list from parent cursors
    components = []
    current = cursor
    
    while current and current.kind != cindex.CursorKind.TRANSLATION_UNIT:
        if current.kind in (cindex.CursorKind.NAMESPACE, 
                            cindex.CursorKind.CLASS_DECL,
                            cindex.CursorKind.STRUCT_DECL,
                            cindex.CursorKind.CLASS_TEMPLATE):
            # For template classes, include the template parameters if available
            if current.kind == cindex.CursorKind.CLASS_TEMPLATE:
                # Try to get template parameters, but use just the name if we can't
                template_name = current.spelling
                components.insert(0, template_name)
            else:
                components.insert(0, current.spelling)
        current = current.semantic_parent
    
    # Add the function/method name
    if cursor.kind in (cindex.CursorKind.FUNCTION_TEMPLATE, 
                      cindex.CursorKind.FUNCTION_DECL, 
                      cindex.CursorKind.CXX_METHOD):
        components.append(cursor.spelling)
    else:
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
                
                # Always use demangled names for output to the LLM
                if is_mangled_name(qualified_name):
                    demangled_name = demangle_name(qualified_name)
                    final_name = demangled_name
                else:
                    final_name = qualified_name
                
                func_info = {
                    "name": final_name,
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
                
                # Always use demangled names for output to the LLM
                if is_mangled_name(qualified_name):
                    demangled_name = demangle_name(qualified_name)
                    final_name = demangled_name
                else:
                    final_name = qualified_name
                    
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
                    "name": final_name,
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
    # print(f"Searching for '{target_name}' in {file_path}")
    try:
        # Add compiler arguments for modern C++
        compiler_args = [
            '-x', 'c++',              # Force C++ mode
            '-std=c++17',             # Use C++17
            '-I/usr/include',         # System includes
            '-I/usr/local/include',
            '-I.',                    # Current directory
            '-Isrc',                  # Common source directory
            '-Iinclude',              # Another common source directory
            '-I..',                   # Parent directory
            '-fparse-all-comments',   # Parse all comments
            '-Wno-unknown-warning-option',  # Ignore unknown warnings
            '-ferror-limit=0',        # Don't stop on errors
            '-D__clang_analyzer__',   # Enable clang analyzer
            '-ftemplate-depth=1024',  # Increase template recursion depth
        ]
        
        # Add the source directory and its parent to include paths
        src_dir = os.path.dirname(file_path)
        if src_dir:
            compiler_args.extend([f'-I{src_dir}'])
            parent_dir = os.path.dirname(src_dir)
            if parent_dir:
                compiler_args.extend([f'-I{parent_dir}'])
                
                # Add potential include directories that might exist in the project structure
                potential_dirs = ['include', 'src', 'lib']
                for p_dir in potential_dirs:
                    full_path = os.path.join(parent_dir, p_dir)
                    if os.path.isdir(full_path):
                        compiler_args.extend([f'-I{full_path}'])
        
        # Process any template arguments in the target name for special handling
        if '<' in target_name and '>' in target_name:
            # Add defines to help with template parsing if needed
            compiler_args.extend(['-fdelayed-template-parsing'])
        
        translation_unit = index.parse(
            file_path,
            args=compiler_args,
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD | 
                   cindex.TranslationUnit.PARSE_INCOMPLETE
        )
        
        # Check for parse errors that might affect template instantiation
        # if translation_unit.diagnostics:
        #     for diag in translation_unit.diagnostics:
        #         if diag.severity >= cindex.Diagnostic.Error:
        #             print(f"Parse error in {file_path}: {diag.spelling}")
                    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None

    functions = find_function_in_cursor(translation_unit.cursor, target_name)
    return functions, translation_unit

def scan_codebase(codebase_dir, target_name):
    # print(f"scanning codebase for {target_name}")
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
    # Remove potential return type in the function name
    # E.g., "void genetic::meanSquareError<float>" -> "genetic::meanSquareError<float>"
    if ' ' in function_name and '::' in function_name:
        # Split by space and check if the second part has "::" indicating it's a namespaced function
        parts = function_name.split(' ', 1)
        if '::' in parts[1]:
            function_name = parts[1]
            print(f"Removed return type from function name: {function_name}")
    
    function_name = function_name.replace('\\<', '<').replace('\\>', '>').replace('\\,', ',')
    if not os.path.isdir(codebase_dir):
        raise FileNotFoundError(f"Directory not found: {codebase_dir}")

    # Ensure clang is initialized
    if not _CLANG_INITIALIZED and not setup_clang():
        raise RuntimeError("Failed to initialize clang library")

    config = config or DependencyExtractorConfig()
    
    # Handle different types of function names
    original_function_name = function_name
    
    # Check if the function name is mangled
    if is_mangled_name(function_name):
        print(f"Detected mangled name: '{function_name}'")
        # Try to demangle it
        demangled_name = demangle_name(function_name, include_parameters=False)
        if demangled_name != function_name:
            print(f"Demangled to: '{demangled_name}'")
            # Keep the original mangled name for reference
            mangled_name = function_name
            # Use the demangled name for the search since Clang uses demangled names internally
            function_name = function_name
            original_function_name = demangled_name
    
    normalized_function_name = normalize_template_name(original_function_name)
    
    # Try with original name first, then with normalized name if needed
    found_functions = []
    translation_units = None
    
    # First try with the original name (with template parameters)
    found_functions, translation_units = scan_codebase(codebase_dir, original_function_name)
    
    # If no results, try with normalized name (templates removed)
    if not found_functions and normalized_function_name != original_function_name:
        print(f"No matches found using '{original_function_name}'. Trying with normalized name: '{normalized_function_name}'")
        found_functions, translation_units = scan_codebase(codebase_dir, normalized_function_name)
    
    # Last resort: try with just the function name (no namespaces/classes)
    if not found_functions and "::" in original_function_name:
        simple_name = original_function_name.split("::")[-1]
        normalized_simple_name = normalize_template_name(simple_name)
        
        if simple_name != original_function_name:
            print(f"No matches found. Trying with simple name: '{simple_name}'")
            found_functions, translation_units = scan_codebase(codebase_dir, simple_name)
        
        if not found_functions and normalized_simple_name != simple_name:
            print(f"No matches found. Trying with normalized simple name: '{normalized_simple_name}'")
            found_functions, translation_units = scan_codebase(codebase_dir, normalized_simple_name)
    
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
        
        # Get fully qualified name including template context if possible
        qualified_name = get_fully_qualified_name(func)
        
        # Always use the demangled name for the LLM
        # The LLM can't work with mangled names effectively
        final_name = qualified_name
        
        # Create function analysis
        func_analysis = {
            "name": final_name,
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
            try:
                with open(extent.start.file.name, 'r') as f:
                    source_lines = f.readlines()
                    # For templated functions, we may need to search for the complete implementation
                    # Start with the basic extraction
                    body_text = ''.join(source_lines[extent.start.line-1:extent.end.line])
                    
                    # If this seems incomplete (missing closing brace), search for the complete implementation
                    if '{' in body_text and '}' not in body_text:
                        # Find the matching closing brace
                        open_braces = 0
                        end_line = extent.end.line
                        for i, line in enumerate(source_lines[extent.start.line-1:], extent.start.line-1):
                            for char in line:
                                if char == '{':
                                    open_braces += 1
                                elif char == '}':
                                    open_braces -= 1
                                    if open_braces == 0:
                                        end_line = i + 1
                                        break
                            if open_braces == 0 and end_line > extent.end.line:
                                break
                        
                        # Extract the complete body using the corrected end line
                        body_text = ''.join(source_lines[extent.start.line-1:end_line])
                    
                    func_analysis["body"] = body_text
            except Exception as e:
                print(f"Error reading function body: {str(e)}")
                func_analysis["body"] = f"Error: {str(e)}"
        
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
        print("\nExample: python extract_deps.py ./codebase genetic::stack<float, 20>::push")
        print("Note: For templated functions, you can now use the full template syntax")
        print("      Return types in function names (e.g., 'void genetic::foo') will be automatically removed")
        sys.exit(1)

    try:
        # Combine all arguments after the codebase directory as the function name
        # This allows for names with spaces and template parameters like "stack<int, 20>::push"
        codebase_dir = sys.argv[1]
        function_name = " ".join(sys.argv[2:])
        
        # Replace escaped characters that might come from shell
        # function_name = function_name.replace('\\<', '<').replace('\\>', '>').replace('\\,', ',')
        
        print(f"Searching for function: {function_name}")
        analysis = extract_dependencies(codebase_dir, function_name)
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
