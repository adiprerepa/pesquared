#!/usr/bin/env python3

import re
import os
from typing import Dict, List, Optional, Tuple

def extract_function_body(content: str, function_name: str) -> Optional[str]:
    """
    Extract a function body from a source code file.
    
    Args:
        content: Source code content
        function_name: Name of the function to extract
        
    Returns:
        Function body as a string or None if not found
    """
    # Try to find the function definition
    pattern = r'(\w+(?:\s+\w+)*\s+' + re.escape(function_name) + r'\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:final\s*)?(?:=\s*default\s*)?(?:=\s*delete\s*)?(?:{\s*(?:[^{}]*(?:{[^{}]*})*[^{}]*)*\s*}|;))'
    match = re.search(pattern, content)
    
    if match:
        return match.group(1)
    return None

def replace_function_body(content: str, original_function: str, new_function: str) -> str:
    """
    Replace a function body in source code with a new implementation.
    
    Args:
        content: Source code content
        original_function: Original function implementation
        new_function: New function implementation
        
    Returns:
        Modified source code with the new function implementation
    """
    # Escape special regex characters in the original function
    escaped_original = re.escape(original_function)
    return re.sub(escaped_original, new_function, content)

def parse_function_signature(function_body: str) -> Dict[str, str]:
    """
    Parse a function's signature to extract return type, name, and parameters.
    
    Args:
        function_body: Function implementation including signature
        
    Returns:
        Dictionary with 'return_type', 'name', and 'parameters' keys
    """
    # Extract signature (everything before the first '{' or ';')
    signature_match = re.match(r'(.*?)(?:{|;)', function_body, re.DOTALL)
    if not signature_match:
        return {'return_type': '', 'name': '', 'parameters': ''}
    
    signature = signature_match.group(1).strip()
    
    # Extract function name and parameters
    func_match = re.search(r'(\w+)\s*\((.*?)\)', signature, re.DOTALL)
    if not func_match:
        return {'return_type': '', 'name': '', 'parameters': ''}
    
    name = func_match.group(1)
    parameters = func_match.group(2).strip()
    
    # Extract return type (everything before the function name)
    return_type = signature[:signature.rfind(name)].strip()
    
    return {
        'return_type': return_type,
        'name': name,
        'parameters': parameters
    }

def get_function_location(codebase_dir: str, function_name: str) -> List[Tuple[str, int]]:
    """
    Find the location of a function in a codebase.
    
    Args:
        codebase_dir: Directory containing the codebase
        function_name: Name of the function to find
        
    Returns:
        List of (file_path, line_number) tuples where the function is defined
    """
    locations = []
    
    # Use grep to find the function definition
    try:
        cmd = f'grep -rn "\\b{function_name}\\s*(" --include="*.cpp" --include="*.h" {codebase_dir}'
        result = os.popen(cmd).read().strip()
        
        if result:
            for line in result.split('\n'):
                parts = line.split(':', 2)
                if len(parts) >= 2:
                    file_path = parts[0]
                    line_number = int(parts[1])
                    locations.append((file_path, line_number))
    except Exception as e:
        print(f"Error finding function: {e}")
    
    return locations