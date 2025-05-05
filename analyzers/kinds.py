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

def in_repo_and_not_variable(node, attr):
    return is_kind(attr['kind'], NodeKind.IN_CODEBASE) and not is_kind(attr['kind'], NodeKind.VARIABLE)

def in_repo(node, attr):
    return is_kind(attr['kind'], NodeKind.IN_CODEBASE)

def is_function(node, attr):
    return is_kind(attr['kind'], NodeKind.FUNCTION)

def is_variable(node, attr):
    return is_kind(attr['kind'], NodeKind.VARIABLE)

def is_type(node, attr):
    return is_kind(attr['kind'], NodeKind.TYPE)
