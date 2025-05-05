import clang.cindex as cindex

def loc_info(cur: cindex.Cursor) -> dict:
    loc = cur.location
    return {"file": str(loc.file) if loc.file else "", "line": loc.line, "col": loc.column}

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