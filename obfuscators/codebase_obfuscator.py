import os
import sys
import random
import string
from clang import cindex

###############################################################################
# 1. Setup & Configuration
###############################################################################

LIBCLANG_PATHS = [
    "/usr/lib/llvm-18/lib",
    "/usr/lib/llvm-17/lib",
    "/usr/lib/llvm-16/lib",
    "/usr/lib/llvm-15/lib",
    "/usr/lib/llvm/lib",
    "/usr/lib64/llvm",
    "/usr/lib/x86_64-linux-gnu",
]

FILE_EXTENSIONS = ('.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx')

OBFUSCATE_DECL_KINDS = {
    cindex.CursorKind.CLASS_DECL,
    cindex.CursorKind.CLASS_TEMPLATE,
    cindex.CursorKind.STRUCT_DECL,
    cindex.CursorKind.ENUM_DECL,
    cindex.CursorKind.TYPEDEF_DECL,
    cindex.CursorKind.FIELD_DECL,
    cindex.CursorKind.VAR_DECL,
    cindex.CursorKind.FUNCTION_DECL,
    cindex.CursorKind.CXX_METHOD,
    cindex.CursorKind.FUNCTION_TEMPLATE
}

def setup_clang():
    if cindex.Config.loaded:
        return
    for path in LIBCLANG_PATHS:
        candidate = os.path.join(path, "libclang.so")
        if os.path.exists(candidate):
            try:
                cindex.Config.set_library_file(candidate)
                cindex.Config.set_compatibility_check(False)
                _ = cindex.Index.create()
                print(f"[INFO] Using libclang from: {candidate}")
                return
            except Exception:
                continue
    raise RuntimeError("Could not find or load libclang.so. Install LLVM and libclang.")

###############################################################################
# 2. Helpers
###############################################################################

import hashlib

def generate_deterministic_identifier(original_name):
    hash_object = hashlib.md5(original_name.encode())
    short_hash = hash_object.hexdigest()[:8]
    return f"o__{short_hash}"

def should_obfuscate(cursor):
    if cursor.kind not in OBFUSCATE_DECL_KINDS:
        return False
    location = cursor.location
    if not location.file or not location.file.name:
        return False
    if ("/usr/include" in location.file.name or 
        "/usr/local/include" in location.file.name):
        return False
    if not os.access(location.file.name, os.W_OK):
        return False
    if not cursor.spelling:
        return False
    if cursor.spelling.startswith("__"):
        return False
    return True

###############################################################################
# 3. Obfuscation - Pass 1 (build mapping)
###############################################################################

def collect_global_renames(codebase_dir):
    global_rename_map = {}    # USR -> new name
    global_name_map = {}      # original spelling -> new name
    index = cindex.Index.create()

    def visit(cursor):
        if should_obfuscate(cursor):
            usr = cursor.get_usr()
            spelling = cursor.spelling
            if usr not in global_rename_map and spelling:
                new_name = generate_deterministic_identifier(spelling)
                global_rename_map[usr] = new_name
                global_name_map[spelling] = new_name
        for child in cursor.get_children():
            visit(child)

    for root, _, files in os.walk(codebase_dir):
        for fname in files:
            if fname.endswith(FILE_EXTENSIONS):
                filepath = os.path.join(root, fname)
                try:
                    tu = index.parse(
                        filepath,
                        args=[
                            '-x', 'c++',
                            '-std=c++17',
                            '-I/usr/include',
                            '-I/usr/local/include',
                            '-I.', '-I' + root
                        ],
                        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                                cindex.TranslationUnit.PARSE_INCOMPLETE
                    )
                    visit(tu.cursor)
                except Exception as e:
                    print(f"[WARN] Failed to parse {filepath}: {e}")

    print(f"[INFO] Collected {len(global_rename_map)} identifiers to obfuscate.")
    return global_rename_map, global_name_map

###############################################################################
# 4. Obfuscation - Pass 2 (apply mapping)
###############################################################################

def collect_file_renames(filepath, global_rename_map, global_name_map):
    index = cindex.Index.create()
    file_renames = {}

    try:
        tu = index.parse(
            filepath,
            args=[
                '-x', 'c++',
                '-std=c++17',
                '-I/usr/include',
                '-I/usr/local/include',
                '-I.', '-I' + os.path.dirname(filepath)
            ],
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                    cindex.TranslationUnit.PARSE_INCOMPLETE
        )
    except Exception as e:
        print(f"[WARN] Parsing failed for {filepath}: {e}")
        return file_renames

    # First pass: rename based on references
    def visit(cursor):
        if should_obfuscate(cursor):
            usr = cursor.get_usr()
            if usr in global_rename_map:
                loc = cursor.location
                if loc.file and loc.file.name == filepath:
                    start = (loc.line, loc.column)
                    end = (loc.line, loc.column + len(cursor.spelling))
                    file_renames[(start[0], start[1], end[0], end[1])] = global_rename_map[usr]
        if cursor.referenced and should_obfuscate(cursor.referenced):
            usr = cursor.referenced.get_usr()
            if usr in global_rename_map:
                loc = cursor.location
                if loc.file and loc.file.name == filepath:
                    spelling = cursor.referenced.spelling
                    if cursor.spelling == spelling:
                        start = (loc.line, loc.column)
                        end = (loc.line, loc.column + len(spelling))
                        file_renames[(start[0], start[1], end[0], end[1])] = global_rename_map[usr]
        for child in cursor.get_children():
            visit(child)
    visit(tu.cursor)

    # Second pass: rename based on raw tokens
    for token in tu.get_tokens(extent=tu.cursor.extent):
        loc = token.location
        if loc.file and loc.file.name == filepath:
            if token.spelling in global_name_map:
                start = (loc.line, loc.column)
                end = (loc.line, loc.column + len(token.spelling))
                file_renames[(start[0], start[1], end[0], end[1])] = global_name_map[token.spelling]

    return file_renames

def apply_renames_in_file(filepath, rename_dict):
    if not rename_dict:
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    edits = []
    for (l1, c1, l2, c2), newtext in rename_dict.items():
        edits.append((l1, c1, l2, c2, newtext))
    edits.sort(key=lambda x: (x[0], x[1]), reverse=True)

    changed = False
    for (l1, c1, l2, c2, newtext) in edits:
        line_idx = l1 - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            c1_idx = c1 - 1
            c2_idx = c2 - 1
            if 0 <= c1_idx <= c2_idx <= len(line):
                lines[line_idx] = line[:c1_idx] + newtext + line[c2_idx:]
                changed = True

    if changed:
        with open(filepath, 'w') as f:
            f.writelines(lines)
    return changed

###############################################################################
# 5. Main
###############################################################################

def obfuscate_codebase(codebase_dir):
    setup_clang()
    if not os.path.isdir(codebase_dir):
        raise ValueError(f"Directory does not exist: {codebase_dir}")

    # Pass 1: Build global mappings
    global_rename_map, global_name_map = collect_global_renames(codebase_dir)

    # Pass 2: Apply renames
    for root, _, files in os.walk(codebase_dir):
        for fname in files:
            if fname.endswith(FILE_EXTENSIONS):
                filepath = os.path.join(root, fname)
                print(f"[INFO] Processing {filepath}...")
                file_renames = collect_file_renames(filepath, global_rename_map, global_name_map)
                if file_renames:
                    print(f"  [INFO] Applying {len(file_renames)} changes...")
                    apply_renames_in_file(filepath, file_renames)

    print("[INFO] Obfuscation complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_codebase>")
        sys.exit(1)
    codebase_dir = sys.argv[1]
    obfuscate_codebase(codebase_dir)
