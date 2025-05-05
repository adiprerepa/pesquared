from typing import List
import clang.cindex
import networkx as nx
from collections import defaultdict
from utils.graph_utils import DAGish

class DataFlowGraph(DAGish):
    def __init__(self, tu_cursor, file: List[str]):
        """
        tu_cursor: a clang.cindex.Cursor pointing at the translation unit
        """
        self.tu_cursor = tu_cursor
        self.file = file
        # Map variable USR -> list of definition nodes (node_ids)
        self.defs = defaultdict(list)
        self._build()

    def _get_node_id(self, cursor):
        """Return a unique node identifier, here by file:line."""
        loc = cursor.location
        fname = loc.file.name if loc.file else "<unknown>"
        return f"{fname}:{loc.line}"

    def _get_code_snippet(self, cursor):
        """Extract the source text covered by this cursor."""
        start = cursor.extent.start
        end   = cursor.extent.end
        lines = self.file  # use the already loaded file!
        if start.line == end.line:
            return lines[start.line-1][start.column-1:end.column-1].strip()
        else:
            snippet = [lines[start.line-1][start.column-1:].rstrip()]
            for l in range(start.line, end.line-1):
                snippet.append(lines[l].rstrip())
            snippet.append(lines[end.line-1][:end.column-1].rstrip())
            return ' '.join(snippet)


    def _add_node(self, cursor):
        nid = self._get_node_id(cursor)
        if not self.has_node(nid):
            self.add_node(
                nid,
                code=self._get_code_snippet(cursor),
                line=cursor.location.line,
                label=self.file[cursor.location.line-1].strip(),
                file=cursor.location.file.name if cursor.location.file else None
            )
        return nid

    def _handle_assignment(self, cursor):
        """
        For assignments (BinaryOperator with '='),
        register the left-hand side as a new definition.
        """
        # find the left side (assumes first child)
        lhs = next(cursor.get_children(), None)
        if lhs and lhs.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            # this is a write to a variable
            var_usr = lhs.get_definition().get_usr()
            def_nid = self._add_node(cursor)
            self.defs[var_usr].append(def_nid)

    def _handle_decl(self, cursor):
        """
        For variable declarations with init, treat as a definition.
        """
        # VarDecl with an initializer
        if cursor.kind == clang.cindex.CursorKind.VAR_DECL:
            if any(c.kind == clang.cindex.CursorKind.INTEGER_LITERAL or
                   c.kind == clang.cindex.CursorKind.CALL_EXPR
                   for c in cursor.get_children()):
                var_usr = cursor.get_usr()
                def_nid = self._add_node(cursor)
                self.defs[var_usr].append(def_nid)

    def _handle_use(self, cursor):
        """
        For DeclRefExpr (reads), link back to all current definitions.
        """
        if cursor.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            # skip if it's part of the definition itself
            def_cursor = cursor.get_definition()
            if def_cursor is None:
                return
            var_usr = def_cursor.get_usr()
            use_nid = self._add_node(cursor)
            for def_nid in self.defs.get(var_usr, []):
                self.add_edge(def_nid, use_nid)

    def _visit(self, cursor):
        """
        Recursively visit AST, handling defs, assigns, and uses.
        """
        # assignments
        if cursor.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            # see if this is '=' operator
            tok = list(cursor.get_tokens())
            if any(t.spelling == '=' for t in tok):
                self._handle_assignment(cursor)

        # declarations
        if cursor.kind == clang.cindex.CursorKind.VAR_DECL:
            self._handle_decl(cursor)

        # uses
        if cursor.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            self._handle_use(cursor)

        # recurse
        for child in cursor.get_children():
            self._visit(child)
    
    def _build(self):
        """
        Build the data flow graph from the AST.
        """
        self._visit(self.tu_cursor)
