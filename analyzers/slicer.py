from clang import cindex
from collections import defaultdict, deque

class CppSlicer:
    def __init__(self, cursor: cindex.Cursor, source_lines: list[str]):
        self.cursor = cursor
        self.source_lines = source_lines  # list of lines of the source file
        self.graph = defaultdict(set)     # line_number -> set of dependent line_numbers
        self.scope_stack = [{}]            # stack of scopes, each is var_name -> defining line_number
        self.blocks = []                  # list of (start_line, end_line, parent_line) for {} blocks
        self.build_graph()

    def build_graph(self):
        self._visit(self.cursor, None)

    def _visit(self, node, parent_start_line):
        if node.location.file and node.location.file.name != self.cursor.location.file.name:
            return

        start_line = node.extent.start.line
        end_line = node.extent.end.line

        if node.kind.is_declaration():
            if node.kind.name == "VAR_DECL":
                var_name = node.spelling
                self.scope_stack[-1][var_name] = start_line

        if node.kind.is_expression() or node.kind.is_statement():
            if node.kind.name == "COMPOUND_STMT":
                self.blocks.append((start_line, end_line, parent_start_line))
                self.scope_stack.append({})  # Enter new scope

            if node.kind.name in ("IF_STMT", "FOR_STMT", "WHILE_STMT", "DO_STMT", "SWITCH_STMT"):
                for child in node.get_children():
                    child_start = child.extent.start.line
                    self.graph[start_line].add(child_start)

            if node.kind.name == "DECL_REF_EXPR":
                var_name = node.spelling
                # Find the nearest definition in scope stack
                for scope in reversed(self.scope_stack):
                    if var_name in scope:
                        def_line = scope[var_name]
                        self.graph[def_line].add(start_line)
                        break

        for child in node.get_children():
            self._visit(child, parent_start_line if node.kind.name != "COMPOUND_STMT" else start_line)

        if node.kind.is_statement() and node.kind.name == "COMPOUND_STMT":
            self.scope_stack.pop()  # Exit scope

    def get_slice(self, line_number: int) -> str:
        visited = set()
        queue = deque([line_number])

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                for parent in self._get_parents(current):
                    queue.append(parent)

        for start, end, parent in self.blocks:
            if any(start <= ln <= end for ln in visited):
                if parent is not None:
                    visited.add(parent)
                visited.add(start)
                visited.add(end)

        collected_lines = [self.source_lines[i - 1] for i in sorted(visited) if 1 <= i <= len(self.source_lines)]
        return "\n".join(collected_lines)

    def _get_parents(self, line_number: int) -> list[int]:
        parents = []
        for src, targets in self.graph.items():
            if line_number in targets:
                parents.append(src)
        return parents