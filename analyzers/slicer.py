from clang import cindex
from collections import defaultdict, deque

class CppSlicer:
    def __init__(self, cursor: cindex.Cursor, source_lines: list[str]):
        self.cursor = cursor
        self.source_lines = source_lines
        self.graph = defaultdict(set)
        self.scope_stack = [{}]
        self.blocks = []
        self.var_last_mutated_at = {}
        self.build_graph()

    def build_graph(self):
        self._visit(self.cursor, None)

    def _descend(self, node):
        yield node
        for child in node.get_children():
            yield from self._descend(child)

    def _is_assignment(self, node):
        tokens = list(node.get_tokens())
        for tok in tokens:
            if tok.spelling == '=':
                return True
        return False

    def _get_lhs_variable(self, node):
        for child in node.get_children():
            if child.kind.name == "DECL_REF_EXPR":
                return child.spelling
        return None

    def _visit(self, node, parent_start_line):
        if node.location.file and node.location.file.name != self.cursor.location.file.name:
            return

        start_line = node.extent.start.line
        end_line = node.extent.end.line

        if node.kind.is_declaration() and node.kind.name == "VAR_DECL":
            var_name = node.spelling
            self.scope_stack[-1][var_name] = start_line
            for child in node.get_children():
                for desc in self._descend(child):
                    if desc.kind.name == "DECL_REF_EXPR":
                        used_var = desc.spelling
                        if used_var in self.var_last_mutated_at:
                            self.graph[self.var_last_mutated_at[used_var]].add(start_line)
                        else:
                            for scope in reversed(self.scope_stack):
                                if used_var in scope:
                                    self.graph[scope[used_var]].add(start_line)
                                    break

        if node.kind.is_expression() or node.kind.is_statement():
            if node.kind.name == "COMPOUND_STMT":
                self.blocks.append((start_line, end_line, parent_start_line))
                self.scope_stack.append({})

            if node.kind.name in ("IF_STMT", "FOR_STMT", "WHILE_STMT", "DO_STMT", "SWITCH_STMT"):
                for child in node.get_children():
                    child_start = child.extent.start.line
                    self.graph[start_line].add(child_start)

            if node.kind.name == "DECL_REF_EXPR":
                var_name = node.spelling
                if var_name in self.var_last_mutated_at:
                    self.graph[self.var_last_mutated_at[var_name]].add(start_line)
                else:
                    for scope in reversed(self.scope_stack):
                        if var_name in scope:
                            self.graph[scope[var_name]].add(start_line)
                            break

            if node.kind.name == "CALL_EXPR":
                for arg in node.get_arguments():
                    for desc in self._descend(arg):
                        if desc.kind.name == "DECL_REF_EXPR":
                            var_name = desc.spelling
                            for scope in reversed(self.scope_stack):
                                if var_name in scope:
                                    self.var_last_mutated_at[var_name] = start_line
                                    scope[var_name] = start_line
                                    self.graph[start_line].add(start_line)
                                    break

            if node.kind.name == "BINARY_OPERATOR" and self._is_assignment(node):
                lhs = self._get_lhs_variable(node)
                if lhs:
                    for scope in reversed(self.scope_stack):
                        if lhs in scope:
                            self.var_last_mutated_at[lhs] = start_line
                            scope[lhs] = start_line
                            self.graph[start_line].add(start_line)
                            break

        for child in node.get_children():
            self._visit(child, parent_start_line if node.kind.name != "COMPOUND_STMT" else start_line)

        if node.kind.is_statement() and node.kind.name == "COMPOUND_STMT":
            self.scope_stack.pop()

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

        sorted_lines = sorted(visited)
        output = []
        last_line = None
        for line_no in sorted_lines:
            if last_line is not None and line_no > last_line + 1:
                gap = line_no - last_line - 1
                # output.append(f"// GAP: {gap} lines excluded for brevity\n")
            if 1 <= line_no <= len(self.source_lines):
                output.append(self.source_lines[line_no - 1])
            last_line = line_no

        return "".join(output)

    def _get_parents(self, line_number: int) -> list[int]:
        return [src for src, targets in self.graph.items() if line_number in targets]
