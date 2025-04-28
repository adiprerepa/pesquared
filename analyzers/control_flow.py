import networkx as nx
from clang import cindex
from clang.cindex import CursorKind
from typing import List, Set, Tuple, Dict
from intervaltree import IntervalTree
from dataclasses import dataclass

@dataclass(frozen=True)
class Extent:
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def __str__(self) -> str:
        return f"{self.start_line}:{self.start_col}-{self.end_line}:{self.end_col}"

class ControlFlowGraph(nx.DiGraph):
    def __init__(self, cursor: cindex.Cursor, file: List[str]):
        super().__init__()
        self.file = file
        self.extent_index = IntervalTree()
        # For goto/label resolution
        self._label_map: Dict[str, str] = {}
        self._pending_gotos: List[Tuple[str, str]] = []  # (label, node_id)
        # Build stacks for break/continue targets
        self._loop_stack: List[str] = []
        self._switch_stack: List[str] = []
        self._line_to_node: Dict[int, str] = {}
        self._build_cfg(cursor)
        
    def _linearize(self, line: int, col: int) -> int:
        return line * 10_000 + col

    def _add_extent(self, node_id: str, extent: Extent):
        start_val = self._linearize(extent.start_line, extent.start_col)
        end_val = self._linearize(extent.end_line, extent.end_col)
        if start_val == end_val:
            end_val += 1
        self.extent_index[start_val:end_val] = node_id

    def query_extent(self, start_line: int, start_col: int, end_line: int, end_col: int) -> List[str]:
        start_val = self._linearize(start_line, start_col)
        end_val = self._linearize(end_line, end_col)
        matches = self.extent_index.envelop(start_val, end_val)
        return [m.data for m in matches]

    def truncate_label(self, s: str, keep: int = 3) -> str:
        if len(s) > 2 * keep:
            return f"{s[:keep]}…{s[-keep:]}"
        return s

    def trace_to_entry(self, node: str) -> str:
        if node not in self:
            raise ValueError(f"Node {node} not in CFG.")
        lines = []
        current = node
        while current != '__entry__':
            preds = list(self.predecessors(current))
            if not preds:
                raise ValueError(f"No path from {node} to '__entry__'")
            pred = preds[0]
            edge_data = self.get_edge_data(pred, current, default={})
            branch_taken = edge_data.get('branch', None)
            pred_label = self.nodes[pred].get('label', '')
            cur_label = self.nodes[current].get('label', '')
            if branch_taken is True:
                lines.append(f"{pred_label} {{  // then")
            elif branch_taken is False:
                lines.append(f"{pred_label} {{  // else")
            lines.append(cur_label)
            current = pred
        lines.append('entry')
        return '\n'.join(reversed(lines))

    def plot_dag_horizontal_level(self,
                                  node_size=1500,
                                  font_size=10,
                                  horiz_gap=1.5,
                                  vert_gap=1.0,
                                  keep_chars=3):
        import matplotlib.pyplot as plt
        from collections import deque, defaultdict

        kind_colors = {
            CursorKind.IF_STMT: 'lightblue',
            CursorKind.FOR_STMT: 'lightgreen',
            CursorKind.WHILE_STMT: 'lightcoral',
            CursorKind.DO_STMT: 'gold',
            CursorKind.SWITCH_STMT: 'violet',
            CursorKind.RETURN_STMT: 'orange',
        }

        if nx.is_directed_acyclic_graph(self):
            G_levels = self
            node_to_scc = {n: n for n in self.nodes}
        else:
            print("Graph is not a DAG; using condensation.")
            condensed = nx.condensation(self)
            node_to_scc = condensed.graph['mapping']
            G_levels = condensed

        roots = [n for n, d in G_levels.in_degree() if d == 0]
        if not roots:
            raise ValueError("Graph has no roots; not a DAG or empty.")

        level = {}
        dq = deque()
        for r in roots:
            level[r] = 0
            dq.append(r)
        while dq:
            u = dq.popleft()
            for v in G_levels.successors(u):
                lvl = level[u] + 1
                if v not in level or lvl < level[v]:
                    level[v] = lvl
                    dq.append(v)

        groups = defaultdict(list)
        for n in self.nodes:
            l = level.get(node_to_scc.get(n, n), 0)
            groups[l].append(n)

        pos = {}
        for l, nodes in groups.items():
            mid = (len(nodes) - 1) / 2.0
            for i, n in enumerate(nodes):
                pos[n] = (l * horiz_gap, (mid - i) * vert_gap)

        labels = {n: self.truncate_label(str(self.nodes[n].get('label', n)), keep_chars)
                  for n in self.nodes}
        node_colors = [kind_colors.get(self.nodes[n].get('kind'), 'gray') for n in self.nodes]

        plt.figure(figsize=(12, 8))
        nx.draw(self, pos,
                labels=labels,
                with_labels=True,
                node_size=node_size,
                font_size=font_size,
                arrows=True,
                edge_color='gray',
                node_color=node_colors,
                connectionstyle='arc3,rad=0.2')
        plt.axis('off')
        plt.show()
    
    def _get_or_create_line_node(self, line: int) -> str:
        if line not in self._line_to_node:
            label = self.file[line - 1].strip()  # Whole line text
            node_id = f"line_{line}"
            self.add_node(node_id, label=label, extent=None)
            ext = Extent(line, 1, line, len(label) + 1)
            self._add_extent(node_id, ext)
            self._line_to_node[line] = node_id
        return self._line_to_node[line]


    def _build_cfg(self, cursor: cindex.Cursor):
        entry, exit = '__entry__', '__exit__'
        self.add_node(entry, label='entry', extent=None)
        self.add_node(exit, label='exit', extent=None)
        def build(cur: cindex.Cursor, preds: List[str], loop_stack: List[str], switch_stack: List[str]) -> List[str]:
            kind = cur.kind

            # --- Step 1: Create (or reuse) a node for this line
            line = cur.extent.start.line
            node = self._get_or_create_line_node(line)

            # --- Step 2: Connect predecessors
            if preds:
                for p in preds:
                    if not self.has_edge(p, node):
                        self.add_edge(p, node)

            # --- Step 3: Handle special control flow
            if kind == CursorKind.IF_STMT:
                children = list(cur.get_children())
                cond_preds = [node]
                if len(children) >= 1:
                    cond_preds = build(children[0], cond_preds, loop_stack, switch_stack)
                then_preds = cond_preds
                else_preds = cond_preds
                if len(children) >= 2:
                    then_preds = build(children[1], cond_preds, loop_stack, switch_stack)
                    for t in then_preds:
                        self.add_edge(node, t, branch=True)
                if len(children) >= 3:
                    else_preds = build(children[2], cond_preds, loop_stack, switch_stack)
                    for e in else_preds:
                        self.add_edge(node, e, branch=False)
                return then_preds + else_preds

            elif kind in (CursorKind.WHILE_STMT, CursorKind.DO_STMT, CursorKind.FOR_STMT):
                loop_stack.append(node)
                body = next((c for c in cur.get_children() if c.kind != CursorKind.PAREN_EXPR), None)
                body_preds = build(body, [node], loop_stack, switch_stack) if body else [node]
                for b in body_preds:
                    self.add_edge(b, node)
                loop_stack.pop()
                return [node]

            elif kind == CursorKind.SWITCH_STMT:
                switch_stack.append(node)
                exits: List[str] = []
                for ch in cur.get_children():
                    exits.extend(build(ch, [node], loop_stack, switch_stack))
                switch_stack.pop()
                return exits or [node]

            elif kind == CursorKind.LABEL_STMT:
                lbl = cur.spelling
                self._label_map[lbl] = node
                for want, nid in list(self._pending_gotos):
                    if want == lbl:
                        self.add_edge(nid, node)
                        self._pending_gotos.remove((want, nid))
                cur_preds = [node]
                for ch in cur.get_children():
                    cur_preds = build(ch, cur_preds, loop_stack, switch_stack)
                return cur_preds

            elif kind == CursorKind.GOTO_STMT:
                lbl = next(cur.get_children()).spelling
                if lbl in self._label_map:
                    self.add_edge(node, self._label_map[lbl])
                else:
                    self._pending_gotos.append((lbl, node))
                return []

            elif kind == CursorKind.BREAK_STMT:
                target = switch_stack[-1] if switch_stack else (loop_stack[-1] if loop_stack else exit)
                self.add_edge(node, target)
                return []

            elif kind == CursorKind.CONTINUE_STMT:
                target = loop_stack[-1] if loop_stack else exit
                self.add_edge(node, target)
                return []

            elif kind in (CursorKind.RETURN_STMT, CursorKind.CXX_THROW_EXPR):
                self.add_edge(node, exit)
                return []

            elif cur.spelling in ('exit', 'abort'):
                self.add_edge(node, exit)
                return []

            # --- Step 4: Default: build children
            cur_preds = [node]
            for ch in cur.get_children():
                cur_preds = build(ch, cur_preds, loop_stack, switch_stack)

            return cur_preds

        # --- Start building from the root cursor
        exits = build(cursor, [entry], self._loop_stack, self._switch_stack)

        for lbl, nid in list(self._pending_gotos):
            if lbl in self._label_map:
                self.add_edge(nid, self._label_map[lbl])
                self._pending_gotos.remove((lbl, nid))

        for e in exits:
            self.add_edge(e, exit)


    def cyclomatic_complexity(self) -> int:
        n = self.number_of_nodes()
        e = self.number_of_edges()
        p = nx.number_weakly_connected_components(nx.DiGraph(self))
        return e - n + 2 * p

    def cognitive_complexity(self,
                             control_node_types=None) -> int:
        if control_node_types is None:
            control_node_types = {CursorKind.IF_STMT, CursorKind.FOR_STMT,
                                  CursorKind.WHILE_STMT, CursorKind.DO_STMT,
                                  CursorKind.SWITCH_STMT, CursorKind.CASE_STMT,
                                  CursorKind.DEFAULT_STMT,
                                  CursorKind.CXX_CATCH_STMT, CursorKind.CXX_TRY_STMT}
        complexity = 0
        entry_nodes = [n for n, d in self.in_degree() if d == 0]
        visited: Set[str] = set()
        def dfs(node: str, nesting: int):
            nonlocal complexity
            if node in visited:
                return
            visited.add(node)
            node_type = self.nodes[node].get('kind')
            if node_type in control_node_types:
                complexity += 1 + nesting
                nesting += 1
            for succ in self.successors(node):
                dfs(succ, nesting)
            visited.remove(node)
        for start in entry_nodes:
            dfs(start, 0)
        return complexity

    def compute_dominator_tree(self, start: str = '__entry__') -> nx.DiGraph:
        from networkx.algorithms.dominance import immediate_dominators
        idoms = immediate_dominators(self, start)
        dom_tree = nx.DiGraph()
        for node, idom in idoms.items():
            if node != start:
                dom_tree.add_edge(idom, node)
        return dom_tree

    def unreachable_nodes(self) -> List[str]:
        entry = '__entry__'
        reachable = {entry} | nx.descendants(self, entry)
        return [n for n in self.nodes if n not in reachable]