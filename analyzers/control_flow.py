from collections import defaultdict, deque
import networkx as nx
from clang import cindex
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from intervaltree import IntervalTree

class ControlFlowGraph(nx.DiGraph):
    def __init__(self, cursor: cindex.Cursor, file: List[str]):
        super().__init__()
        self.file = file
        self.extent_index = IntervalTree()
        self._build_cfg(cursor)

    def _linearize(self, line: int, col: int) -> int:
        return line * 10_000 + col

    def _add_extent(self, node_id: str, extent: Tuple[Tuple[int, int], Tuple[int, int]]):
        start, end = extent
        start_val = self._linearize(*start)
        end_val = self._linearize(*end)
        if start_val == end_val:
            end_val += 1  # Make it nonzero width
        self.extent_index[start_val:end_val] = node_id

    def query_extent(self, start_line: int, start_col: int, end_line: int, end_col: int) -> List[str]:
        start_val = self._linearize(start_line, start_col)
        end_val = self._linearize(end_line, end_col)
        matches = self.extent_index.envelop(start_val, end_val)
        return [m.data for m in matches]

    def truncate_label(self, s, keep=3):
        """
        If s is longer than 2*keep, return s[:keep] + "…" + s[-keep:],
        otherwise return s unchanged.
        """
        if len(s) > 2 * keep:
            return f"{s[:keep]}…{s[-keep:]}"
        return s
    
    def trace_to_entry(self, node: str) -> str:
        """
        Trace back the path to '__entry__' and reconstruct a pseudo source-code flow.
        Handles if branches based on taken path (true/false).
        Returns a plain text string.
        """
        if node not in self:
            raise ValueError(f"Node {node} not in CFG.")

        lines = []
        current = node
        while current != '__entry__':
            preds = list(self.predecessors(current))
            if not preds:
                raise ValueError(f"No path from {node} to '__entry__'")
            pred = preds[0]  # pick one (single path assumption)

            # Check if current came via a labeled 'branch' edge
            edge_data = self.get_edge_data(pred, current, default={})
            branch_taken = edge_data.get('branch', None)  # could be True, False, or None

            pred_label = self.nodes[pred].get('label', '')
            current_label = self.nodes[current].get('label', '')

            if branch_taken is True:
                # Took the 'then' branch: expand if normally
                lines.append(f"{pred_label} {{")
            elif branch_taken is False:
                # Took the 'else' branch (false path)
                lines.append(f"{pred_label} {{")
                lines.append("    // skipped body")
                lines.append("}")
            
            lines.append(current_label)
            current = pred

        # Entry node (optional)
        lines.append('entry')

        lines.reverse()
        return '\n'.join(lines)


    def plot_dag_horizontal_level(self, 
                                        node_size=1500, 
                                        font_size=10,
                                        horiz_gap=1.5,
                                        vert_gap=1.0,
                                        keep_chars=3):
        """
        Plot any DAG G left-to-right in true level order, truncating node labels.

        Each node in G should have a 'label' attribute.  Labels longer than
        2*keep_chars will be shown as first keep_chars + “…” + last keep_chars.

        Parameters
        ----------
        G : networkx.DiGraph
            A directed acyclic graph whose nodes have a 'label' attribute.
        node_size : int
            Node size for drawing.
        font_size : int
            Font size for node labels.
        horiz_gap : float
            Horizontal spacing between levels.
        vert_gap : float
            Vertical spacing between nodes within the same level.
        keep_chars : int
            Number of characters to keep at both ends of long labels.
        """
        # 1) find root nodes and do BFS to compute each node's level
        roots = [n for n,d in self.in_degree() if d == 0]
        if not roots:
            raise ValueError("Graph has no roots (in_degree=0); not a DAG or empty.")
        
        level = {}
        dq = deque()
        for r in roots:
            level[r] = 0
            dq.append(r)
        
        while dq:
            u = dq.popleft()
            for v in self.successors(u):
                lvl = level[u] + 1
                if v not in level or lvl < level[v]:
                    level[v] = lvl
                    dq.append(v)
        
        # 2) group nodes by level
        groups = defaultdict(list)
        for n,l in level.items():
            groups[l].append(n)
        
        # 3) assign positions
        pos = {}
        for l, nodes in groups.items():
            mid = (len(nodes) - 1) / 2.0
            for i, n in enumerate(nodes):
                x = l * horiz_gap
                y = (mid - i) * vert_gap
                pos[n] = (x, y)
        
        # 4) prepare truncated labels
        labels = {}
        for n in self.nodes:
            raw = str(self.nodes[n].get('label', n))
            labels[n] = self.truncate_label(raw, keep=keep_chars)
        # make the plot big
        plt.figure(figsize=(10, 6))
        
        # 5) draw
        nx.draw(self, pos,
                labels=labels,
                with_labels=True,
                node_size=node_size,
                font_size=font_size,
                arrows=True,
                edge_color='gray')
        plt.axis('off')
        plt.show()

    def _build_cfg(self, cursor: cindex.Cursor):
        entry = '__entry__'
        exit = '__exit__'
        self.add_node(entry, label='entry', extent=None)
        self.add_node(exit, label='exit', extent=None)

        def extent_key(cur: cindex.Cursor) -> str:
            ext = cur.extent
            start = (ext.start.line, ext.start.column)
            end = (ext.end.line, ext.end.column)
            return f"{start[0]}:{start[1]}-{end[0]}:{end[1]}"

        def get_label(extent: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
            (sl, sc), (el, ec) = extent
            if sl == el:
                return self.file[sl-1][sc-1:ec-1]
            lines = [self.file[sl-1][sc-1:]] + self.file[sl:el-1] + [self.file[el-1][:ec-1]]
            return ''.join(lines)

        def make_node(cur: cindex.Cursor) -> str:
            key = extent_key(cur)
            ext = cur.extent
            start = (ext.start.line, ext.start.column)
            end = (ext.end.line, ext.end.column)

            if cur.kind in (
                cindex.CursorKind.IF_STMT,
                cindex.CursorKind.FOR_STMT,
                cindex.CursorKind.WHILE_STMT,
                cindex.CursorKind.DO_STMT,
                cindex.CursorKind.SWITCH_STMT,
            ):
                tokens = list(cur.get_tokens())
                header_tokens = []
                for tok in tokens:
                    header_tokens.append(tok.spelling)
                    if tok.spelling == ')':
                        break
                    if cur.kind == cindex.CursorKind.DO_STMT and tok.spelling.lower() == 'do':
                        break
                label = ''.join(header_tokens)
            else:
                label = get_label((start, end))

            self.add_node(key, cursor=cur, label=label, extent=(start, end))
            self._add_extent(key, (start, end))
            return key

        def build(cur: cindex.Cursor, preds: List[str]) -> List[str]:
            exits: List[str] = []
            kind = cur.kind

            if kind in (
                cindex.CursorKind.COMPOUND_STMT,
                cindex.CursorKind.FUNCTION_DECL,
                cindex.CursorKind.CXX_METHOD,
            ):
                current_preds = preds
                for child in cur.get_children():
                    current_preds = build(child, current_preds)
                return current_preds

            if kind == cindex.CursorKind.IF_STMT:
                node = make_node(cur)
                for p in preds:
                    self.add_edge(p, node)  # No label needed here
                
                children = list(cur.get_children())
                
                # Then branch
                if len(children) > 1:
                    then_exits = build(children[1], [node])
                    for ex in then_exits:
                        self.add_edge(node, ex, branch=True)  # True branch

                else:
                    then_exits = [node]
                    
                # Else branch
                if len(children) > 2:
                    else_exits = build(children[2], [node])
                    for ex in else_exits:
                        self.add_edge(node, ex, branch=False)  # False branch
                else:
                    else_exits = [node]
                
                return then_exits + else_exits


            if kind in (cindex.CursorKind.WHILE_STMT, cindex.CursorKind.DO_STMT):
                node = make_node(cur)
                for p in preds:
                    self.add_edge(p, node)
                body = None
                for c in cur.get_children():
                    if c.kind not in (cindex.CursorKind.PAREN_EXPR,):
                        body = c
                        break
                body_exits = build(body, [node]) if body else [node]
                for b in body_exits:
                    self.add_edge(b, node)
                return [node]

            if kind == cindex.CursorKind.FOR_STMT:
                node = make_node(cur)
                for p in preds:
                    self.add_edge(p, node)
                body = list(cur.get_children())[-1]
                body_exits = build(body, [node])
                for b in body_exits:
                    self.add_edge(b, node)
                return [node]

            if kind == cindex.CursorKind.SWITCH_STMT:
                node = make_node(cur)
                for p in preds:
                    self.add_edge(p, node)
                exits = []
                for child in cur.get_children():
                    if child.kind in (
                        cindex.CursorKind.CASE_STMT,
                        cindex.CursorKind.DEFAULT_STMT,
                    ):
                        exits.extend(build(child, [node]))
                return exits or [node]

            if kind == cindex.CursorKind.GOTO_STMT:
                node = make_node(cur)
                for p in preds:
                    self.add_edge(p, node)
                target = cur.get_definition()
                if target:
                    tgt_key = extent_key(target)
                    tgt_ext = ((target.extent.start.line, target.extent.start.column),
                               (target.extent.end.line, target.extent.end.column))
                    tgt_label = get_label(tgt_ext)
                    self.add_node(tgt_key, cursor=target, label=tgt_label, extent=tgt_ext)
                    self._add_extent(tgt_key, tgt_ext)
                    self.add_edge(node, tgt_key)
                return []

            if kind == cindex.CursorKind.RETURN_STMT:
                node = make_node(cur)
                for p in preds:
                    self.add_edge(p, node)
                self.add_edge(node, exit)
                return []

            node = make_node(cur)
            for p in preds:
                self.add_edge(p, node)
            return [node]

        exits = build(cursor, [entry])
        for e in exits:
            self.add_edge(e, exit)