import re
from collections import defaultdict, deque
from clang.cindex import Cursor, CursorKind
from typing import List, Set, Tuple, Dict, Optional
import networkx as nx


class FunctionSlicer:
    """
    Builds forward-slice and backward-slice graphs + nested-block map for a function,
    then generates collapsed snippets for slices of various orders.
    Allows parameterized gap-collapse thresholds.
    Ensures any { … } block shown has its closing brace, including the function brace.
    Provides methods for forward, backward, and whole (union) slices.
    """

    def __init__(self, cursor: Cursor, lines: List[str]):
        self._lines = lines
        self._start = cursor.extent.start.line
        self._end   = cursor.extent.end.line

        # Collect all blocks (header, brace_start, brace_end)
        self._blocks: List[Tuple[int,int,int]] = []
        self._collect_blocks(cursor)

        # Collect defs/uses for slicing
        self.defs: Dict[int, Set[str]] = defaultdict(set)
        self.uses: Dict[int, Set[str]] = defaultdict(set)
        self._collect_deps(cursor)

        # Build the forward‑slice graph using networkx
        self._graph = nx.DiGraph()
        self._build_graph()

        # Build the backward‑slice (reverse) graph using networkx
        self._rev_graph = self._graph.reverse(copy=True)

    # ─── Block collection ────────────────────────────────────────────────────────
    def _collect_blocks(self, node: Cursor):
        for child in node.get_children():
            if child.kind in (
                CursorKind.IF_STMT,
                CursorKind.FOR_STMT,
                CursorKind.WHILE_STMT,
                CursorKind.SWITCH_STMT,
                CursorKind.DO_STMT,
            ):
                header = child.extent.start.line
                for grand in child.get_children():
                    if grand.kind == CursorKind.COMPOUND_STMT:
                        bstart, bend = grand.extent.start.line, grand.extent.end.line
                        if self._start < header < self._end:
                            self._blocks.append((header, bstart, bend))
                        break
            self._collect_blocks(child)

    # ─── Def/use collection ─────────────────────────────────────────────────────
    def _collect_deps(self, node: Cursor):
        ln = node.extent.start.line
        if self._start <= ln <= self._end:
            if node.kind == CursorKind.VAR_DECL:
                self.defs[ln].add(node.spelling)
            elif node.kind == CursorKind.DECL_REF_EXPR:
                self.uses[ln].add(node.spelling)
            elif node.kind == CursorKind.BINARY_OPERATOR:
                tokens = list(node.get_tokens())
                for i, tok in enumerate(tokens):
                    if tok.spelling == '=':
                        lhs = {t.spelling for t in tokens[:i] if re.match(r'^[A-Za-z_]\w*$', t.spelling)}
                        rhs = {t.spelling for t in tokens[i+1:] if re.match(r'^[A-Za-z_]\w*$', t.spelling)}
                        self.defs[ln].update(lhs)
                        self.uses[ln].update(rhs)
                        break
            elif node.kind == CursorKind.CALL_EXPR:
                for arg in node.get_arguments() or []:
                    self._collect_arg_defs(arg, ln)
        for c in node.get_children():
            self._collect_deps(c)

    def _collect_arg_defs(self, node: Cursor, ln: int):
        if node.kind == CursorKind.DECL_REF_EXPR:
            self.defs[ln].add(node.spelling)
        for c in node.get_children():
            self._collect_arg_defs(c, ln)

    # ─── Graph building ──────────────────────────────────────────────────────────
    def _build_graph(self):
        uses_inv: Dict[str, Set[int]] = defaultdict(set)
        for ln, vars_used in self.uses.items():
            for v in vars_used:
                uses_inv[v].add(ln)
        for def_ln, vars_defined in self.defs.items():
            for v in vars_defined:
                for use_ln in uses_inv.get(v, []):
                    if use_ln >= def_ln:
                        self._graph.add_edge(def_ln, use_ln)

    # ─── Slice queries ──────────────────────────────────────────────────────────
    def slice_lines(self, line: int, order: Optional[int] = None, backward: Optional[bool] = False) -> Set[int]:
        """
        Return the forward, backward, or whole slice of `line` up to `order` steps (inclusive).
        If `order` is None, returns the full closure.
        backward=False: forward slice
        backward=True: backward slice
        backward=None: union of forward and backward (whole slice)
        """
        # Whole slice case
        if backward is None:
            fwd = self.slice_lines(line, order=order, backward=False)
            bwd = self.slice_lines(line, order=order, backward=True)
            return fwd.union(bwd)

        graph = self._rev_graph if backward else self._graph
        result = {line}
        q = deque([(line, 0)])
        while q:
            cur, depth = q.popleft()
            if order is not None and depth >= order:
                continue
            for nxt in graph.successors(cur):
                if nxt not in result:
                    result.add(nxt)
                    q.append((nxt, depth + 1))
        return result

    def forward_slice_lines(self, line: int) -> Set[int]:
        return self.slice_lines(line, order=None, backward=False)

    def immediate_slice_lines(self, line: int) -> Set[int]:
        return self.slice_lines(line, order=1, backward=False)

    def backward_slice_lines(self, line: int) -> Set[int]:
        return self.slice_lines(line, order=None, backward=True)

    def immediate_backward_slice_lines(self, line: int) -> Set[int]:
        return self.slice_lines(line, order=1, backward=True)

    def whole_slice_lines(self, line: int, order: Optional[int] = None) -> Set[int]:
        return self.slice_lines(line, order=order, backward=None)

    # ─── Snippet rendering ──────────────────────────────────────────────────────
    def slice_snippet(
        self,
        line: int,
        order: Optional[int] = None,
        collapse_threshold: int = 0,
        long_gap_marker: str = "// …",
        backward: Optional[bool] = False
    ) -> str:
        """
        Render a collapsed snippet for the slice of `line` up to `order` steps.
        collapse_threshold: gaps > threshold use long_gap_marker.
        backward(False/True/None): forward/backward/whole slice.
        """
        picks = self._ensure_braces(self.slice_lines(line, order, backward))
        return self._render_snippet(picks, collapse_threshold, long_gap_marker)

    def forward_slice_snippet(self, line: int, collapse_threshold: int = 0) -> str:
        return self.slice_snippet(line, order=None, collapse_threshold=collapse_threshold, backward=False)

    def immediate_slice_snippet(self, line: int, collapse_threshold: int = 5) -> str:
        return self.slice_snippet(line, order=1, collapse_threshold=collapse_threshold, backward=False)

    def backward_slice_snippet(self, line: int, collapse_threshold: int = 0) -> str:
        return self.slice_snippet(line, order=None, collapse_threshold=collapse_threshold, backward=True)

    def immediate_backward_slice_snippet(self, line: int, collapse_threshold: int = 5) -> str:
        return self.slice_snippet(line, order=1, collapse_threshold=collapse_threshold, backward=True)

    def whole_slice_snippet(
        self, line: int, order: Optional[int] = None,
        collapse_threshold: int = 0,
        long_gap_marker: str = "// …"
    ) -> str:
        return self.slice_snippet(line, order=order, collapse_threshold=collapse_threshold, backward=None)

    def _ensure_braces(self, picks: Set[int]) -> Set[int]:
        for header, bstart, bend in self._blocks:
            if header in picks or any(bstart < ln < bend for ln in picks):
                picks.add(header)
                picks.add(bend)
        return picks

    def _render_snippet(
        self,
        picks: Set[int],
        collapse_threshold: int,
        long_gap_marker: str
    ) -> str:
        out: List[str] = []
        sorted_picks = sorted(ln for ln in picks if self._start <= ln <= self._end)

        # Show function signature
        out.append(self._lines[self._start - 1].rstrip("\n"))
        prev = self._start

        for ln in sorted_picks:
            gap = ln - prev - 1
            if gap > collapse_threshold > 0:
                out.append(f"{self._body_indent()}{long_gap_marker}")
            out.append(self._lines[ln - 1].rstrip("\n"))
            prev = ln

        # Tail gap
        tail = self._end - prev
        if tail > collapse_threshold > 0:
            out.append(f"{self._body_indent()}{long_gap_marker}")

        # Always include the function's closing brace
        out.append(self._lines[self._end - 1].rstrip("\n"))

        return "\n".join(out)

    def _body_indent(self) -> str:
        if self._start < len(self._lines):
            m = re.match(r"^(\s+)", self._lines[self._start].rstrip("\n"))
            if m:
                return m.group(1)
        return "  "
