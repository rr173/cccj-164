"""Independent brute-force oracle for the near-neighbour query semantics.

Nothing here calls the solver.  For a small document it enumerates, for every
query leaf, either one of that leaf token's positions or *unused*, checks each
full assignment with a direct recursive predicate written from the published
rules, then computes hits and applies the root-window rules itself.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from nearidx.lexer import token_spans
from nearidx.query import (
    NearNode,
    Node,
    OrNode,
    SeqNode,
    TokenNode,
    iter_leaves,
)


def _leaves_in(node: Node) -> List[TokenNode]:
    return list(iter_leaves(node))


def satisfying(node: Node, assign: Sequence[Optional[int]]) -> bool:
    """Direct predicate for one full witness assignment."""
    if isinstance(node, TokenNode):
        return assign[node.leaf] is not None
    if isinstance(node, OrNode):
        # Exactly one child is the selected hit; every leaf below all other
        # children must be unused (None).  A child that is "partly used" but
        # not a hit means leaves were consumed outside the chosen branch.
        chosen = 0
        for child in node.children:
            ids = [leaf.leaf for leaf in _leaves_in(child)]
            if satisfying(child, assign):
                chosen += 1
            elif any(assign[i] is not None for i in ids):
                return False
        return chosen == 1
    if isinstance(node, SeqNode):
        for child in node.children:
            if not satisfying(child, assign):
                return False
        intervals = [_child_interval(child, assign) for child in node.children]
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            if not e1 < s2:
                return False
            if s2 - e1 - 1 > node.gap:
                return False
        return True
    assert isinstance(node, NearNode)
    if not all(satisfying(child, assign) for child in node.children):
        return False
    used: List[int] = []
    for child in node.children:
        used.extend(_child_used_positions(child, assign))
    if len(set(used)) != len(used):
        return False
    return max(used) - min(used) + 1 <= node.near_width


def _used_positions_of_subtree(
    node: Node, assign: Sequence[Optional[int]]
) -> List[int]:
    """Positions actually used below *node*.

    Unused leaves (the branches an OR did not choose) are skipped, not
    asserted away: a NearNode's used-position set is the union of what its
    children really use, and OR children legitimately leave leaves at -1.
    """
    return [assign[leaf.leaf] for leaf in _leaves_in(node) if assign[leaf.leaf] is not None]


_child_used_positions = _used_positions_of_subtree


def _child_interval(node: Node, assign: Sequence[Optional[int]]) -> Tuple[int, int]:
    used = _child_used_positions(node, assign)
    return min(used), max(used)


def all_hits(root: Node, text: str) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """Every root hit as (start, end, full witness) with -1 for unused."""
    positions_by_token: Dict[str, List[int]] = {}
    for tok in token_spans(text):
        positions_by_token.setdefault(tok.norm, []).append(tok.pos)
    leaves = list(iter_leaves(root))
    per_leaf: List[List[Optional[int]]] = []
    for leaf in leaves:
        opts: List[Optional[int]] = list(positions_by_token.get(leaf.token, []))
        opts.append(None)
        per_leaf.append(opts)
    hits: List[Tuple[int, int, Tuple[int, ...]]] = []
    for assign in itertools.product(*per_leaf):
        if not satisfying(root, assign):
            continue
        used = [p for p in assign if p is not None]
        # -1 marks a leaf that the chosen OR branch did not use, matching the
        # published witness ordering rule.
        hits.append(
            (min(used), max(used), tuple(-1 if p is None else p for p in assign))
        )
    return hits


def root_windows(root: Node, text: str) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """Apply the published root rules independently of the implementation."""
    hits = all_hits(root, text)
    total_leaves = len(list(iter_leaves(root)))
    by_interval: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    for start, end, witness in hits:
        assert len(witness) == total_leaves
        cur = by_interval.get((start, end))
        if cur is None or witness < cur:
            by_interval[(start, end)] = witness
    intervals = sorted(by_interval)
    kept = []
    for iv in intervals:
        strictly_inside = any(
            other != iv and iv[0] <= other[0] and other[1] <= iv[1]
            for other in intervals
        )
        if not strictly_inside:
            kept.append((iv[0], iv[1], by_interval[iv]))
    return kept


def cross_check_solver(root: Node, text: str, solver_hits) -> None:
    """Compare raw solver hits to the oracle (used for debugging tests)."""
    expected = set(all_hits(root, text))
    got = set()
    for hit in solver_hits:
        witness = [-1] * len(list(iter_leaves(root)))
        for leaf, pos in hit.slots:
            witness[leaf] = pos
        used = [p for p in witness if p is not None]
        got.add((min(used), max(used), tuple(witness)))
    assert got == expected, (sorted(expected), sorted(got))
