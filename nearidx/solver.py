"""Candidate-window driven evaluation of the query tree.

Why this is not a naive whole-corpus scan
-----------------------------------------
1. The query is reduced to *requirement groups* (see
   :func:`requirement_groups`).  A document must contain at least one token
   from every group, otherwise no root hit can exist.  SQLite intersects the
   inverted ``postings`` table for those groups, so documents missing a
   required word never reach the solver and their bodies are never read.
2. Only the posting rows of the query's leaf tokens, for candidate docs, are
   loaded -- never all documents' text.
3. The original text is fetched only for documents that produced deliverable
   root windows, and only the returned page's documents are sliced.

What is kept and what is safely pruned
--------------------------------------
Every partial hit keeps the exact leaf -> position assignment (the witness).
Identity is decided by that assignment: two partial hits with the same
assignment are literally the same hit and are deduplicated.  Hits that have
the *same hull interval but different used positions* are never merged --
merging them would lose witnesses that a sibling later needs, which is why
internal nodes keep all assignments and the minimum-window rule is applied to
root intervals only.

Safe pruning happens only where it cannot change any witness:

* a child with no postings kills the whole conjunction;
* ``SEQ`` rejects a next child whose interval starts at or before the previous
  interval's end, or leaves more than ``gap`` unoccupied positions between
  them;
* ``NEAR`` rejects a combination whose used-position sets intersect, or
  whose hull grows wider than ``width``;
* ``OR`` children are alternatives, so only exact witness duplicates are
  dropped.

No shortest-interval selection is performed below the root.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .db import AccessTrace, Index
from .query import (
    Cursor,
    CursorError,
    NearNode,
    Node,
    OrNode,
    SeqNode,
    TokenNode,
    decode_cursor,
    encode_cursor,
    iter_leaves,
    query_signature,
)

__all__ = [
    "SolverLimitError",
    "QueryError",
    "Window",
    "LeafUse",
    "SearchPage",
    "requirement_groups",
    "solve_document",
    "search",
    "select_root_windows",
]


class QueryError(ValueError):
    pass


class SolverLimitError(QueryError):
    """Raised when intermediate hit enumeration exceeds the safety budget."""


# ---------------------------------------------------------------------------
# Hit representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Hit:
    """A partial or root hit.

    ``slots`` is a tuple of ``(leaf_id, position)`` pairs, holding the actual
    positions this hit assigns.  ``used`` is the set of those positions, used
    for the ``NEAR`` pairwise-disjointness check, and ``start``/``end`` is
    their closed hull.  All three follow from ``slots``; keeping them
    precomputed is purely an optimisation.  Sets hold one entry per used leaf,
    so position magnitude (large documents) does not inflate them.
    """

    slots: Tuple[Tuple[int, int], ...]
    used: frozenset
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start + 1


# ---------------------------------------------------------------------------
# Requirements (safe candidate filter)
# ---------------------------------------------------------------------------

def requirement_groups(node: Node) -> List[frozenset]:
    """Necessary token groups: a matching doc needs one token from each group.

    ``SEQ``/``NEAR`` require every child, so their groups concatenate.
    ``OR`` requires one alternative, so all leaves below it form one
    disjunction.  This is deliberately weaker than the exact semantics but
    never excludes a document that could match.
    """
    if isinstance(node, TokenNode):
        return [frozenset((node.token,))]
    if isinstance(node, OrNode):
        toks = frozenset(leaf.token for leaf in iter_leaves(node))
        return [toks]
    groups: List[frozenset] = []
    for child in node.children:
        groups.extend(requirement_groups(child))
    return groups


def _intersect_groups(groups: Sequence[frozenset]) -> List[frozenset]:
    """Merge groups with equal token sets and drop ones implied by another.

    Because each group must be satisfied, a group whose token set is a subset
    of another group's set is redundant for the necessary condition.
    """
    uniq: List[frozenset] = []
    for g in groups:
        if g and g not in uniq:
            uniq.append(g)
    kept: List[frozenset] = []
    for g in uniq:
        if any(g < other for other in uniq):
            continue
        kept.append(g)
    return kept


# ---------------------------------------------------------------------------
# One-document solver
# ---------------------------------------------------------------------------

class _DocSolver:
    def __init__(
        self,
        root: Node,
        total_leaves: int,
        postings: Dict[str, List[Tuple[int, int, int, int, int]]],
        *,
        max_hits: int = 2_000_000,
    ):
        self.root = root
        self.total_leaves = total_leaves
        self.postings = postings
        self.max_hits = max_hits
        self._memo: Dict[int, List[Hit]] = {}

    # -- entry -----------------------------------------------------------
    def solve(self) -> List[Hit]:
        return self._hits(self.root)

    def _guard(self, node: Node, hits: List[Hit]) -> List[Hit]:
        if len(hits) > self.max_hits:
            raise SolverLimitError(
                f"more than {self.max_hits} partial hits while evaluating "
                f"{type(node).__name__}; narrow the query"
            )
        return hits

    # -- recursion -------------------------------------------------------
    def _hits(self, node: Node) -> List[Hit]:
        key = id(node)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        if isinstance(node, TokenNode):
            rows = self.postings.get(node.token, [])
            hits = [
                Hit(
                    slots=((node.leaf, pos),),
                    used=frozenset((pos,)),
                    start=pos,
                    end=pos,
                )
                for pos, _cs, _ce, _bs, _be in rows
            ]
        elif isinstance(node, OrNode):
            hits = self._or(node)
        elif isinstance(node, SeqNode):
            hits = self._seq(node)
        else:
            assert isinstance(node, NearNode)
            hits = self._near(node)
        hits = self._guard(node, hits)
        self._memo[key] = hits
        return hits

    @staticmethod
    def _dedupe_identical(hits: Iterable[Hit]) -> List[Hit]:
        seen = set()
        out: List[Hit] = []
        for h in hits:
            if h.slots in seen:
                continue
            seen.add(h.slots)
            out.append(h)
        return out

    def _or(self, node: OrNode) -> List[Hit]:
        combined: List[Hit] = []
        for child in node.children:
            combined.extend(self._hits(child))
        # Only exact witness duplicates are removed; same interval with a
        # different used-position set stays.
        return self._dedupe_identical(combined)

    def _seq(self, node: SeqNode) -> List[Hit]:
        gap = node.gap
        children = node.children
        states: List[Hit] = list(self._hits(children[0]))
        for child in children[1:]:
            next_hits = self._hits(child)
            if not states or not next_hits:
                return []
            new_states: List[Hit] = []
            for st in states:
                for h in next_hits:
                    # Child intervals are ordered: the previous child's hull
                    # must end strictly before this child's hull starts.
                    if h.start <= st.end:
                        continue
                    # number of unused positions strictly between intervals
                    if h.start - st.end - 1 > gap:
                        continue
                    new_states.append(
                        Hit(
                            slots=st.slots + h.slots,
                            used=st.used | h.used,
                            start=st.start,
                            end=h.end,
                        )
                    )
            states = self._dedupe_identical(new_states)
            if not states:
                return []
        return states

    def _near(self, node: NearNode) -> List[Hit]:
        width = node.near_width
        children = node.children
        states: List[Hit] = [
            h for h in self._hits(children[0]) if h.width <= width
        ]
        for child in children[1:]:
            next_hits = [h for h in self._hits(child) if h.width <= width]
            if not states or not next_hits:
                return []
            new_states: List[Hit] = []
            for st in states:
                for h in next_hits:
                    # Children may have overlapping intervals; only the actual
                    # used positions must be pairwise disjoint.
                    if not st.used.isdisjoint(h.used):
                        continue
                    start = st.start if st.start < h.start else h.start
                    end = st.end if st.end > h.end else h.end
                    if end - start + 1 > width:
                        continue
                    new_states.append(
                        Hit(
                            slots=st.slots + h.slots,
                            used=st.used | h.used,
                            start=start,
                            end=end,
                        )
                    )
            states = self._dedupe_identical(new_states)
            if not states:
                return []
        return states


def solve_document(
    root: Node,
    postings: Dict[str, List[Tuple[int, int, int, int, int]]],
    *,
    max_hits: int = 2_000_000,
) -> List[Hit]:
    """All root hits for one document, before minimum-window filtering."""
    total = sum(1 for _ in iter_leaves(root))
    return _DocSolver(root, total, postings, max_hits=max_hits).solve()


# ---------------------------------------------------------------------------
# Root window selection
# ---------------------------------------------------------------------------

def _witness(hit: Hit, total_leaves: int) -> Tuple[int, ...]:
    """Expand slots to a full witness tuple; unused OR leaves are -1."""
    witness = [-1] * total_leaves
    for leaf, pos in hit.slots:
        witness[leaf] = pos
    return tuple(witness)


def select_root_windows(
    hits: Iterable[Hit], total_leaves: int
) -> List[Tuple[int, int, Tuple[int, ...], Hit]]:
    """Apply the root rules: one witness per interval, then minimal windows.

    Same interval -> exactly one witness, the lexicographically smallest leaf
    position sequence (unused leaves count as -1).  Then drop every interval
    that strictly contains another qualifying root interval.  A strict
    containment cannot cross documents, so the caller may apply this per doc.
    """
    best: Dict[Tuple[int, int], Tuple[Tuple[int, ...], Hit]] = {}
    for h in hits:
        witness = _witness(h, total_leaves)
        key = (h.start, h.end)
        cur = best.get(key)
        if cur is None or witness < cur[0]:
            best[key] = (witness, h)
    items = [
        (start, end, witness, h) for (start, end), (witness, h) in best.items()
    ]
    items.sort(key=lambda it: (it[0], it[1], it[2]))
    # Rule: delete the *larger* interval when another qualifying root interval
    # is strictly inside it.  So (s, e) is dropped iff another interval has
    # s <= s2 and e2 <= e and (s, e) != (s2, e2).  Items are sorted by
    # (start, end): within a start group only the shortest end can contain its
    # siblings, and for later start groups only their minimum end matters.
    # A suffix minimum over start groups makes this linear after sorting.
    groups: List[Tuple[int, List[Tuple[int, int, Tuple[int, ...], Hit]]]] = []
    for item in items:
        if groups and groups[-1][0] == item[0]:
            groups[-1][1].append(item)
        else:
            groups.append((item[0], [item]))
    # min_end_of_later[gi] = smallest end among start groups after gi, or -1.
    min_end_of_later: List[int] = [-1] * len(groups)
    running = -1
    for gi in range(len(groups) - 1, -1, -1):
        min_end_of_later[gi] = running
        group_min = min(it[1] for it in groups[gi][1])
        running = group_min if running == -1 else min(running, group_min)
    kept: List[Tuple[int, int, Tuple[int, ...], Hit]] = []
    for gi, (_start, group) in enumerate(groups):
        smallest_here = min(it[1] for it in group)
        later = min_end_of_later[gi]
        for start, end, witness, h in group:
            covered = end > smallest_here or (later != -1 and end >= later)
            if not covered:
                kept.append((start, end, witness, h))
    return kept


# ---------------------------------------------------------------------------
# Search / pagination
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LeafUse:
    leaf: int
    token: str            # folded lookup key
    spelling: str         # token as written in the query
    positions: List[int]  # empty when the leaf is unused

    def as_dict(self) -> Dict[str, object]:
        return {
            "leaf": self.leaf,
            "token": self.token,
            "spelling": self.spelling,
            "positions": list(self.positions),
            "used": bool(self.positions),
        }


@dataclass(slots=True)
class Window:
    doc_id: str
    start: int
    end: int
    witness: Tuple[int, ...]
    leaves: List[LeafUse]
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    snippet: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
            "width": self.end - self.start + 1,
            "witness": list(self.witness),
            "leaves": [leaf.as_dict() for leaf in self.leaves],
            "char_span": [self.char_start, self.char_end],
            "byte_span": [self.byte_start, self.byte_end],
            "snippet": self.snippet,
        }


@dataclass(slots=True)
class SearchPage:
    windows: List[Window]
    offset: int
    limit: int
    next_cursor: Optional[str]
    has_more: bool
    query_signature: str
    index_fingerprint: str
    trace: Dict[str, object]
    total_candidate_docs: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "windows": [w.as_dict() for w in self.windows],
            "offset": self.offset,
            "limit": self.limit,
            "has_more": self.has_more,
            "next_cursor": self.next_cursor,
            "query_signature": self.query_signature,
            "index_fingerprint": self.index_fingerprint,
            "trace": self.trace,
            "total_candidate_docs": self.total_candidate_docs,
        }


def _candidate_docs(index: Index, root: Node, trace: AccessTrace) -> List[str]:
    """Intersect the inverted table for the query's necessary token groups.

    A document missing any group cannot match, so it is never returned here
    and its body is never fetched.  The intersection runs inside SQLite over
    the ``postings`` primary key, not over document text.
    """
    groups = _intersect_groups(requirement_groups(root))
    if not groups:
        return []
    conn = index.conn  # internal handle; evidence is counted in one place
    sql = " INTERSECT ".join(
        "SELECT DISTINCT doc_id FROM postings WHERE token IN ("
        + ",".join("?" for _ in g)
        + ")"
        for g in groups
    )
    params: List[str] = []
    for g in groups:
        params.extend(sorted(g))
    trace.postings_queries += 1
    docs = [
        r[0]
        for r in conn.execute(
            f"SELECT doc_id FROM ({sql}) ORDER BY doc_id", params
        )
    ]
    trace.candidate_docs = len(docs)
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    trace.missing_token_docs = max(0, total_docs - len(docs))
    return docs


def _leaf_order(root: Node) -> List[TokenNode]:
    return list(iter_leaves(root))


def _span_by_position(
    leaves: Sequence[TokenNode],
    rows_by_token: Dict[str, List[Tuple[int, int, int, int, int]]],
) -> Dict[int, Tuple[int, int, int, int]]:
    """Map token position -> (char_start, char_end, byte_start, byte_end).

    Only the query's leaf tokens are fetched, so any position used by a hit is
    present here.  Distinct leaves with the same folded token share the rows.
    """
    span_by_pos: Dict[int, Tuple[int, int, int, int]] = {}
    for leaf in leaves:
        for pos, cs, ce, bs, be in rows_by_token.get(leaf.token, []):
            span_by_pos[pos] = (cs, ce, bs, be)
    return span_by_pos


def _window_from_hit(
    index: Index,
    doc_id: str,
    hit: Hit,
    start: int,
    end: int,
    witness: Tuple[int, ...],
    leaves: Sequence[TokenNode],
    spans: Dict[int, Tuple[int, int, int, int]],
    text_cache: Dict[str, str],
) -> Window:
    try:
        char_start, _cse, byte_start, _bse = spans[start]
        _css, char_end, _bss, byte_end = spans[end]
    except KeyError as exc:  # pragma: no cover - would signal an index bug
        raise QueryError(
            f"window boundary position {exc} is not a used leaf position in "
            f"document {doc_id!r}"
        ) from None
    if doc_id not in text_cache:
        # The original text is fetched only for documents with deliverable
        # windows; the snippet is cut from it, never re-joined from tokens.
        text_cache[doc_id] = index.text(doc_id)
    text = text_cache[doc_id]
    if char_start < 0 or char_end > len(text) or char_end < char_start:
        raise QueryError(f"invalid snippet span for document {doc_id!r}")
    snippet = text[char_start:char_end]

    slots_by_leaf: Dict[int, List[int]] = {}
    for leaf_id, pos in hit.slots:
        slots_by_leaf.setdefault(leaf_id, []).append(pos)
    leaf_uses: List[LeafUse] = []
    for leaf in leaves:
        pos_list = sorted(slots_by_leaf.get(leaf.leaf, ()))
        leaf_uses.append(
            LeafUse(
                leaf=leaf.leaf,
                token=leaf.token,
                spelling=leaf.spelling,
                positions=pos_list,
            )
        )
    return Window(
        doc_id=doc_id,
        start=start,
        end=end,
        witness=witness,
        leaves=leaf_uses,
        char_start=char_start,
        char_end=char_end,
        byte_start=byte_start,
        byte_end=byte_end,
        snippet=snippet,
    )


def _stream_root_windows(
    index: Index,
    root: Node,
    docs: Sequence[str],
    leaves: Sequence[TokenNode],
    distinct_tokens: Sequence[str],
    total_leaves: int,
    *,
    max_hits: int,
    trace: AccessTrace,
    resume: Optional[Tuple[Optional[str], int]] = None,
) -> Iterator[
    Tuple[str, int, int, Tuple[int, ...], Hit, Dict[int, Tuple[int, int, int, int]]]
]:
    """Yield root windows of *docs* in delivery order, one at a time.

    Each item is ``(doc_id, start, end, witness, hit, spans)`` where ``spans``
    maps the document's token positions to their original char/byte spans, so
    a delivered window can be materialised without re-fetching postings.  The
    stream is lazy: a document is solved only when the caller pulls far
    enough to reach it, so stopping early never touches later documents.

    ``resume`` may be a ``(doc_id, skip)`` pair: streaming then starts at that
    candidate document and skips its first ``skip`` windows, which an earlier
    page already delivered.  A resume point that does not match this index
    generation raises :class:`CursorError` -- the state belongs to a cursor
    that was not issued against exactly this content.
    """
    resume_doc, resume_skip = resume if resume is not None else (None, 0)
    start_at = 0
    if resume_doc is not None:
        i = bisect.bisect_left(docs, resume_doc)
        if i >= len(docs) or docs[i] != resume_doc:
            raise CursorError(
                "cursor resume state does not match this index generation: "
                f"{resume_doc!r} is not a candidate document"
            )
        start_at = i
    for doc_id in docs[start_at:]:
        on_resume_doc = resume_doc is not None and doc_id == resume_doc
        trace.docs_considered += 1
        rows_by_token = index.postings_for_doc(distinct_tokens, doc_id)
        trace.docs_solved += 1
        hits = solve_document(root, rows_by_token, max_hits=max_hits)
        windows = select_root_windows(hits, total_leaves) if hits else []
        if on_resume_doc and resume_skip > len(windows):
            raise CursorError(
                "cursor resume state does not match this index generation: "
                f"document {resume_doc!r} has {len(windows)} root windows, "
                f"the cursor skips {resume_skip}"
            )
        if not windows:
            continue
        spans = _span_by_position(leaves, rows_by_token)
        for j, (start, end, witness, hit) in enumerate(windows):
            if on_resume_doc and j < resume_skip:
                continue
            yield doc_id, start, end, witness, hit, spans


def search(
    index: Index,
    root: Node,
    *,
    limit: int = 20,
    offset: int = 0,
    cursor: Optional[str] = None,
    max_hits: int = 2_000_000,
    stop_early: bool = True,
    trace: Optional[AccessTrace] = None,
) -> SearchPage:
    """Run *root* over the index and return one deterministic page.

    Ordering is fixed: doc_id, start, end.  A cursor encodes the offset, page
    size, query signature and the fingerprint of the index generation the
    first page was served from.  Follow-up pages are resolved against that
    exact generation -- even after a refresh published a newer one and the
    index was reopened in another process -- so a paginating reader finishes
    the generation it started, with the original windows, order and page
    boundaries.  A cursor whose generation is no longer retained raises
    :class:`GenerationExpiredError` instead of silently reading current data.
    """
    if limit < 1:
        raise QueryError("limit must be at least 1")
    signature = query_signature(root)
    if cursor is not None:
        cur = decode_cursor(cursor)
        if cur.query_sig != signature:
            raise CursorError("cursor was issued for a different query")
        # Bind the whole call to the cursor's generation.  ``for_generation``
        # returns the handle itself when the cursor names the generation this
        # handle already serves, so every read below stays inside one
        # generation's file from candidate lookup to snippet slicing.
        index = index.for_generation(cur.index_sig)
        offset = cur.offset
        limit = cur.page_size
    if offset < 0:
        raise QueryError("offset must be non-negative")

    # Evidence is per query: a fresh trace records exactly what this call
    # touched, independent of earlier calls on the same handle.  A caller may
    # pass its own trace object to inspect the evidence after the call.
    if trace is None:
        trace = AccessTrace()
    index.trace = trace
    leaves = _leaf_order(root)
    total_leaves = len(leaves)
    distinct_tokens = list(dict.fromkeys(leaf.token for leaf in leaves))
    docs = _candidate_docs(index, root, trace)

    # Minimal-window filtering is per document, and the output order is
    # doc_id then start/end, so once enough windows exist for this page we can
    # stop before touching later documents: they could only append.
    selected: List[Tuple[str, int, int, Tuple[int, ...], Hit]] = []
    # One window beyond the requested page is enough to prove ``has_more``.
    # ``docs`` is ordered by doc_id and windows within a doc are ordered by
    # (start, end), so pulling keeps the global delivery order without any
    # repeated sort.  Stopping there is safe: later docs have larger doc_ids,
    # and they cannot affect windows already selected because minimum-window
    # filtering is strictly per document.
    target = offset + limit + 1
    for doc_id, start, end, witness, hit, _spans in _stream_root_windows(
        index,
        root,
        docs,
        leaves,
        distinct_tokens,
        total_leaves,
        max_hits=max_hits,
        trace=trace,
    ):
        selected.append((doc_id, start, end, witness, hit))
        if stop_early and len(selected) >= target:
            break
    page_items = selected[offset : offset + limit]

    text_cache: Dict[str, str] = {}
    windows_out: List[Window] = []
    spans_by_doc: Dict[str, Dict[int, Tuple[int, int, int, int]]] = {}
    for doc_id, start, end, witness, hit in page_items:
        if doc_id not in spans_by_doc:
            rows = index.postings_for_doc(distinct_tokens, doc_id)
            spans_by_doc[doc_id] = _span_by_position(leaves, rows)
        windows_out.append(
            _window_from_hit(
                index,
                doc_id,
                hit,
                start,
                end,
                witness,
                leaves,
                spans_by_doc[doc_id],
                text_cache,
            )
        )
    trace.body_reads = list(dict.fromkeys(trace.body_reads))
    has_more = len(selected) > offset + limit
    next_cursor = None
    if has_more:
        next_cursor = encode_cursor(
            offset + len(windows_out), limit, signature, index.fingerprint
        )
    return SearchPage(
        windows=windows_out,
        offset=offset,
        limit=limit,
        next_cursor=next_cursor,
        has_more=has_more,
        query_signature=signature,
        index_fingerprint=index.fingerprint,
        trace=trace.as_dict(),
        total_candidate_docs=len(docs),
    )
