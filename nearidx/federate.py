"""Federated search over several independent near-neighbour indexes.

A *federation* is a manifest mapping stable source names to index files.  One
``NEAR``/``SEQ``/``OR`` query runs against every source at once; the merged
result keeps the source name next to the original window, witness and text
span, and is ordered globally by ``(doc_id, source_name, start, end)``.  The
same ``doc_id`` appearing in two sources is two different results -- sources
are never merged or deduplicated into each other.

Read-only
---------
The federation only reads its input indexes: every source is opened through
:func:`nearidx.db.open_index` (read-only SQLite) and no call here writes to
any index file.

Generation pinning and cursors
------------------------------
The first page pins the generation each source handle actually opened.  The
returned cursor names, per source, the index identity (``index_id``) and the
pinned generation fingerprint, plus the per-source resume position.  A
follow-up call re-opens exactly those generations -- even in a later process,
after one or more sources were refreshed -- and refuses to continue when:

* the query or the page size differs from the cursor's,
* the source set changed (a source added or dropped),
* a source name now points at a different index (``index_id`` mismatch),
* any pinned generation fell out of retention
  (:class:`GenerationExpiredError`).

A missing, corrupt or expired source fails the *whole* call; no source is
ever silently skipped to produce a partial page that looks complete.

Streaming merge
---------------
Paging is a k-way merge over per-source window streams that are already in
the global order.  Only enough windows to fill the current page -- plus the
look-ahead proving ``has_more`` -- are ever pulled, and only the delivered
windows' original texts are read.  The federation never collects every
source's hits into memory before sorting them.
"""

from __future__ import annotations

import base64
import heapq
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

from .db import AccessTrace, Index, open_index
from .query import CursorError, Node, iter_leaves, query_signature
from .solver import (
    Hit,
    QueryError,
    Window,
    _candidate_docs,
    _stream_root_windows,
    _window_from_hit,
)

__all__ = [
    "Federation",
    "FederationError",
    "FederatedPage",
    "FederatedWindow",
    "federated_search",
    "load_federation_manifest",
]


class FederationError(ValueError):
    """The federation manifest or one of its sources is invalid.

    Raised before any window is produced: a federation with a missing,
    corrupt or misconfigured source never answers a query partially.
    """


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def _validate_source_name(name: object, *, where: str) -> str:
    if not isinstance(name, str) or not name:
        raise FederationError(
            f"{where}: source names must be non-empty strings"
        )
    if "\x00" in name:
        raise FederationError(f"{where}: source name {name!r} contains NUL")
    return name


def _validate_source_path(value: object, *, where: str) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise FederationError(
            f"{where}: the index path must be a non-empty string"
        )
    if "\x00" in value:
        raise FederationError(f"{where}: index path contains NUL")
    return value


def _normalize_sources(sources: object) -> Dict[str, str]:
    """Validate a source mapping, returning it as a dict.

    ``sources`` may be a mapping of name -> path or an iterable of
    ``(name, path)`` pairs; duplicate names are rejected either way.
    """
    if isinstance(sources, Mapping):
        pairs = list(sources.items())
    elif isinstance(sources, (str, bytes)):
        raise FederationError(
            "sources must be a mapping of source name -> index path"
        )
    else:
        try:
            pairs = list(sources)  # type: ignore[arg-type]
        except TypeError:
            raise FederationError(
                "sources must be a mapping of source name -> index path"
            ) from None
    out: Dict[str, str] = {}
    for i, pair in enumerate(pairs):
        where = f"source entry {i + 1}"
        if isinstance(pair, (str, bytes)):
            raise FederationError(f"{where}: expected a (name, path) pair")
        try:
            name, value = pair
        except (TypeError, ValueError):
            raise FederationError(
                f"{where}: expected a (name, path) pair"
            ) from None
        name = _validate_source_name(name, where=where)
        path = _validate_source_path(value, where=where)
        if name in out:
            raise FederationError(f"duplicate source name {name!r}")
        out[name] = path
    if not out:
        raise FederationError("the federation names no sources")
    return out


def load_federation_manifest(path: os.PathLike | str) -> Dict[str, str]:
    """Read a federation manifest: a JSON object mapping name -> index path.

    Relative index paths resolve against the manifest file's directory.
    Duplicate source names are rejected even though a JSON parser would
    normally keep only the last duplicate key.
    """
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise FederationError(
            f"cannot read federation manifest {str(p)!r}: {exc}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FederationError(
            f"federation manifest {str(p)!r} is not valid UTF-8: {exc}"
        ) from exc

    def no_duplicates(pairs: List[Tuple[str, object]]) -> Dict[str, object]:
        obj: Dict[str, object] = {}
        for key, value in pairs:
            if key in obj:
                raise FederationError(
                    f"duplicate source name {key!r} in manifest {str(p)!r}"
                )
            obj[key] = value
        return obj

    try:
        data = json.loads(text, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as exc:
        raise FederationError(
            f"federation manifest {str(p)!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FederationError(
            f"federation manifest {str(p)!r} must be a JSON object mapping "
            "source names to index files"
        )
    mapping: Dict[str, str] = {}
    for raw_name, raw_value in data.items():
        where = f"source {raw_name!r}"
        name = _validate_source_name(raw_name, where=where)
        value = _validate_source_path(raw_value, where=where)
        index_path = Path(value)
        if not index_path.is_absolute():
            index_path = p.parent / index_path
        mapping[name] = str(index_path)
    if not mapping:
        raise FederationError(f"federation manifest {str(p)!r} names no sources")
    return mapping


# ---------------------------------------------------------------------------
# the federation handle
# ---------------------------------------------------------------------------

class Federation:
    """A read-only set of named near-neighbour indexes searched as one.

    Every source is opened eagerly: a missing file or an invalid index fails
    the construction -- and therefore every query through it -- instead of
    being silently skipped.  Each handle serves the generation that was
    current when the federation was opened; the first page of a federated
    search pins exactly those generations.
    """

    def __init__(
        self,
        sources: Mapping[str, os.PathLike | str] | List[Tuple[str, os.PathLike | str]],
        *,
        verify: bool = True,
    ):
        mapping = _normalize_sources(sources)
        self._paths: Dict[str, str] = dict(mapping)
        self._handles: Dict[str, Index] = {}
        try:
            for name, index_path in mapping.items():
                self._handles[name] = open_index(index_path, verify=verify)
        except BaseException:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:  # noqa: BLE001 - nothing to salvage here
                    pass
            self._handles.clear()
            raise

    @classmethod
    def from_manifest(
        cls, path: os.PathLike | str, *, verify: bool = True
    ) -> "Federation":
        """Open the federation described by a JSON manifest file."""
        return cls(load_federation_manifest(path), verify=verify)

    @property
    def source_names(self) -> List[str]:
        """The source names in canonical (sorted) order."""
        return sorted(self._handles)

    def source_path(self, name: str) -> str:
        """The index path a source name maps to."""
        return self._paths[name]

    def handle(self, name: str) -> Index:
        """The open read-only handle of one source."""
        return self._handles[name]

    def sources_info(self) -> Dict[str, Dict[str, object]]:
        """Per-source identity of the generations the handles currently serve."""
        return {
            name: {
                "path": self._paths[name],
                "index_id": handle.index_id,
                "generation": handle.generation,
                "fingerprint": handle.fingerprint,
            }
            for name, handle in sorted(self._handles.items())
        }

    def search(self, root: Node, **kwargs: object) -> "FederatedPage":
        """Convenience wrapper around :func:`federated_search`."""
        return federated_search(self, root, **kwargs)

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - closing must not fail the rest
                pass
        self._handles.clear()

    def __enter__(self) -> "Federation":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# federated pagination cursor
# ---------------------------------------------------------------------------

_CURSOR_KIND = "nearidx-federated-v1"


@dataclass(frozen=True, slots=True)
class _FCursorEntry:
    """One source's pinned identity, generation and resume position."""

    name: str
    index_id: str
    fingerprint: str
    doc: Optional[str]   # last delivered candidate document, None before any
    skip: int            # windows of ``doc`` already delivered


@dataclass(frozen=True, slots=True)
class _FCursor:
    query_sig: str
    page_size: int
    emitted: int
    entries: Tuple[_FCursorEntry, ...]  # sorted by source name


def _encode_fcursor(
    *,
    query_sig: str,
    page_size: int,
    emitted: int,
    entries: List[_FCursorEntry],
) -> str:
    raw = json.dumps(
        {
            "v": _CURSOR_KIND,
            "q": query_sig,
            "n": page_size,
            "e": emitted,
            "s": [
                {
                    "name": entry.name,
                    "id": entry.index_id,
                    "fp": entry.fingerprint,
                    "doc": entry.doc,
                    "skip": entry.skip,
                }
                for entry in entries
            ],
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_fcursor(token: str) -> _FCursor:
    try:
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(token + pad)
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict) or obj.get("v") != _CURSOR_KIND:
            raise ValueError
        query_sig = obj["q"]
        page_size = obj["n"]
        emitted = obj["e"]
        sources = obj["s"]
        if not isinstance(query_sig, str) or not query_sig:
            raise ValueError
        if not isinstance(page_size, int) or isinstance(page_size, bool):
            raise ValueError
        if page_size < 1:
            raise ValueError
        if not isinstance(emitted, int) or isinstance(emitted, bool) or emitted < 0:
            raise ValueError
        if not isinstance(sources, list) or not sources:
            raise ValueError
        entries: List[_FCursorEntry] = []
        seen = set()
        for item in sources:
            if not isinstance(item, dict) or set(item) != {
                "name",
                "id",
                "fp",
                "doc",
                "skip",
            }:
                raise ValueError
            name = item["name"]
            index_id = item["id"]
            fingerprint = item["fp"]
            doc = item["doc"]
            skip = item["skip"]
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError
            seen.add(name)
            if not isinstance(index_id, str) or not index_id:
                raise ValueError
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError
            if doc is not None and not isinstance(doc, str):
                raise ValueError
            if not isinstance(skip, int) or isinstance(skip, bool) or skip < 0:
                raise ValueError
            # The encoder only ever emits (None, 0) or (doc, >=1).
            if (doc is None) != (skip == 0):
                raise ValueError
            entries.append(_FCursorEntry(name, index_id, fingerprint, doc, skip))
        entries.sort(key=lambda entry: entry.name)
    except Exception as exc:  # noqa: BLE001 - surface as a query-level error
        raise CursorError("invalid federated pagination cursor") from exc
    return _FCursor(
        query_sig=query_sig,
        page_size=page_size,
        emitted=emitted,
        entries=tuple(entries),
    )


# ---------------------------------------------------------------------------
# streaming k-way merge
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _SourceStream:
    """One source's lazily-pulled window stream and its resume position."""

    name: str
    index: Index
    stream: Iterator
    last_doc: Optional[str]
    skip: int


def _pull_next(
    state: _SourceStream, heap: List[tuple], counter: "itertools.count"
) -> None:
    """Pull one window from the source and push it onto the merge heap."""
    item = next(state.stream, None)
    if item is None:
        return
    doc_id, start, end, witness, hit, spans = item
    # The (doc_id, source_name, start, end) key is unique across the whole
    # merge; the counter only keeps the payload from ever being compared.
    heapq.heappush(
        heap,
        (doc_id, state.name, start, end, next(counter), (state, witness, hit, spans)),
    )


@dataclass(slots=True)
class FederatedWindow:
    """One delivered window, tagged with the source it came from."""

    source_name: str
    window: Window

    @property
    def doc_id(self) -> str:
        return self.window.doc_id

    @property
    def start(self) -> int:
        return self.window.start

    @property
    def end(self) -> int:
        return self.window.end

    @property
    def witness(self) -> Tuple[int, ...]:
        return self.window.witness

    @property
    def snippet(self) -> str:
        return self.window.snippet

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {"source_name": self.source_name}
        out.update(self.window.as_dict())
        return out


@dataclass(slots=True)
class FederatedPage:
    """One merged page of a federated search."""

    windows: List[FederatedWindow]
    offset: int
    limit: int
    next_cursor: Optional[str]
    has_more: bool
    query_signature: str
    #: name -> {path, index_id, generation, fingerprint} actually read.
    sources: Dict[str, Dict[str, object]]
    #: name -> AccessTrace evidence of that source for this call.
    trace: Dict[str, Dict[str, object]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "windows": [window.as_dict() for window in self.windows],
            "offset": self.offset,
            "limit": self.limit,
            "has_more": self.has_more,
            "next_cursor": self.next_cursor,
            "query_signature": self.query_signature,
            "sources": self.sources,
            "trace": self.trace,
        }


def federated_search(
    federation: Federation,
    root: Node,
    *,
    limit: int = 20,
    cursor: Optional[str] = None,
    max_hits: int = 2_000_000,
) -> FederatedPage:
    """Run *root* against every source of *federation* and merge one page.

    Windows are delivered in the global order ``(doc_id, source_name, start,
    end)``; the same ``doc_id`` in two sources stays two distinct results.
    Every returned window is produced by the same solver, witness and snippet
    machinery as single-index :func:`nearidx.solver.search`.

    Without ``cursor`` the call pins the generation each source handle
    currently serves and returns the first page.  With ``cursor`` the call
    re-opens the exact generations the cursor pinned (raising
    :class:`nearidx.query.GenerationExpiredError` if any fell out of
    retention) and continues the merge from the per-source resume positions.
    The cursor's own page size wins over ``limit``.  A different query, a
    different source set, or a source name that now points at a different
    index raises :class:`CursorError`.  Any missing, corrupt or expired
    source fails the whole call -- never a silently partial page.
    """
    if limit < 1:
        raise QueryError("limit must be at least 1")
    names = federation.source_names
    if not names:
        raise FederationError("the federation has no open sources (closed?)")
    signature = query_signature(root)

    if cursor is not None:
        cur = _decode_fcursor(cursor)
        if cur.query_sig != signature:
            raise CursorError("cursor was issued for a different query")
        limit = cur.page_size
        pinned = {entry.name: entry for entry in cur.entries}
        if sorted(pinned) != names:
            raise CursorError(
                "cursor was issued for a different set of sources "
                f"(cursor: {sorted(pinned)!r}, federation: {names!r})"
            )
        offset = cur.emitted
    else:
        pinned = None
        offset = 0

    # Pin every source's generation before streaming anything: a source whose
    # pinned generation expired -- or whose name now points at another index
    # -- fails the whole call instead of being silently skipped.
    resolved: Dict[str, Tuple[Index, _FCursorEntry]] = {}
    for name in names:
        handle = federation.handle(name)
        if pinned is None:
            entry = _FCursorEntry(
                name, handle.index_id, handle.fingerprint, None, 0
            )
        else:
            entry = pinned[name]
            if entry.index_id != handle.index_id:
                raise CursorError(
                    f"source {name!r} now points at a different index than "
                    "the one the cursor was issued against"
                )
        # Raises GenerationExpiredError when the pinned generation is gone.
        gen_index = handle.for_generation(entry.fingerprint)
        resolved[name] = (gen_index, entry)

    leaves = list(iter_leaves(root))
    total_leaves = len(leaves)
    distinct_tokens = list(dict.fromkeys(leaf.token for leaf in leaves))

    heap: List[tuple] = []
    counter = itertools.count()
    streams: Dict[str, _SourceStream] = {}
    traces: Dict[str, AccessTrace] = {}
    for name in names:
        gen_index, entry = resolved[name]
        trace = AccessTrace()
        gen_index.trace = trace
        docs = _candidate_docs(gen_index, root, trace)
        stream = _stream_root_windows(
            gen_index,
            root,
            docs,
            leaves,
            distinct_tokens,
            total_leaves,
            max_hits=max_hits,
            trace=trace,
            resume=(entry.doc, entry.skip),
        )
        state = _SourceStream(
            name=name,
            index=gen_index,
            stream=stream,
            last_doc=entry.doc,
            skip=entry.skip,
        )
        streams[name] = state
        traces[name] = trace
        _pull_next(state, heap, counter)

    # K-way merge: pop the globally smallest window, deliver it, and pull the
    # replacement from the same source.  Only ``limit`` windows plus the
    # per-source look-ahead are ever pulled -- never every hit of every
    # source.  The heap is exactly the "one more window" evidence: a
    # non-empty heap after the page is full proves ``has_more``.
    picked: List[
        Tuple[
            _SourceStream,
            str,
            int,
            int,
            Tuple[int, ...],
            Hit,
            Dict[int, Tuple[int, int, int, int]],
        ]
    ] = []
    while len(picked) < limit and heap:
        _doc_id, _name, start, end, _c, payload = heapq.heappop(heap)
        state, witness, hit, spans = payload
        picked.append((state, _doc_id, start, end, witness, hit, spans))
        if _doc_id == state.last_doc:
            state.skip += 1
        else:
            state.last_doc, state.skip = _doc_id, 1
        _pull_next(state, heap, counter)
    has_more = bool(heap)

    # Only the delivered windows' documents have their original text read.
    text_caches: Dict[str, Dict[str, str]] = {name: {} for name in names}
    windows_out: List[FederatedWindow] = []
    for state, doc_id, start, end, witness, hit, spans in picked:
        window = _window_from_hit(
            state.index,
            doc_id,
            hit,
            start,
            end,
            witness,
            leaves,
            spans,
            text_caches[state.name],
        )
        windows_out.append(FederatedWindow(source_name=state.name, window=window))
    for trace in traces.values():
        trace.body_reads = list(dict.fromkeys(trace.body_reads))

    next_cursor = None
    if has_more:
        next_cursor = _encode_fcursor(
            query_sig=signature,
            page_size=limit,
            emitted=offset + len(windows_out),
            entries=[
                _FCursorEntry(
                    name,
                    resolved[name][1].index_id,
                    resolved[name][1].fingerprint,
                    streams[name].last_doc,
                    streams[name].skip,
                )
                for name in names
            ],
        )

    sources_meta = {
        name: {
            "path": federation.source_path(name),
            "index_id": gen_index.index_id,
            "generation": gen_index.generation,
            "fingerprint": gen_index.fingerprint,
        }
        for name, (gen_index, _entry) in resolved.items()
    }
    return FederatedPage(
        windows=windows_out,
        offset=offset,
        limit=limit,
        next_cursor=next_cursor,
        has_more=has_more,
        query_signature=signature,
        sources=sources_meta,
        trace={name: traces[name].as_dict() for name in names},
    )
