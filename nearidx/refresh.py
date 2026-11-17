"""Batch refresh: validate a manifest, publish it as one new generation.

A *manifest* is a JSON Lines file, one operation per line::

    {"op": "upsert", "doc_id": "d1", "text": "new body"}
    {"op": "upsert", "doc_id": "d2", "path": "docs/d2.txt"}
    {"op": "delete", "doc_id": "d3"}

The whole batch is validated and every ``path`` source is read and decoded
*before* anything is built, so a bad line, an unreadable source, a duplicate
``doc_id`` or a delete of a missing document rejects the entire batch and
leaves the current index -- and every in-flight pagination cursor -- exactly
as it was.

A successful batch is published as a new *generation*: the new content is
built as a complete index file, the publish is journaled in the generations
registry, and two atomic renames swap it in.  A crash anywhere is recovered
by :func:`nearidx.db.recover_generations` on the next open, leaving either
the complete old generation or the complete new one -- never a mixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .db import (
    DEFAULT_MAX_DOC_CHARS,
    DEFAULT_RETAIN_OLD,
    Document,
    IndexErrorBase,
    IndexFormatError,
    _cleanup_orphans,
    _drop_retained_files,
    _fsync_dir,
    _fsync_file,
    _gen_file_name,
    _gen_file_path,
    _meta,
    _read_registry,
    _trim_retained,
    _write_index_file,
    _write_registry,
    recover_generations,
)

__all__ = [
    "ManifestError",
    "ManifestOp",
    "RefreshReport",
    "parse_manifest",
    "refresh_index",
]


class ManifestError(IndexErrorBase, ValueError):
    """The refresh manifest is invalid; the whole batch was rejected."""


@dataclass(frozen=True, slots=True)
class ManifestOp:
    """One validated manifest operation."""

    op: str  # "upsert" | "delete"
    doc_id: str
    text: Optional[str] = None
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None


@dataclass(slots=True)
class RefreshReport:
    path: str
    generation: int
    index_fingerprint: str
    upserted: int
    deleted: int
    doc_count: int
    token_count: int
    retained_generations: List[int]
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "generation": self.generation,
            "index_fingerprint": self.index_fingerprint,
            "upserted": self.upserted,
            "deleted": self.deleted,
            "doc_count": self.doc_count,
            "token_count": self.token_count,
            "retained_generations": list(self.retained_generations),
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# manifest parsing and validation
# ---------------------------------------------------------------------------

_KNOWN_KEYS = {"op", "doc_id", "text", "path"}


def _validate_op(
    obj: object,
    *,
    where: str,
    current_ids: Optional[set],
    max_doc_chars: int,
) -> ManifestOp:
    if not isinstance(obj, Mapping):
        raise ManifestError(f"{where}: each operation must be a JSON object")
    unknown = set(obj) - _KNOWN_KEYS
    if unknown:
        raise ManifestError(
            f"{where}: unknown keys {sorted(unknown)!r}; allowed keys are "
            f"{sorted(_KNOWN_KEYS)!r}"
        )
    op = obj.get("op")
    doc_id = obj.get("doc_id")
    if not isinstance(doc_id, str) or doc_id == "":
        raise ManifestError(f"{where}: doc_id must be a non-empty string")
    if op == "upsert":
        has_text = "text" in obj
        has_path = "path" in obj
        if has_text == has_path:
            raise ManifestError(
                f"{where}: upsert must provide exactly one of 'text' or 'path'"
            )
        if has_text:
            text = obj["text"]
            if not isinstance(text, str):
                raise ManifestError(f"{where}: 'text' must be a string")
            _check_text(text, doc_id=doc_id, where=where,
                        max_doc_chars=max_doc_chars)
            return ManifestOp(op="upsert", doc_id=doc_id, text=text)
        path = obj["path"]
        if not isinstance(path, str) or path == "":
            raise ManifestError(f"{where}: 'path' must be a non-empty string")
        return _op_from_path(doc_id, path, where=where,
                             max_doc_chars=max_doc_chars)
    if op == "delete":
        if "text" in obj or "path" in obj:
            raise ManifestError(
                f"{where}: delete takes only 'op' and 'doc_id', not 'text'/'path'"
            )
        if current_ids is not None and doc_id not in current_ids:
            raise ManifestError(
                f"{where}: cannot delete {doc_id!r}: no such document in the "
                "current index"
            )
        return ManifestOp(op="delete", doc_id=doc_id)
    raise ManifestError(f"{where}: unknown op {op!r} (expected 'upsert' or 'delete')")


def _check_text(text: str, *, doc_id: str, where: str, max_doc_chars: int) -> None:
    if "\x00" in text:
        raise ManifestError(f"{where}: document {doc_id!r} contains NUL")
    if len(text) > max_doc_chars:
        raise ManifestError(
            f"{where}: document {doc_id!r} has {len(text)} chars, limit is "
            f"{max_doc_chars}"
        )


def _op_from_path(
    doc_id: str, path: str, *, where: str, max_doc_chars: int
) -> ManifestOp:
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{where}: cannot read {path!r}: {exc}") from exc
    try:
        text = raw.decode("utf-8")  # strict: no silent replacement
    except UnicodeDecodeError as exc:
        raise ManifestError(
            f"{where}: {path!r} is not valid UTF-8: {exc}"
        ) from exc
    _check_text(text, doc_id=doc_id, where=where, max_doc_chars=max_doc_chars)
    return ManifestOp(
        op="upsert",
        doc_id=doc_id,
        text=text,
        source_path=str(p.resolve()),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_manifest(
    source: Union[str, os.PathLike, Iterable[Mapping[str, object]]],
    *,
    current_ids: Optional[Iterable[str]] = None,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> List[ManifestOp]:
    """Parse and fully validate a refresh manifest.

    ``source`` is the path of a JSON Lines manifest or an iterable of
    operation mappings.  Every line/entry is checked, every ``path`` source
    is read and decoded, and the batch is rejected as a whole on the first
    problem.  ``current_ids`` (the doc_ids of the current index) enables the
    delete-must-exist check.
    """
    ids = set(current_ids) if current_ids is not None else None
    entries: List[Tuple[str, object]] = []
    if isinstance(source, (str, os.PathLike)):
        manifest_path = Path(source)
        try:
            raw = manifest_path.read_bytes()
        except OSError as exc:
            raise ManifestError(
                f"cannot read manifest {str(manifest_path)!r}: {exc}"
            ) from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestError(
                f"manifest {str(manifest_path)!r} is not valid UTF-8: {exc}"
            ) from exc
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{manifest_path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{where}: invalid JSON: {exc}") from exc
            entries.append((where, obj))
    else:
        for i, obj in enumerate(source):
            entries.append((f"manifest entry {i + 1}", obj))
    ops: List[ManifestOp] = []
    seen: set = set()
    for where, obj in entries:
        op = _validate_op(obj, where=where, current_ids=ids,
                          max_doc_chars=max_doc_chars)
        if op.doc_id in seen:
            raise ManifestError(
                f"{where}: doc_id {op.doc_id!r} appears more than once in "
                "this batch"
            )
        seen.add(op.doc_id)
        ops.append(op)
    return ops


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------

def _validate_retain_old(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "retain_old must be an integer >= 1: the current generation plus "
            "at least one old generation are always kept"
        )
    return value


def refresh_index(
    path: os.PathLike | str,
    manifest: Union[str, os.PathLike, Iterable[Mapping[str, object]]],
    *,
    retain_old: Optional[int] = None,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> RefreshReport:
    """Validate *manifest* and publish it as one new index generation.

    The batch is atomic: on any validation or input error nothing changes;
    a crash during the publish is recovered on the next open to either the
    complete old or the complete new generation.  ``retain_old`` controls how
    many old generations stay queryable for in-flight cursors (default and
    minimum come from the registry / :data:`DEFAULT_RETAIN_OLD`, and it can
    never be less than 1).
    """
    started = time.monotonic()
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    # Resolve any interrupted earlier publish before starting a new one.
    recover_generations(target)

    conn = sqlite3.connect(
        f"file:{target.resolve().as_posix()}?mode=ro", uri=True,
        isolation_level=None,
    )
    try:
        meta = _meta(conn)
        rows = conn.execute(
            "SELECT doc_id, text, source_path, source_sha256 FROM documents "
            "ORDER BY doc_id"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise IndexFormatError(
            f"{target} is not a readable nearidx index: {exc}"
        ) from exc
    finally:
        conn.close()
    try:
        old_generation = int(meta["generation"])
    except (KeyError, ValueError):
        old_generation = 1
    old_fingerprint = meta.get("index_fingerprint", "")
    index_id = meta.get("index_id", "")
    name = meta.get("name", target.stem)
    created_utc = meta.get(
        "created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    current_ids = {r[0] for r in rows}
    # Everything that can fail is done here, before any file is touched.
    ops = parse_manifest(
        manifest, current_ids=current_ids, max_doc_chars=max_doc_chars
    )

    reg = _read_registry(target) or {}
    if retain_old is None:
        keep = reg.get("retain_old", DEFAULT_RETAIN_OLD)
        retain_old = keep if isinstance(keep, int) and not isinstance(keep, bool) \
            else DEFAULT_RETAIN_OLD
    retain_old = _validate_retain_old(retain_old)

    # Apply the batch to the current document set, in memory.
    upserts = {op.doc_id: op for op in ops if op.op == "upsert"}
    deletes = {op.doc_id for op in ops if op.op == "delete"}
    final: Dict[str, Document] = {}
    for doc_id, text, spath, ssha in rows:
        if doc_id in deletes:
            continue
        if doc_id in upserts:
            op = upserts[doc_id]
            final[doc_id] = Document(
                doc_id=doc_id,
                text=op.text,
                source_path=op.source_path,
                source_sha256=op.source_sha256,
            )
        else:
            final[doc_id] = Document(
                doc_id=doc_id, text=text, source_path=spath, source_sha256=ssha
            )
    for doc_id, op in upserts.items():
        if doc_id not in current_ids:
            final[doc_id] = Document(
                doc_id=doc_id,
                text=op.text,
                source_path=op.source_path,
                source_sha256=op.source_sha256,
            )
    docs_final = [final[k] for k in sorted(final)]

    new_generation = old_generation + 1
    tmp = target.with_name(
        f".{target.name}.refresh-{os.getpid()}-{time.time_ns()}.tmp"
    )
    if tmp.exists():
        tmp.unlink()
    doc_count, token_count, fingerprint = _write_index_file(
        tmp,
        docs_final,
        name=name,
        index_id=index_id,
        generation=new_generation,
        created_utc=created_utc,
        max_doc_chars=max_doc_chars,
    )
    _fsync_file(tmp)

    # Stage the complete new generation as an unpublished sidecar file.
    staged = _gen_file_path(target, new_generation)
    os.replace(tmp, staged)
    _fsync_dir(target.parent)

    carried: List[Dict[str, object]] = []
    for entry in reg.get("retained", []) or []:
        if not isinstance(entry, dict):
            continue
        g = entry.get("generation")
        if isinstance(g, int) and g not in (old_generation,) and g < new_generation:
            carried.append(entry)

    # Journal the pending publish before moving anything.
    reg_out: Dict[str, object] = {
        "version": 1,
        "index_id": index_id,
        "retain_old": retain_old,
        "current": {"generation": old_generation, "fingerprint": old_fingerprint},
        "retained": carried,
        "pending": {
            "generation": new_generation,
            "fingerprint": fingerprint,
            "from_generation": old_generation,
            "from_fingerprint": old_fingerprint,
        },
    }
    _write_registry(target, reg_out)

    # Publish: move the old main aside, then move the new generation in.
    # Both renames are atomic; a crash between them is rolled back by
    # recover_generations (old main restored from its generation file).
    os.replace(target, _gen_file_path(target, old_generation))
    _fsync_dir(target.parent)
    os.replace(staged, target)
    _fsync_dir(target.parent)

    # Finalise: the new generation is current, the old one joins retention.
    retained: List[Dict[str, object]] = [
        {
            "generation": old_generation,
            "fingerprint": old_fingerprint,
            "file": _gen_file_name(target, old_generation),
        }
    ]
    retained.extend(carried)
    kept, dropped = _trim_retained(retained, retain_old)
    reg_out["current"] = {"generation": new_generation, "fingerprint": fingerprint}
    reg_out["retained"] = kept
    reg_out["pending"] = None
    _write_registry(target, reg_out)
    _drop_retained_files(target, dropped)
    _cleanup_orphans(target, reg_out)

    return RefreshReport(
        path=str(target),
        generation=new_generation,
        index_fingerprint=fingerprint,
        upserted=len(upserts),
        deleted=len(deletes),
        doc_count=doc_count,
        token_count=token_count,
        retained_generations=[int(e["generation"]) for e in kept],
        elapsed_seconds=time.monotonic() - started,
    )
