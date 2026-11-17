"""SQLite persistence: schema, builder, reopen, integrity and explicit rebuild.

Layout
------
``meta``       schema version, index id/name, fingerprint, build stats.
``documents``  one row per document: original Unicode text, its UTF-8 SHA-256,
               token count, and a digest over that document's posting rows.
``postings``   one row per token occurrence: folded token key, doc_id, 0-based
               position, original character span and UTF-8 byte span.

The index file is self-contained: snippets are cut from the stored original
text, never rebuilt from tokens.  A document's text and its postings are bound
through ``text_sha256`` and ``postings_digest``; replacing the content without
calling :meth:`Index.rebuild_document` is detected by :meth:`Index.verify`
(and by the source-file check when a document was loaded from a file).

Generations
-----------
A batch refresh (:func:`nearidx.refresh.refresh_index`) never edits the main
file in place: it builds the new generation as a complete sibling file and
swaps it in with atomic renames, journaled in ``INDEX.generations.json``.
The previous main file is kept as ``INDEX.gen-<N>`` so pagination cursors
issued against it keep reading their original generation; the registry
controls how many old generations are retained (current + two by default).
``recover_generations`` runs on every open and resolves an interrupted
publish to either the complete old or the complete new generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .lexer import token_spans
from .query import GenerationExpiredError

SCHEMA_VERSION = 1
FINGERPRINT_VERSION = "nearidx-fingerprint-v1"
DEFAULT_MAX_DOC_CHARS = 32_000_000
#: How many *old* generations a refresh keeps next to the current one.
#: The current generation is always served; at least one old generation must
#: be retained so in-flight pagination survives a publish.
DEFAULT_RETAIN_OLD = 2

__all__ = [
    "Document",
    "GenerationInfo",
    "Index",
    "IndexErrorBase",
    "IndexExistsError",
    "IndexFormatError",
    "StaleSourceError",
    "BuildReport",
    "build_index",
    "open_index",
    "recover_generations",
    "text_sha256",
]


class IndexErrorBase(Exception):
    """Base class for index-level errors."""


class IndexExistsError(IndexErrorBase):
    pass


class IndexFormatError(IndexErrorBase):
    pass


class StaleSourceError(IndexErrorBase):
    """The original text no longer matches the digest recorded in the index."""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _postings_digest(rows: Iterable[Tuple[int, str, int, int, int, int]]) -> str:
    h = hashlib.sha256()
    for pos, norm, cs, ce, bs, be in rows:
        h.update(f"{pos}\x1f{norm}\x1f{cs}\x1f{ce}\x1f{bs}\x1f{be}\n".encode("utf-8"))
    return h.hexdigest()


def _fingerprint(docs: Sequence[Tuple[str, str, str]]) -> str:
    """Hash of (doc_id, text sha, postings digest) in code-point order."""
    h = hashlib.sha256(FINGERPRINT_VERSION.encode("ascii") + b"\n")
    for doc_id, sha, pd in sorted(docs, key=lambda d: d[0]):
        h.update(f"{doc_id}\x1f{sha}\x1f{pd}\n".encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None

    @staticmethod
    def from_file(doc_id: str, path: os.PathLike | str) -> "Document":
        p = Path(path)
        raw = p.read_bytes()
        text = raw.decode("utf-8")  # strict: no silent replacement
        return Document(
            doc_id=doc_id,
            text=text,
            source_path=str(p.resolve()),
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(slots=True)
class BuildReport:
    path: str
    doc_count: int
    token_count: int
    index_fingerprint: str
    elapsed_seconds: float


_SCHEMA_SQL = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    text_sha256     TEXT NOT NULL,
    char_len        INTEGER NOT NULL,
    byte_len        INTEGER NOT NULL,
    token_count     INTEGER NOT NULL,
    postings_digest TEXT NOT NULL,
    source_path     TEXT,
    source_sha256   TEXT
);
CREATE TABLE postings (
    token      TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    pos        INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    byte_start INTEGER NOT NULL,
    byte_end   INTEGER NOT NULL,
    PRIMARY KEY (token, doc_id, pos)
) WITHOUT ROWID;
CREATE INDEX postings_by_doc ON postings (doc_id, pos);
"""


def _connect_build(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA page_size = 4096")
    return conn


def _meta(conn: sqlite3.Connection) -> Dict[str, str]:
    return {k: v for k, v in conn.execute("SELECT key, value FROM meta")}


def _fsync_file(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_index_file(
    tmp: Path,
    docs: Iterable[Document | Tuple[str, str]],
    *,
    name: str,
    index_id: str,
    generation: int,
    created_utc: str,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> Tuple[int, int, str]:
    """Write a complete, self-contained index file for one generation.

    Shared by :func:`build_index` (generation 1) and the batch refresh
    (generation N+1).  Returns ``(doc_count, token_count, fingerprint)``.
    On any error the temporary file is removed and nothing is left behind.
    """
    conn = _connect_build(tmp)
    doc_count = 0
    total_tokens = 0
    fingerprint_rows: List[Tuple[str, str, str]] = []
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.execute("BEGIN")
        seen = set()
        for item in docs:
            if isinstance(item, Document):
                doc = item
            elif isinstance(item, tuple) and len(item) == 2:
                doc_id, text = item
                doc = Document(doc_id=doc_id, text=text)
            else:
                raise TypeError(
                    "each document must be Document or a (doc_id, text) tuple"
                )
            if not isinstance(doc.doc_id, str) or doc.doc_id == "":
                raise ValueError("doc_id must be a non-empty string")
            if not isinstance(doc.text, str):
                raise ValueError(f"document {doc.doc_id!r} text must be str")
            if "\x00" in doc.text:
                raise ValueError(f"document {doc.doc_id!r} contains NUL")
            if len(doc.text) > max_doc_chars:
                raise ValueError(
                    f"document {doc.doc_id!r} has {len(doc.text)} chars, "
                    f"limit is {max_doc_chars}"
                )
            if doc.doc_id in seen:
                raise ValueError(f"duplicate doc_id {doc.doc_id!r}")
            seen.add(doc.doc_id)
            sha = text_sha256(doc.text)
            toks = token_spans(doc.text)
            rows = [
                (t.pos, t.norm, t.char_start, t.char_end, t.byte_start, t.byte_end)
                for t in toks
            ]
            digest = _postings_digest(rows)
            conn.execute(
                "INSERT INTO documents(doc_id, text, text_sha256, char_len, "
                "byte_len, token_count, postings_digest, source_path, "
                "source_sha256) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc.doc_id,
                    doc.text,
                    sha,
                    len(doc.text),
                    len(doc.text.encode("utf-8")),
                    len(toks),
                    digest,
                    doc.source_path,
                    doc.source_sha256,
                ),
            )
            conn.executemany(
                "INSERT INTO postings(token, doc_id, pos, char_start, char_end, "
                "byte_start, byte_end) VALUES (?,?,?,?,?,?,?)",
                [(t.norm, doc.doc_id, t.pos, t.char_start, t.char_end,
                  t.byte_start, t.byte_end) for t in toks],
            )
            fingerprint_rows.append((doc.doc_id, sha, digest))
            doc_count += 1
            total_tokens += len(toks)
        fingerprint = _fingerprint(fingerprint_rows)
        meta = {
            "schema_version": str(SCHEMA_VERSION),
            "fingerprint_version": FINGERPRINT_VERSION,
            "index_id": index_id,
            "name": name,
            "created_utc": created_utc,
            "doc_count": str(doc_count),
            "token_count": str(total_tokens),
            "index_fingerprint": fingerprint,
            "generation": str(generation),
        }
        conn.executemany("INSERT INTO meta(key, value) VALUES (?,?)", meta.items())
        conn.execute("COMMIT")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("ANALYZE")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        conn.close()
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    conn.close()
    return doc_count, total_tokens, fingerprint


def build_index(
    path: os.PathLike | str,
    docs: Iterable[Document | Tuple[str, str]],
    *,
    name: Optional[str] = None,
    overwrite: bool = False,
    max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
) -> BuildReport:
    """Build a fresh index file atomically.

    Duplicate ``doc_id`` values are rejected, empty documents are allowed
    (they simply have no postings), and every text is tokenised with the same
    lexer used for queries.  The file is written to a sibling temporary file
    and renamed into place, so a failed build never leaves a half index.
    Building over an existing path discards any retained generations of the
    previous index: it is a brand-new index, not a new generation.
    """
    started = time.monotonic()
    target = Path(path)
    if target.exists() and not overwrite:
        raise IndexExistsError(f"{target} already exists (pass overwrite=True)")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.build-{os.getpid()}-{time.time_ns()}.tmp")
    if tmp.exists():
        tmp.unlink()

    index_id = hashlib.sha256(
        f"{time.time_ns()}:{os.getpid()}".encode("ascii")
    ).hexdigest()[:16]
    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc_count, total_tokens, fingerprint = _write_index_file(
        tmp,
        docs,
        name=name or target.stem,
        index_id=index_id,
        generation=1,
        created_utc=created_utc,
        max_doc_chars=max_doc_chars,
    )
    os.replace(tmp, target)
    _discard_generation_state(target)
    _fsync_dir(target.parent)
    return BuildReport(
        path=str(target),
        doc_count=doc_count,
        token_count=total_tokens,
        index_fingerprint=fingerprint,
        elapsed_seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# generation registry, crash recovery and retention
# ---------------------------------------------------------------------------
#
# Layout next to the index file ``INDEX``:
#
# ``INDEX``                    the current generation (a complete index file)
# ``INDEX.gen-<N>``            a retained old generation (complete index file)
# ``INDEX.generations.json``   the registry: current pointer, retained list,
#                              retention setting and the publish journal
#
# A refresh never mutates the main file in place.  It builds the new
# generation as a complete file, journals a ``pending`` entry in the
# registry, then performs two atomic renames (old main aside, new main in)
# and finally clears the journal.  ``recover_generations`` resolves any
# interrupted publish the next time the index is opened, so a reopen always
# sees either the complete old generation or the complete new one.


def _registry_path(path: Path) -> Path:
    return path.with_name(path.name + ".generations.json")


def _gen_file_name(path: Path, generation: int) -> str:
    return f"{path.name}.gen-{generation}"


def _gen_file_path(path: Path, generation: int) -> Path:
    return path.with_name(_gen_file_name(path, generation))


def _read_registry(path: Path) -> Optional[Dict[str, object]]:
    reg_path = _registry_path(Path(path))
    try:
        raw = reg_path.read_bytes()
    except OSError:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != 1:
        return None
    return data


def _write_registry(path: Path, data: Dict[str, object]) -> None:
    reg_path = _registry_path(Path(path))
    tmp = reg_path.with_name(
        f".{reg_path.name}.{os.getpid()}-{time.time_ns()}.tmp"
    )
    tmp.write_bytes(json.dumps(data, indent=2, sort_keys=True).encode("utf-8"))
    _fsync_file(tmp)
    os.replace(tmp, reg_path)
    _fsync_dir(reg_path.parent)


def _read_main_meta(path: Path) -> Optional[Dict[str, str]]:
    """Meta of the main index file, or None when the file is missing.

    A file that exists but is not a readable nearidx database is reported via
    a sentinel so recovery leaves an unknown state untouched instead of
    making it worse.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        conn = sqlite3.connect(
            f"file:{p.resolve().as_posix()}?mode=ro", uri=True,
            isolation_level=None,
        )
        try:
            return _meta(conn)
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return {"__corrupt__": "1"}


def _trim_retained(
    retained: List[Dict[str, object]], keep: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split the retained list into the entries kept and the ones dropped."""
    kept = list(retained[:keep])
    dropped = list(retained[keep:])
    return kept, dropped


def _drop_retained_files(path: Path, entries: Iterable[Dict[str, object]]) -> None:
    for entry in entries:
        fname = entry.get("file")
        if not isinstance(fname, str) or not fname:
            continue
        candidate = Path(path).with_name(fname)
        # Only ever unlink derived generation files of this index.
        if candidate.name.startswith(f"{Path(path).name}.gen-"):
            try:
                candidate.unlink()
            except OSError:
                pass


def _cleanup_orphans(path: Path, reg: Optional[Dict[str, object]]) -> None:
    """Remove leftover temp files and unreferenced generation files.

    A generation file that no registry entry (retained or pending) points at
    is unpublished data; it must never be served, so it is deleted.  Only
    runs from the writer/recovery path, never on a plain read-only open.
    """
    path = Path(path)
    name = path.name
    referenced = set()
    if reg:
        for entry in reg.get("retained", []) or []:
            if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                referenced.add(entry["file"])
        pending = reg.get("pending")
        if isinstance(pending, dict) and isinstance(pending.get("generation"), int):
            referenced.add(_gen_file_name(path, pending["generation"]))
    try:
        children = list(path.parent.iterdir())
    except OSError:
        return
    for child in children:
        n = child.name
        if n.startswith(f".{name}.") and n.endswith(".tmp"):
            try:
                child.unlink()
            except OSError:
                pass
        elif (
            n.startswith(f"{name}.gen-")
            and n[len(name) + 5:].isdigit()
            and n not in referenced
        ):
            try:
                child.unlink()
            except OSError:
                pass


def recover_generations(path: os.PathLike | str) -> None:
    """Resolve an interrupted publish, if one is journaled.

    After this returns, the main file is either the complete old generation
    or the complete new one, and the registry matches reality.  It is a no-op
    when the index was never refreshed or the last publish completed.
    """
    path = Path(path)
    reg = _read_registry(path)
    if reg is None:
        return
    pending = reg.get("pending")
    if not isinstance(pending, dict):
        return
    new_gen = pending.get("generation")
    new_fp = pending.get("fingerprint")
    from_gen = pending.get("from_generation")
    from_fp = pending.get("from_fingerprint")
    if not all(isinstance(v, (int, str)) for v in (new_gen, new_fp, from_gen, from_fp)):
        # A journal we cannot understand: leave everything untouched.
        return
    main = _read_main_meta(path)
    if main is not None and "__corrupt__" in main:
        return
    if (
        main is not None
        and main.get("generation") == str(new_gen)
        and main.get("index_fingerprint") == new_fp
    ):
        # The new generation was installed but the registry was never
        # finalised: roll forward.
        retained: List[Dict[str, object]] = []
        if _gen_file_path(path, int(from_gen)).is_file():
            retained.append(
                {
                    "generation": int(from_gen),
                    "fingerprint": from_fp,
                    "file": _gen_file_name(path, int(from_gen)),
                }
            )
        for entry in reg.get("retained", []) or []:
            if not isinstance(entry, dict):
                continue
            g = entry.get("generation")
            if isinstance(g, int) and g != int(from_gen) and g < int(new_gen):
                retained.append(entry)
        keep = reg.get("retain_old", DEFAULT_RETAIN_OLD)
        if not isinstance(keep, int) or keep < 1:
            keep = DEFAULT_RETAIN_OLD
        kept, dropped = _trim_retained(retained, keep)
        reg["current"] = {"generation": int(new_gen), "fingerprint": new_fp}
        reg["retained"] = kept
        reg["pending"] = None
        _write_registry(path, reg)
        _drop_retained_files(path, dropped)
        stray = _gen_file_path(path, int(new_gen))
        if stray.exists():
            try:
                stray.unlink()
            except OSError:
                pass
    else:
        # The publish never completed: the main file is still the old
        # generation (or was moved aside), so roll back to it.
        if main is None:
            old_file = _gen_file_path(path, int(from_gen))
            if old_file.is_file():
                os.replace(old_file, path)
                _fsync_dir(path.parent)
        staged = _gen_file_path(path, int(new_gen))
        if staged.exists():
            try:
                staged.unlink()
            except OSError:
                pass
        reg["pending"] = None
        _write_registry(path, reg)
    _cleanup_orphans(path, reg)


def _discard_generation_state(path: Path) -> None:
    """Remove registry and retained generations (fresh build over a path)."""
    reg = _read_registry(path)
    _cleanup_orphans(path, reg)
    try:
        _registry_path(path).unlink()
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class GenerationInfo:
    """One retained generation of an index."""

    generation: int
    fingerprint: str
    path: str
    current: bool


# ---------------------------------------------------------------------------
# read / reopen
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AccessTrace:
    """Evidence of what a query actually touched in the index.

    ``candidate_docs`` are doc_ids returned by the inverted table for the
    query's required tokens; ``docs_solved`` had postings loaded; ``body_reads``
    lists doc_ids whose original text was fetched from ``documents`` (only
    those that produced deliverable windows need it).
    """

    candidate_docs: int = 0
    docs_considered: int = 0
    docs_solved: int = 0
    posting_rows_fetched: int = 0
    postings_queries: int = 0
    body_reads: List[str] = field(default_factory=list)
    missing_token_docs: int = 0
    pruned_by_leaf_count: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate_docs": self.candidate_docs,
            "docs_considered": self.docs_considered,
            "docs_solved": self.docs_solved,
            "posting_rows_fetched": self.posting_rows_fetched,
            "postings_queries": self.postings_queries,
            "body_reads": list(self.body_reads),
            "missing_token_docs": self.missing_token_docs,
            "pruned_by_leaf_count": self.pruned_by_leaf_count,
        }


class Index:
    """A read-only handle on a built index, safe to reopen in another process.

    The handle always serves the generation that was current when it was
    opened.  :meth:`for_generation` resolves older retained generations for
    pagination cursors that were issued before a refresh was published.
    """

    def __init__(self, path: os.PathLike | str, *, trace: Optional[AccessTrace] = None):
        self.path = Path(path)
        recover_generations(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        try:
            self._conn = sqlite3.connect(uri, uri=True, isolation_level=None)
            self._conn.execute("PRAGMA query_only = ON")
            meta = _meta(self._conn)
        except sqlite3.DatabaseError as exc:
            raise IndexFormatError(
                f"{self.path} is not a nearidx database: {exc}"
            ) from exc
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise IndexFormatError(
                f"unsupported schema_version {meta.get('schema_version')!r} "
                f"(expected {SCHEMA_VERSION})"
            )
        self.meta: Dict[str, str] = meta
        self.name = meta.get("name", self.path.stem)
        self.index_id = meta["index_id"]
        self.fingerprint = meta["index_fingerprint"]
        self.generation = int(meta.get("generation", "1"))
        self.trace = trace
        self._gen_cache: Dict[str, "Index"] = {}

    @property
    def conn(self) -> sqlite3.Connection:
        """Read-only connection for modules that build SQL themselves."""
        return self._conn

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        for sub in self._gen_cache.values():
            try:
                sub.close()
            except sqlite3.Error:
                pass
        self._gen_cache.clear()
        self._conn.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _t(self) -> Optional[AccessTrace]:
        return self.trace

    # -- generations -----------------------------------------------------
    def generations(self) -> List[GenerationInfo]:
        """The current generation plus every retained old generation."""
        infos = [
            GenerationInfo(
                generation=self.generation,
                fingerprint=self.fingerprint,
                path=str(self.path),
                current=True,
            )
        ]
        reg = _read_registry(self.path)
        if reg:
            for entry in reg.get("retained", []) or []:
                if not isinstance(entry, dict):
                    continue
                g = entry.get("generation")
                fp = entry.get("fingerprint")
                fname = entry.get("file")
                if not isinstance(g, int) or not isinstance(fp, str):
                    continue
                if not isinstance(fname, str) or not fname:
                    fname = _gen_file_name(self.path, g)
                infos.append(
                    GenerationInfo(
                        generation=g,
                        fingerprint=fp,
                        path=str(self.path.with_name(fname)),
                        current=False,
                    )
                )
        return infos

    def for_generation(self, index_sig: str) -> "Index":
        """Resolve the generation a pagination cursor was issued against.

        Returns this handle when the cursor's fingerprint is the generation
        the handle itself serves (an open handle pins its generation).
        Otherwise the registry is consulted on every call: the current
        generation is reopened from the main path, a retained old generation
        is opened (and cached) from its generation file, and anything else
        raises :class:`GenerationExpiredError` -- a cursor whose generation
        fell out of retention is never silently served current data.
        """
        if index_sig == self.fingerprint:
            return self
        reg = _read_registry(self.path)
        target_path: Optional[Path] = None
        if reg:
            current = reg.get("current")
            if (
                isinstance(current, dict)
                and current.get("fingerprint") == index_sig
                and isinstance(current.get("generation"), int)
                and current["generation"] != self.generation
            ):
                # The registry still points at the current generation, but
                # this handle was opened before it was published: reopen the
                # main path to see it.
                target_path = self.path
            if target_path is None:
                for entry in reg.get("retained", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("fingerprint") != index_sig:
                        continue
                    fname = entry.get("file")
                    if isinstance(fname, str) and fname:
                        target_path = self.path.with_name(fname)
                    elif isinstance(entry.get("generation"), int):
                        target_path = _gen_file_path(self.path, entry["generation"])
                    break
        if target_path is None:
            raise GenerationExpiredError(
                f"index generation with fingerprint {index_sig[:12]}... is no "
                "longer retained (it expired or never existed); pagination "
                "cannot continue -- start a new search"
            )
        cached = self._gen_cache.get(index_sig)
        if cached is not None:
            return cached
        try:
            sub = Index(target_path)
        except (FileNotFoundError, IndexFormatError) as exc:
            raise GenerationExpiredError(
                f"retained generation file for fingerprint {index_sig[:12]}... "
                f"is unavailable: {exc}"
            ) from exc
        if sub.fingerprint != index_sig:
            sub.close()
            raise GenerationExpiredError(
                f"index generation with fingerprint {index_sig[:12]}... is no "
                "longer retained; pagination cannot continue"
            )
        self._gen_cache[index_sig] = sub
        return sub

    def stats(self) -> Dict[str, object]:
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(token_count),0) FROM documents"
        ).fetchone()
        distinct = self._conn.execute(
            "SELECT COUNT(DISTINCT token) FROM postings"
        ).fetchone()[0]
        return {
            "doc_count": row[0],
            "token_occurrences": row[1],
            "distinct_tokens": distinct,
            "index_id": self.index_id,
            "fingerprint": self.fingerprint,
            "name": self.name,
            "generation": self.generation,
            "retained_generations": [
                g.generation for g in self.generations() if not g.current
            ],
        }

    # -- documents -------------------------------------------------------
    def doc_ids(self) -> List[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT doc_id FROM documents ORDER BY doc_id"
        )]

    def doc_meta(self, doc_id: str) -> Optional[Dict[str, object]]:
        row = self._conn.execute(
            "SELECT doc_id, text_sha256, char_len, byte_len, token_count, "
            "postings_digest, source_path, source_sha256 FROM documents "
            "WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("doc_id", "text_sha256", "char_len", "byte_len", "token_count",
                "postings_digest", "source_path", "source_sha256")
        return dict(zip(keys, row))

    def text(self, doc_id: str) -> str:
        """Fetch the original text.  Only called for documents that need it."""
        row = self._conn.execute(
            "SELECT text FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        tr = self._t()
        if tr is not None:
            tr.body_reads.append(doc_id)
        return row[0]

    def snippet(self, doc_id: str, char_start: int, char_end: int) -> str:
        """Cut the original span, no token re-joining."""
        if char_start < 0 or char_end < char_start:
            raise ValueError("invalid character span")
        text = self.text(doc_id)
        if char_end > len(text):
            raise IndexFormatError(
                f"span {char_start}:{char_end} exceeds document {doc_id!r}"
            )
        return text[char_start:char_end]

    def token_spans(
        self, doc_id: str, pos_start: int, pos_end: int
    ) -> List[Tuple[int, int, int, int, int]]:
        """Posting spans for a position range of one document."""
        return self._conn.execute(
            "SELECT pos, char_start, char_end, byte_start, byte_end FROM postings "
            "WHERE doc_id = ? AND pos BETWEEN ? AND ? ORDER BY pos",
            (doc_id, pos_start, pos_end),
        ).fetchall()

    # -- inverted index --------------------------------------------------
    def postings_for_doc(self, tokens: Sequence[str], doc_id: str) -> Dict[str, list]:
        tr = self._t()
        if tr is not None:
            tr.postings_queries += 1
        placeholders = ",".join("?" for _ in tokens)
        rows = self._conn.execute(
            "SELECT token, pos, char_start, char_end, byte_start, byte_end "
            "FROM postings WHERE doc_id = ? AND token IN (" + placeholders + ") "
            "ORDER BY pos",
            (doc_id, *tokens),
        ).fetchall()
        if tr is not None:
            tr.posting_rows_fetched += len(rows)
        out: Dict[str, list] = {t: [] for t in tokens}
        for token, pos, cs, ce, bs, be in rows:
            out[token].append((pos, cs, ce, bs, be))
        return out

    # -- integrity / binding --------------------------------------------
    def verify(
        self, *, full: bool = True, sample: int = 0, check_sources: bool = True,
        generations: bool = False,
    ) -> List[str]:
        """Check structural and content binding.

        ``full`` re-tokenises every stored text and recomputes its posting
        digest, so a database whose text was swapped without rebuild is
        reported.  With ``generations=True`` every retained old generation is
        checked against its registered fingerprint the same way.  Returns a
        list of problem strings (empty means OK).
        """
        problems: List[str] = []
        quick = self._conn.execute("PRAGMA quick_check").fetchone()
        if not quick or quick[0] != "ok":
            problems.append(f"quick_check: {quick[0] if quick else 'no result'}")
        rows = self._conn.execute(
            "SELECT doc_id, text, text_sha256, char_len, byte_len, token_count, "
            "postings_digest, source_path, source_sha256 FROM documents "
            "ORDER BY doc_id"
        ).fetchall()
        meta = meta_rows(self._conn)
        if meta.get("doc_count") != str(len(rows)):
            problems.append("meta doc_count does not match documents table")
        checked = 0
        fp_rows: List[Tuple[str, str, str]] = []
        for doc_id, text, sha, clen, blen, tcount, pdigest, spath, ssha in rows:
            fp_rows.append((doc_id, sha, pdigest))
            if full or (sample and checked < sample):
                checked += 1
                if text_sha256(text) != sha:
                    problems.append(f"{doc_id}: stored text does not match sha256")
                if len(text) != clen or len(text.encode("utf-8")) != blen:
                    problems.append(f"{doc_id}: stored length metadata mismatch")
                toks = token_spans(text)
                if len(toks) != tcount:
                    problems.append(f"{doc_id}: token_count mismatch")
                digest = _postings_digest(
                    (t.pos, t.norm, t.char_start, t.char_end, t.byte_start,
                     t.byte_end) for t in toks
                )
                if digest != pdigest:
                    problems.append(f"{doc_id}: postings digest mismatch")
                stored = self._conn.execute(
                    "SELECT pos, token, char_start, char_end, byte_start, byte_end "
                    "FROM postings WHERE doc_id = ? ORDER BY pos", (doc_id,)
                ).fetchall()
                recomputed = [
                    (t.pos, t.norm, t.char_start, t.char_end, t.byte_start, t.byte_end)
                    for t in toks
                ]
                if [tuple(r) for r in stored] != recomputed:
                    problems.append(f"{doc_id}: posting rows differ from text")
            if check_sources and ssha is not None and spath is not None:
                try:
                    actual = hashlib.sha256(Path(spath).read_bytes()).hexdigest()
                except OSError as exc:
                    problems.append(f"{doc_id}: source file unreadable: {exc}")
                else:
                    if actual != ssha:
                        problems.append(
                            f"{doc_id}: source file changed since indexing "
                            f"(expected sha256 {ssha}, got {actual})"
                        )
        expected_fp = _fingerprint(fp_rows)
        if expected_fp != meta.get("index_fingerprint"):
            problems.append("index fingerprint does not match document set")
        if generations:
            problems.extend(self._verify_retained(full=full, sample=sample,
                                                  check_sources=check_sources))
        return problems

    def _verify_retained(
        self, *, full: bool, sample: int, check_sources: bool
    ) -> List[str]:
        """Verify every retained old generation against the registry."""
        problems: List[str] = []
        reg = _read_registry(self.path)
        if reg is None:
            return problems
        seen_files = set()
        for entry in reg.get("retained", []) or []:
            if not isinstance(entry, dict):
                problems.append("registry: malformed retained entry")
                continue
            g = entry.get("generation")
            fp = entry.get("fingerprint")
            fname = entry.get("file")
            label = f"generation {g}"
            if not isinstance(g, int) or not isinstance(fp, str):
                problems.append("registry: retained entry missing generation "
                                "or fingerprint")
                continue
            if not isinstance(fname, str) or not fname:
                fname = _gen_file_name(self.path, g)
            if fname in seen_files:
                problems.append(f"{label}: duplicate retained file {fname}")
                continue
            seen_files.add(fname)
            gpath = self.path.with_name(fname)
            if not gpath.is_file():
                problems.append(f"{label}: retained file {fname} is missing")
                continue
            try:
                sub = Index(gpath)
            except (FileNotFoundError, IndexFormatError) as exc:
                problems.append(f"{label}: cannot open retained file: {exc}")
                continue
            try:
                if sub.generation != g:
                    problems.append(
                        f"{label}: file reports generation {sub.generation}"
                    )
                if sub.fingerprint != fp:
                    problems.append(
                        f"{label}: registry fingerprint does not match file"
                    )
                for problem in sub.verify(
                    full=full, sample=sample, check_sources=check_sources
                ):
                    problems.append(f"{label}: {problem}")
            finally:
                sub.close()
        return problems

    def check_source(self, doc_id: str, *, path: Optional[str] = None,
                     text: Optional[str] = None) -> None:
        """Bind an external file/text to this index entry.

        Raises :class:`StaleSourceError` when the supplied content differs from
        what was indexed.  The caller must then call :meth:`rebuild_document`
        explicitly; nothing is silently re-pointed.
        """
        meta = self.doc_meta(doc_id)
        if meta is None:
            raise KeyError(doc_id)
        if text is not None:
            actual_sha = text_sha256(text)
            what = "supplied text"
        else:
            p = Path(path or meta["source_path"])
            raw = p.read_bytes()
            actual_sha = hashlib.sha256(raw).hexdigest()
            what = str(p)
        if actual_sha != meta["text_sha256"]:
            raise StaleSourceError(
                f"document {doc_id!r} {what} sha256 {actual_sha} does not match "
                f"indexed sha256 {meta['text_sha256']}; the document must be "
                "rebuilt explicitly before it can be queried"
            )


def meta_rows(conn: sqlite3.Connection) -> Dict[str, str]:
    return {k: v for k, v in conn.execute("SELECT key, value FROM meta")}


def open_index(path: os.PathLike | str, *, verify: bool = True,
               trace: Optional[AccessTrace] = None) -> Index:
    """Reopen a built index from a new process.

    Rejects databases whose fingerprint or document table is inconsistent.
    """
    idx = Index(path, trace=trace)
    if verify:
        # Internal consistency only: a stale *source* file is surfaced by
        # verify(check_sources=True) and by Index.check_source, but it must not
        # stop the index from being queried -- the stored text is still the
        # content the index was built from.
        problems = idx.verify(full=False, check_sources=False)
        if problems:
            idx.close()
            raise IndexFormatError("; ".join(problems))
    return idx


# ---------------------------------------------------------------------------
# explicit rebuild
# ---------------------------------------------------------------------------

def rebuild_document(
    path: os.PathLike | str,
    doc_id: str,
    text: str,
    *,
    source_path: Optional[str] = None,
    source_sha256: Optional[str] = None,
) -> str:
    """Replace one document's text and postings in place, atomically.

    This is the single-document repair path; for a batch of changes use
    :func:`nearidx.refresh.refresh_index`, which publishes a new generation
    and keeps the old one queryable.  The in-place rebuild changes the index
    fingerprint without retaining the old generation, so pagination cursors
    issued against the old content are rejected instead of silently skipping
    or duplicating windows.
    """
    from .lexer import token_spans as _spans

    if not isinstance(text, str):
        raise ValueError("text must be str")
    target = Path(path)
    recover_generations(target)
    conn = sqlite3.connect(str(target), isolation_level=None)
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        sha = text_sha256(text)
        toks = _spans(text)
        digest = _postings_digest(
            (t.pos, t.norm, t.char_start, t.char_end, t.byte_start, t.byte_end)
            for t in toks
        )
        conn.execute("DELETE FROM postings WHERE doc_id = ?", (doc_id,))
        conn.execute(
            "UPDATE documents SET text=?, text_sha256=?, char_len=?, byte_len=?, "
            "token_count=?, postings_digest=?, source_path=?, source_sha256=? "
            "WHERE doc_id=?",
            (text, sha, len(text), len(text.encode("utf-8")), len(toks), digest,
             source_path, source_sha256, doc_id),
        )
        conn.executemany(
            "INSERT INTO postings(token, doc_id, pos, char_start, char_end, "
            "byte_start, byte_end) VALUES (?,?,?,?,?,?,?)",
            [(t.norm, doc_id, t.pos, t.char_start, t.char_end, t.byte_start,
              t.byte_end) for t in toks],
        )
        rows = conn.execute(
            "SELECT doc_id, text_sha256, postings_digest FROM documents "
            "ORDER BY doc_id"
        ).fetchall()
        fp = _fingerprint([tuple(r) for r in rows])
        meta_now = dict(conn.execute("SELECT key, value FROM meta"))
        generation = int(meta_now["generation"]) + 1
        conn.execute(
            "UPDATE meta SET value=? WHERE key='index_fingerprint'", (fp,)
        )
        conn.execute(
            "UPDATE meta SET value=? WHERE key='generation'", (str(generation),)
        )
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()
    return fp
