"""Streaming archive/restore of an index and every generation a cursor may pin.

What travels together
---------------------
A nearidx index is more than ``INDEX``: retained old generations
(``INDEX.gen-<N>``) keep in-flight single-index and federated pagination
cursors readable, and ``INDEX.generations.json`` binds those files to the
index identity and generation fingerprints.  :func:`archive_index` writes one
self-contained archive containing the current generation, **all** retained
generations registered at that moment, the registry file itself, and a
versioned manifest describing the whole generation state.

The archive is streamed: member files are copied in fixed-size chunks while
their length and SHA-256 are computed; no member is ever read into memory as a
whole.  The manifest is written after the payload, followed by a fixed footer
that binds the manifest bytes.  Archiving snapshots the registry bytes and
every member's file signature before copying and re-checks them afterwards: if
a refresh, in-place rebuild or registry rewrite raced the copy, the archive is
abandoned and nothing is published at the destination.

Restore
-------
:func:`restore_index` refuses to touch an existing path.  It validates the
container, manifest version, member list, payload lengths and every member's
SHA-256 *before* opening any member as SQLite, then fully re-verifies each
generation's stored text/postings binding, ``index_id``, generation number and
fingerprint against the manifest.  Only then are files installed, with the
current-generation file renamed into place last; the restored registry maps
the archive's original file names onto the new target's file names while
preserving identity, generation numbers and fingerprints, so single-index and
federated cursors issued before the archive keep working -- also after a
restore under a different file name.

Rejections (path traversal, duplicate or unregistered members, symlinks,
missing or extra bytes, corrupt databases, unsupported manifest versions)
never leave an openable partial index and never overwrite an existing one.
An interrupted install is journaled through a marker file: a later restore
finishes or discards it, and the target path is only made visible by the
final rename of the complete current generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sqlite3
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .db import (
    SCHEMA_VERSION,
    IndexErrorBase,
    _fingerprint,
    _fsync_dir,
    _fsync_file,
    _gen_file_name,
    _meta,
    _postings_digest,
    _read_main_meta,
    _registry_path,
    _write_registry,
    text_sha256,
)
from .lexer import token_spans

ARCHIVE_FORMAT = "nearidx-archive"
ARCHIVE_VERSION = 1
MANIFEST_VERSION = 1
FOOTER_VERSION = 1

#: Fixed copy/buffer size.  Members are streamed through this; the archive
#: never holds a member file in memory, only one chunk at a time.
CHUNK_SIZE = 1024 * 1024

#: Logical uses of an archive member.
ROLE_CURRENT = "current"
ROLE_GENERATION = "generation"
ROLE_REGISTRY = "registry"
_ROLES = (ROLE_CURRENT, ROLE_GENERATION, ROLE_REGISTRY)

_INSTALL_MARKER_SUFFIX = ".nearidx-restore"

#: Indirection for ``open`` so tests can instrument streaming reads.
_open = open

__all__ = [
    "ArchiveError",
    "ArchiveFormatError",
    "ArchiveExistsError",
    "ArchiveManifestError",
    "ArchiveMember",
    "ArchiveManifest",
    "ArchiveReport",
    "RestoreReport",
    "archive_index",
    "inspect_archive",
    "restore_index",
    "recover_restore",
]


class ArchiveError(IndexErrorBase):
    """Base class for archive/restore errors."""


class ArchiveFormatError(ArchiveError):
    """The archive container, manifest or a member failed validation."""


class ArchiveManifestError(ArchiveFormatError):
    """The manifest is malformed or has an unsupported version."""


class ArchiveExistsError(ArchiveError):
    """The destination archive/index path already exists."""


# ---------------------------------------------------------------------------
# manifest model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """One file inside the archive.

    ``member_path`` is the archive-internal (flat) name; ``file_name`` is the
    original file name next to the source index, kept so the logical role is
    described without leaking absolute paths.
    """

    role: str
    generation: int
    fingerprint: str
    file_name: str
    member_path: str
    offset: int
    size: int
    sha256: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "role": self.role,
            "generation": self.generation,
            "fingerprint": self.fingerprint,
            "file_name": self.file_name,
            "path": self.member_path,
            "offset": self.offset,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """The validated, versioned manifest of an archive.

    ``retained`` is the registry's retained list in registry order (oldest
    last, as ``retain_old`` trimming keeps it).
    """

    manifest_version: int
    format: str
    archive_version: int
    index_id: str
    name: str
    schema_version: int
    fingerprint_version: str
    current_generation: int
    current_fingerprint: str
    retained: Tuple[Tuple[int, str], ...]
    retain_old: int
    created_utc: str
    members: Tuple[ArchiveMember, ...]

    @property
    def has_registry(self) -> bool:
        return any(m.role == ROLE_REGISTRY for m in self.members)

    def member_for(
        self, role: str, generation: Optional[int] = None
    ) -> Optional[ArchiveMember]:
        for m in self.members:
            if m.role == role and (generation is None or m.generation == generation):
                return m
        return None

    def generation_fingerprints(self) -> Dict[int, str]:
        out = {self.current_generation: self.current_fingerprint}
        for gen, fp in self.retained:
            out[gen] = fp
        return out

    def as_dict(self) -> Dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "format": self.format,
            "archive_version": self.archive_version,
            "index_id": self.index_id,
            "name": self.name,
            "schema_version": self.schema_version,
            "fingerprint_version": self.fingerprint_version,
            "created_utc": self.created_utc,
            "current": {
                "generation": self.current_generation,
                "fingerprint": self.current_fingerprint,
            },
            "retained": [
                {"generation": g, "fingerprint": fp} for g, fp in self.retained
            ],
            "retain_old": self.retain_old,
            "members": [m.as_dict() for m in self.members],
        }


@dataclass(slots=True)
class ArchiveReport:
    """What :func:`archive_index` wrote."""

    path: str
    index_id: str
    current_generation: int
    current_fingerprint: str
    generations: List[int]
    members: List[Dict[str, object]]
    size: int
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "index_id": self.index_id,
            "current_generation": self.current_generation,
            "current_fingerprint": self.current_fingerprint,
            "generations": list(self.generations),
            "members": [dict(m) for m in self.members],
            "size": self.size,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class RestoreReport:
    """What :func:`restore_index` installed."""

    path: str
    archive: str
    index_id: str
    current_generation: int
    current_fingerprint: str
    generations: List[int]
    registry_path: Optional[str]
    members: List[Dict[str, object]]
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "archive": self.archive,
            "index_id": self.index_id,
            "current_generation": self.current_generation,
            "current_fingerprint": self.current_fingerprint,
            "generations": list(self.generations),
            "registry_path": self.registry_path,
            "members": [dict(m) for m in self.members],
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# container primitives
# ---------------------------------------------------------------------------
#
# Layout:
#
#   header:           8s magic b"NEARIDXA", u32 version, u64 manifest_offset,
#                     u64 manifest_length, u32 reserved       (32 bytes)
#   payload:          member files concatenated, each exactly `size` bytes
#   manifest:         UTF-8 JSON (versioned, lists every member)
#   footer:           32s manifest_sha256, u64 manifest_length, u32 version,
#                     8s magic b"NEARIDXF"                    (52 bytes)
#
# The header and footer disagree if bytes were inserted, removed or appended;
# the manifest's member offsets must tile the payload region exactly, so a
# missing or extra member cannot hide.

_MAGIC = b"NEARIDXA"
_FOOTER_MAGIC = b"NEARIDXF"
_HEADER = struct.Struct("<8sIQQI")
_HEADER_SIZE = _HEADER.size
_FOOTER = struct.Struct("<32sQI8s")
_FOOTER_SIZE = _FOOTER.size
_MAX_MANIFEST = 64 * 1024 * 1024


def _strict_json(raw: bytes) -> object:
    """JSON parse that rejects duplicate keys (they hide members)."""

    def no_dupes(pairs: List[Tuple[str, object]]) -> Dict[str, object]:
        obj: Dict[str, object] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key {key!r}")
            obj[key] = value
        return obj

    return json.loads(raw.decode("utf-8"), object_pairs_hook=no_dupes)


def _copy_stream(src: Path, dst, *, chunk_size: int = CHUNK_SIZE) -> Tuple[int, str]:
    """Copy *src* to *dst* in bounded chunks, hashing as we go."""
    h = hashlib.sha256()
    size = 0
    with _open(src, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def _file_signature(path: Path) -> Optional[Tuple[int, int, int, int]]:
    """Identity of a file's contents for concurrent-change detection.

    ``(st_dev, st_ino, st_size, st_mtime_ns)``; a refresh renames a fresh
    file in (new inode/mtime), an in-place rebuild changes size or mtime.
    """
    try:
        st = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _member_name(role: str, generation: int = 0) -> str:
    if role == ROLE_REGISTRY:
        return "generations.json"
    return f"{role}-{generation}.sqlite"


def _is_symlink(path: Path) -> bool:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(st.st_mode)


def _reject_symlink(path: Path, *, what: str) -> None:
    if _is_symlink(path):
        raise ArchiveFormatError(f"{what} {str(path)!r} is a symbolic link")


def _safe_member_path(name: object) -> str:
    """Validate an archive-internal member name; reject traversal outright.

    Members are flat files inside the staging directory.  Absolute paths,
    drive prefixes, separators, ``.``/``..`` segments, empty components, NUL
    bytes and over-long names are all rejected, so a crafted manifest can
    never make the extractor write outside the staging directory.
    """
    if not isinstance(name, str) or not name:
        raise ArchiveManifestError("member path must be a non-empty string")
    if "\x00" in name:
        raise ArchiveManifestError(f"member path {name!r} contains NUL")
    if name.startswith(("/", "\\")):
        raise ArchiveManifestError(f"member path {name!r} is absolute")
    if len(name) > 255:
        raise ArchiveManifestError(f"member path {name!r} is too long")
    if len(name) >= 2 and name[1] == ":":
        raise ArchiveManifestError(f"member path {name!r} has a drive prefix")
    if "/" in name or "\\" in name:
        raise ArchiveManifestError(
            f"member path {name!r} must not contain a directory component"
        )
    if name in (".", ".."):
        raise ArchiveManifestError(
            f"member path {name!r} is not a plain file name"
        )
    return name


def _safe_file_name(name: object, *, where: str) -> str:
    if not isinstance(name, str) or not name:
        raise ArchiveManifestError(f"{where} must be a non-empty string")
    if "\x00" in name:
        raise ArchiveManifestError(f"{where} contains NUL")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise ArchiveManifestError(f"{where} must be a plain file name")
    return name


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------

def _read_registry_bytes(index_path: Path) -> Optional[Tuple[bytes, Dict[str, object]]]:
    """Read and structurally validate the registry next to *index_path*."""
    reg_path = _registry_path(index_path)
    if not reg_path.exists():
        return None
    _reject_symlink(reg_path, what="generation registry")
    try:
        raw = reg_path.read_bytes()
    except OSError as exc:
        raise ArchiveFormatError(
            f"cannot read generation registry {str(reg_path)!r}: {exc}"
        ) from exc
    try:
        data = _strict_json(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArchiveFormatError(
            f"generation registry {str(reg_path)!r} is unreadable: {exc}"
        ) from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ArchiveFormatError(
            f"generation registry {str(reg_path)!r} has an unsupported version"
        )
    return raw, data


def _snapshot_generation_state(
    index_path: Path,
) -> Tuple[Dict[str, object], List[Tuple[str, int, str, str, Path]], Optional[bytes]]:
    """Snapshot the current file, registry and every retained generation.

    Returns ``(meta, members, registry_bytes)`` where ``members`` has entries
    ``(role, generation, fingerprint, original_file_name, source_path)``.
    Any inconsistency (registry not matching the current file, missing or
    symlinked generation file, registration mismatch) fails the archive
    instead of producing a set that cannot be restored coherently.
    """
    _reject_symlink(index_path, what="index")
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    main_meta = _read_main_meta(index_path)
    if main_meta is None:
        raise FileNotFoundError(index_path)
    if "__corrupt__" in main_meta:
        raise ArchiveFormatError(f"{index_path} is not a readable nearidx index")
    if main_meta.get("schema_version") != str(SCHEMA_VERSION):
        raise ArchiveFormatError(
            f"unsupported schema_version {main_meta.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    index_id = main_meta.get("index_id")
    fingerprint = main_meta.get("index_fingerprint")
    try:
        generation = int(main_meta.get("generation", "1"))
    except (TypeError, ValueError):
        raise ArchiveFormatError(
            f"index reports a non-numeric generation "
            f"{main_meta.get('generation')!r}"
        ) from None
    if not isinstance(index_id, str) or not index_id:
        raise ArchiveFormatError("index has no index_id")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ArchiveFormatError("index has no index_fingerprint")

    members: List[Tuple[str, int, str, str, Path]] = [
        (ROLE_CURRENT, generation, fingerprint, index_path.name, index_path)
    ]
    registry_raw: Optional[bytes] = None
    retained: List[Tuple[int, str]] = []
    retain_old = 0
    read = _read_registry_bytes(index_path)
    if read is not None:
        registry_raw, reg = read
        current = reg.get("current")
        if not isinstance(current, dict) or \
                current.get("fingerprint") != fingerprint or \
                current.get("generation") != generation:
            raise ArchiveFormatError(
                "generation registry does not describe the current index file "
                f"(registry: {current!r}, file: generation {generation}, "
                f"fingerprint {fingerprint[:12]}...)"
            )
        if reg.get("index_id") not in (None, index_id):
            raise ArchiveFormatError(
                "generation registry belongs to a different index identity"
            )
        if isinstance(reg.get("pending"), dict):
            raise ArchiveFormatError(
                "index has an interrupted publish journaled; open the index "
                "(which recovers it) before archiving"
            )
        keep = reg.get("retain_old")
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ArchiveFormatError(
                "generation registry has an invalid retain_old setting"
            )
        retain_old = keep
        entries = reg.get("retained")
        if not isinstance(entries, list):
            raise ArchiveFormatError("generation registry has no retained list")
        seen_gens = {generation}
        seen_files = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ArchiveFormatError("registry: malformed retained entry")
            g = entry.get("generation")
            fp = entry.get("fingerprint")
            fname = entry.get("file")
            if isinstance(g, bool) or not isinstance(g, int) or g < 1:
                raise ArchiveFormatError(
                    "registry: retained entry has an invalid generation"
                )
            if not isinstance(fp, str) or not fp:
                raise ArchiveFormatError(
                    f"registry: generation {g} has no fingerprint"
                )
            if g in seen_gens:
                raise ArchiveFormatError(
                    f"registry: generation {g} is registered more than once"
                )
            if not isinstance(fname, str) or not fname:
                fname = _gen_file_name(index_path, g)
            _safe_file_name(fname, where=f"registry: generation {g} file")
            if fname in seen_files:
                raise ArchiveFormatError(
                    f"registry: duplicate retained file {fname!r}"
                )
            seen_gens.add(g)
            seen_files.add(fname)
            retained.append((g, fp))
            gpath = index_path.with_name(fname)
            _reject_symlink(gpath, what=f"retained generation {g} file")
            if not gpath.is_file():
                raise ArchiveFormatError(
                    f"retained generation {g} file {fname!r} is missing; the "
                    "index state is inconsistent"
                )
            gmeta = _read_main_meta(gpath)
            if gmeta is None or "__corrupt__" in gmeta:
                raise ArchiveFormatError(
                    f"retained generation {g} file {fname!r} is not a readable "
                    "nearidx index"
                )
            if gmeta.get("index_id") != index_id:
                raise ArchiveFormatError(
                    f"retained generation {g} belongs to a different index"
                )
            if gmeta.get("generation") != str(g) or \
                    gmeta.get("index_fingerprint") != fp:
                raise ArchiveFormatError(
                    f"retained generation {g} file does not match its registry "
                    "entry (generation/fingerprint mismatch)"
                )
            members.append((ROLE_GENERATION, g, fp, fname, gpath))
        members.append((
            ROLE_REGISTRY, 0, "", _registry_path(index_path).name,
            _registry_path(index_path),
        ))
        if len(retained) > retain_old:
            raise ArchiveFormatError(
                "generation registry retains more generations than retain_old"
            )
    meta = {
        "index_id": index_id,
        "name": main_meta.get("name", ""),
        "schema_version": int(main_meta["schema_version"]),
        "fingerprint_version": main_meta.get("fingerprint_version", ""),
        "generation": generation,
        "fingerprint": fingerprint,
        "retained": retained,
        "retain_old": retain_old,
    }
    return meta, members, registry_raw


def archive_index(
    index: os.PathLike | str,
    archive: os.PathLike | str,
    *,
    overwrite: bool = False,
) -> ArchiveReport:
    """Write a streaming archive of *index* and every retained generation.

    The archive holds the current generation, every generation still
    registered at snapshot time, the registry file and a versioned manifest
    (roles, sizes, SHA-256, index identity, generation numbers and
    fingerprints).  If the index is refreshed, rebuilt or its registry
    rewritten while the archive is written, the call fails with
    :class:`ArchiveFormatError` and the unfinished destination is removed: a
    published archive always describes one complete, consistent registry
    state.  A pre-existing destination is only replaced with
    ``overwrite=True``, and even then it is replaced by one atomic rename.
    """
    started = time.monotonic()
    src_index = Path(index)
    dst = Path(archive)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise ArchiveExistsError(
                f"{dst} already exists (pass overwrite=True)"
            )
        _reject_symlink(dst, what="archive destination")
        if not dst.is_file():
            raise ArchiveExistsError(
                f"{dst} exists and is not a regular file"
            )
    dst.parent.mkdir(parents=True, exist_ok=True)

    meta, members, registry_raw = _snapshot_generation_state(src_index)
    # File signatures taken after the registry snapshot: any later change to
    # a member or to the registry is caught by the checks below.
    signatures = {
        str(path): _file_signature(path)
        for _role, _gen, _fp, _name, path in members
    }
    reg_path = _registry_path(src_index)

    tmp = dst.with_name(f".{dst.name}.archive-{os.getpid()}-{time.time_ns()}.tmp")
    _safe_unlink(tmp)
    out_members: List[ArchiveMember] = []
    try:
        with _open(tmp, "wb") as out:
            out.write(b"\x00" * _HEADER_SIZE)  # patched after the payload
            offset = _HEADER_SIZE
            for role, gen, fp, fname, path in members:
                before = signatures[str(path)]
                if before is None:
                    raise ArchiveFormatError(
                        f"member {str(path)!r} disappeared while archiving"
                    )
                try:
                    size, sha = _copy_stream(path, out)
                except OSError as exc:
                    raise ArchiveFormatError(
                        f"cannot read member {str(path)!r}: {exc}"
                    ) from exc
                after = _file_signature(path)
                if after != before or size != before[2]:
                    raise ArchiveFormatError(
                        f"{path.name} changed while it was being archived "
                        "(a refresh/rebuild was published concurrently); no "
                        "archive was written"
                    )
                out_members.append(ArchiveMember(
                    role=role, generation=gen, fingerprint=fp,
                    file_name=fname, member_path=_member_name(role, gen),
                    offset=offset, size=size, sha256=sha,
                ))
                offset += size
            manifest_obj: Dict[str, object] = {
                "manifest_version": MANIFEST_VERSION,
                "format": ARCHIVE_FORMAT,
                "archive_version": ARCHIVE_VERSION,
                "index_id": meta["index_id"],
                "name": meta["name"],
                "schema_version": meta["schema_version"],
                "fingerprint_version": meta["fingerprint_version"],
                "created_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "current": {
                    "generation": meta["generation"],
                    "fingerprint": meta["fingerprint"],
                },
                "retained": [
                    {"generation": g, "fingerprint": fp}
                    for g, fp in meta["retained"]
                ],
                "retain_old": meta["retain_old"],
                "members": [m.as_dict() for m in out_members],
            }
            manifest_bytes = json.dumps(
                manifest_obj, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            manifest_offset = out.tell()
            out.write(manifest_bytes)
            manifest_sha = hashlib.sha256(manifest_bytes).digest()
            out.write(_FOOTER.pack(
                manifest_sha, len(manifest_bytes), FOOTER_VERSION, _FOOTER_MAGIC
            ))
            out.flush()
            _fsync_file(tmp)
            out.seek(0)
            out.write(_HEADER.pack(
                _MAGIC, ARCHIVE_VERSION, manifest_offset,
                len(manifest_bytes), 0,
            ))
            out.flush()
            _fsync_file(tmp)

        # Final consistency: the current file and the registry must still be
        # exactly the generation state the manifest describes.
        if _file_signature(src_index) != signatures[str(src_index)]:
            raise ArchiveFormatError(
                "the current index generation changed while archiving; no "
                "archive was written"
            )
        if registry_raw is not None:
            try:
                now_registry = reg_path.read_bytes()
            except OSError:
                now_registry = None
            if now_registry != registry_raw or \
                    _file_signature(reg_path) != signatures[str(reg_path)]:
                raise ArchiveFormatError(
                    "the generation registry changed while archiving; no "
                    "archive was written"
                )
        os.replace(tmp, dst)
        _fsync_dir(dst.parent)
    except BaseException:
        _safe_unlink(tmp)
        raise

    return ArchiveReport(
        path=str(dst),
        index_id=str(meta["index_id"]),
        current_generation=int(meta["generation"]),
        current_fingerprint=str(meta["fingerprint"]),
        generations=[int(meta["generation"])] + [
            int(g) for g, _fp in meta["retained"]
        ],
        members=[m.as_dict() for m in out_members],
        size=dst.stat().st_size,
        elapsed_seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# reading / validating an archive
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _ParsedArchive:
    manifest: ArchiveManifest
    manifest_offset: int
    payload_end: int


def _read_exact(fh, n: int, *, what: str) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ArchiveFormatError(f"truncated archive while reading {what}")
    return data


def _parse_archive(fh) -> _ParsedArchive:
    """Parse and structurally validate the container (no member hashing)."""
    fh.seek(0, os.SEEK_END)
    total = fh.tell()
    if total < _HEADER_SIZE + _FOOTER_SIZE:
        raise ArchiveFormatError("truncated archive")
    fh.seek(0)
    header = _read_exact(fh, _HEADER_SIZE, what="header")
    magic, header_version, manifest_offset, manifest_length, _reserved = \
        _HEADER.unpack(header)
    if magic != _MAGIC:
        raise ArchiveFormatError("not a nearidx archive (bad magic)")
    if header_version != ARCHIVE_VERSION:
        raise ArchiveFormatError(
            f"unsupported archive container version {header_version} "
            f"(expected {ARCHIVE_VERSION})"
        )
    if manifest_length > _MAX_MANIFEST:
        raise ArchiveFormatError("manifest is implausibly large")
    fh.seek(total - _FOOTER_SIZE)
    footer = _read_exact(fh, _FOOTER_SIZE, what="footer")
    manifest_sha, footer_len, footer_version, footer_magic = _FOOTER.unpack(footer)
    if footer_magic != _FOOTER_MAGIC:
        raise ArchiveFormatError("truncated archive (bad footer magic)")
    if footer_version != FOOTER_VERSION:
        raise ArchiveFormatError(
            f"unsupported archive footer version {footer_version}"
        )
    if footer_len != manifest_length:
        raise ArchiveFormatError(
            "archive header and footer disagree about the manifest length"
        )
    expected_manifest_at = total - _FOOTER_SIZE - manifest_length
    if expected_manifest_at < _HEADER_SIZE:
        raise ArchiveFormatError("truncated archive (manifest outside file)")
    if manifest_offset != expected_manifest_at:
        raise ArchiveFormatError(
            "archive header and footer disagree about the manifest position"
        )
    fh.seek(manifest_offset)
    manifest_bytes = _read_exact(fh, manifest_length, what="manifest")
    if hashlib.sha256(manifest_bytes).digest() != manifest_sha:
        raise ArchiveFormatError("archive manifest does not match its checksum")
    try:
        obj = _strict_json(manifest_bytes)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArchiveManifestError(
            f"archive manifest is not valid JSON: {exc}"
        ) from exc
    return _validate_manifest(obj, payload_end=manifest_offset)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ArchiveManifestError(msg)


def _validate_manifest(
    obj: object, *, payload_end: Optional[int]
) -> _ParsedArchive:
    """Validate manifest structure and its relation to the registry state.

    ``payload_end`` is the manifest's file offset for a real archive (member
    offsets must tile ``[_HEADER_SIZE, payload_end)``) or ``None`` when
    re-validating a stored journal.
    """
    _require(isinstance(obj, dict), "manifest must be a JSON object")
    assert isinstance(obj, dict)
    version = obj.get("manifest_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ArchiveManifestError("manifest_version must be an integer")
    if version != MANIFEST_VERSION:
        raise ArchiveManifestError(
            f"unsupported manifest_version {version} (expected "
            f"{MANIFEST_VERSION})"
        )
    _require(obj.get("format") == ARCHIVE_FORMAT,
             f"manifest format must be {ARCHIVE_FORMAT!r}")
    _require(obj.get("archive_version") == ARCHIVE_VERSION,
             "manifest archive_version does not match the container")
    index_id = obj.get("index_id")
    _require(isinstance(index_id, str) and index_id != "",
             "manifest index_id must be a non-empty string")
    name = obj.get("name")
    _require(isinstance(name, str), "manifest name must be a string")
    schema_version = obj.get("schema_version")
    _require(isinstance(schema_version, int)
             and not isinstance(schema_version, bool),
             "manifest schema_version must be an integer")
    if schema_version != SCHEMA_VERSION:
        raise ArchiveManifestError(
            f"unsupported index schema_version {schema_version} "
            f"(expected {SCHEMA_VERSION})"
        )
    fpv = obj.get("fingerprint_version")
    _require(isinstance(fpv, str),
             "manifest fingerprint_version must be a string")
    created = obj.get("created_utc")
    _require(isinstance(created, str), "manifest created_utc must be a string")
    current = obj.get("current")
    _require(isinstance(current, dict), "manifest current must be an object")
    assert isinstance(current, dict)
    cur_gen = current.get("generation")
    cur_fp = current.get("fingerprint")
    _require(isinstance(cur_gen, int) and not isinstance(cur_gen, bool)
             and cur_gen >= 1,
             "manifest current.generation must be a positive integer")
    _require(isinstance(cur_fp, str) and cur_fp != "",
             "manifest current.fingerprint must be a non-empty string")
    retained_raw = obj.get("retained")
    _require(isinstance(retained_raw, list), "manifest retained must be a list")
    assert isinstance(retained_raw, list)
    retained: List[Tuple[int, str]] = []
    seen_gens = {cur_gen}
    for item in retained_raw:
        _require(isinstance(item, dict),
                 "manifest retained entries must be objects")
        assert isinstance(item, dict)
        g = item.get("generation")
        fp = item.get("fingerprint")
        _require(isinstance(g, int) and not isinstance(g, bool) and g >= 1,
                 "manifest retained generation must be a positive integer")
        _require(isinstance(fp, str) and fp != "",
                 "manifest retained fingerprint must be a non-empty string")
        if g in seen_gens:
            raise ArchiveManifestError(
                f"manifest generation {g} is listed more than once"
            )
        if g >= cur_gen:
            raise ArchiveManifestError(
                f"manifest retained generation {g} is not older than the "
                f"current generation {cur_gen}"
            )
        seen_gens.add(g)
        retained.append((g, fp))
    retain_old = obj.get("retain_old")
    _require(isinstance(retain_old, int) and not isinstance(retain_old, bool)
             and retain_old >= 0,
             "manifest retain_old must be a non-negative integer")
    if retained and retain_old < 1:
        raise ArchiveManifestError(
            "manifest retains old generations but records retain_old=0"
        )
    if len(retained) > retain_old:
        raise ArchiveManifestError(
            "manifest retains more generations than its retain_old setting "
            "allows"
        )

    members_raw = obj.get("members")
    _require(isinstance(members_raw, list) and len(members_raw) > 0,
             "manifest must list at least one member")
    assert isinstance(members_raw, list)
    by_role: Dict[str, Dict[int, ArchiveMember]] = {
        ROLE_CURRENT: {}, ROLE_GENERATION: {}, ROLE_REGISTRY: {},
    }
    seen_paths = set()
    offset = _HEADER_SIZE
    ordered: List[ArchiveMember] = []
    for i, item in enumerate(members_raw):
        where = f"manifest member {i + 1}"
        _require(isinstance(item, dict), f"{where} must be an object")
        assert isinstance(item, dict)
        role = item.get("role")
        if role not in _ROLES:
            raise ArchiveManifestError(
                f"{where}: unknown role {role!r} (expected one of {list(_ROLES)!r})"
            )
        gen = item.get("generation")
        if role == ROLE_REGISTRY:
            _require(gen == 0 or gen is None,
                     f"{where}: registry member must have generation 0")
            gen = 0
            fp = item.get("fingerprint", "")
            _require(isinstance(fp, str), f"{where}: fingerprint must be a string")
        else:
            _require(isinstance(gen, int) and not isinstance(gen, bool)
                     and gen >= 1,
                     f"{where}: generation must be a positive integer")
            fp = item.get("fingerprint")
            _require(isinstance(fp, str) and fp != "",
                     f"{where}: fingerprint must be a non-empty string")
        fname = item.get("file_name")
        _require(isinstance(fname, str) and fname != "",
                 f"{where}: file_name must be a non-empty string")
        _safe_file_name(fname, where=f"{where} file_name")
        member_path = _safe_member_path(item.get("path"))
        size = item.get("size")
        _require(isinstance(size, int) and not isinstance(size, bool)
                 and size >= 0,
                 f"{where}: size must be a non-negative integer")
        sha = item.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(
            c not in "0123456789abcdef" for c in sha
        ):
            raise ArchiveManifestError(
                f"{where}: sha256 must be 64 lowercase hex digits"
            )
        member_offset = item.get("offset")
        _require(isinstance(member_offset, int)
                 and not isinstance(member_offset, bool),
                 f"{where}: offset must be an integer")
        if member_offset != offset:
            raise ArchiveManifestError(
                f"{where}: offset {member_offset} does not follow the previous "
                f"member (expected {offset})"
            )
        if member_path in seen_paths:
            raise ArchiveManifestError(
                f"duplicate archive member {member_path!r}"
            )
        seen_paths.add(member_path)
        role_map = by_role[role]
        if role != ROLE_REGISTRY and gen in role_map:
            raise ArchiveManifestError(
                f"{where}: generation {gen} has more than one {role} member"
            )
        if role == ROLE_REGISTRY and role_map:
            raise ArchiveManifestError(
                f"{where}: more than one registry member"
            )
        member = ArchiveMember(
            role=role, generation=gen, fingerprint=fp, file_name=fname,
            member_path=member_path, offset=member_offset, size=size, sha256=sha,
        )
        role_map[gen] = member
        ordered.append(member)
        offset += size

    if payload_end is not None and offset != payload_end:
        raise ArchiveFormatError(
            f"archive payload covers {offset} bytes but the manifest starts at "
            f"{payload_end}; members are missing or extra bytes were inserted"
        )
    if len(by_role[ROLE_CURRENT]) != 1:
        raise ArchiveManifestError(
            "manifest must list exactly one current-generation member"
        )
    cur_member = by_role[ROLE_CURRENT].get(cur_gen)
    if cur_member is None:
        raise ArchiveManifestError(
            "manifest current generation has no current member"
        )
    if cur_member.fingerprint != cur_fp:
        raise ArchiveManifestError(
            "manifest current member fingerprint does not match current entry"
        )
    member_gens = set(by_role[ROLE_GENERATION])
    retained_gens = {g for g, _fp in retained}
    if member_gens != retained_gens:
        raise ArchiveManifestError(
            "manifest retained generations and generation members disagree "
            f"(retained={sorted(retained_gens)!r}, "
            f"members={sorted(member_gens)!r})"
        )
    for g, fp in retained:
        if by_role[ROLE_GENERATION][g].fingerprint != fp:
            raise ArchiveManifestError(
                f"manifest generation {g} member fingerprint does not match "
                "the retained entry"
            )
    # The registry file must be present exactly when old generations are
    # registered, and its own entries must agree with the manifest.
    registry_member = by_role[ROLE_REGISTRY].get(0) if by_role[ROLE_REGISTRY] else None
    if retained and registry_member is None:
        raise ArchiveManifestError(
            "manifest retains old generations but carries no registry file"
        )
    manifest = ArchiveManifest(
        manifest_version=version,
        format=ARCHIVE_FORMAT,
        archive_version=ARCHIVE_VERSION,
        index_id=index_id,
        name=name,
        schema_version=schema_version,
        fingerprint_version=fpv,
        current_generation=cur_gen,
        current_fingerprint=cur_fp,
        retained=tuple(retained),
        retain_old=retain_old,
        created_utc=created,
        members=tuple(ordered),
    )
    return _ParsedArchive(
        manifest=manifest,
        manifest_offset=payload_end if payload_end is not None else 0,
        payload_end=payload_end if payload_end is not None else 0,
    )


def _validate_registry_member(
    path: Path, manifest: ArchiveManifest
) -> None:
    """Check the archived generation registry against the manifest."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ArchiveFormatError(f"cannot read registry member: {exc}") from exc
    try:
        reg = _strict_json(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArchiveFormatError(
            f"registry member is not valid JSON: {exc}"
        ) from exc
    if not isinstance(reg, dict) or reg.get("version") != 1:
        raise ArchiveFormatError("registry member has an unsupported version")
    if reg.get("pending") is not None:
        raise ArchiveFormatError("registry member still has a pending publish")
    if reg.get("index_id") not in (None, manifest.index_id):
        raise ArchiveFormatError(
            "registry member belongs to a different index identity"
        )
    current = reg.get("current")
    if not isinstance(current, dict) or \
            current.get("generation") != manifest.current_generation or \
            current.get("fingerprint") != manifest.current_fingerprint:
        raise ArchiveFormatError(
            "registry member's current generation does not match the manifest"
        )
    keep = reg.get("retain_old")
    if isinstance(keep, bool) or not isinstance(keep, int) or keep != manifest.retain_old:
        raise ArchiveFormatError(
            "registry member's retain_old does not match the manifest"
        )
    entries = reg.get("retained")
    if not isinstance(entries, list):
        raise ArchiveFormatError("registry member has no retained list")
    got: List[Tuple[int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ArchiveFormatError("registry member: malformed retained entry")
        g = entry.get("generation")
        fp = entry.get("fingerprint")
        if not isinstance(g, int) or isinstance(g, bool) or \
                not isinstance(fp, str) or not fp:
            raise ArchiveFormatError(
                "registry member: retained entry missing generation/fingerprint"
            )
        got.append((g, fp))
    if got != list(manifest.retained):
        raise ArchiveFormatError(
            "registry member's retained generations do not match the manifest"
        )
    # The registry's file names must be exactly the archived members' original
    # names, so restore can map them onto the new target's names.
    current_member = manifest.member_for(ROLE_CURRENT)
    assert current_member is not None
    for entry in entries:
        g = entry["generation"]
        fname = entry.get("file")
        member = manifest.member_for(ROLE_GENERATION, g)
        assert member is not None
        if isinstance(fname, str) and fname and fname != member.file_name:
            raise ArchiveFormatError(
                f"registry member names generation {g} file {fname!r} but the "
                f"archive carries {member.file_name!r}"
            )


def inspect_archive(
    archive: os.PathLike | str, *, verify: bool = True
) -> ArchiveManifest:
    """Read an archive's manifest, optionally hashing every member payload.

    With ``verify=True`` (the default) the payload is streamed once and every
    member's length and SHA-256 is checked, so the returned manifest is known
    to describe exactly the bytes in the container.  Nothing is written and
    no member is opened as SQLite.
    """
    path = Path(archive)
    if not path.is_file():
        raise FileNotFoundError(path)
    _reject_symlink(path, what="archive")
    with _open(path, "rb") as fh:
        parsed = _parse_archive(fh)
        if verify:
            _verify_payload(fh, parsed)
    return parsed.manifest


def _verify_payload(fh, parsed: _ParsedArchive) -> None:
    """Stream the payload region, checking each member's size and SHA-256."""
    for member in parsed.manifest.members:
        fh.seek(member.offset)
        h = hashlib.sha256()
        remaining = member.size
        while remaining:
            chunk = fh.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                raise ArchiveFormatError(
                    f"archive member {member.member_path!r} is truncated"
                )
            h.update(chunk)
            remaining -= len(chunk)
        if h.hexdigest() != member.sha256:
            raise ArchiveFormatError(
                f"archive member {member.member_path!r} sha256 mismatch "
                "(corrupt, truncated or substituted payload)"
            )


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------
#
# Install protocol inside the target directory:
#
#   .<name>.nearidx-restore             marker/journal (full manifest copy)
#   .<name>.nearidx-restore-<pid>-<ns>/ private staging directory
#
# 1. validate container + manifest, hash every member, extract to staging
# 2. SQLite-verify every generation from the staged bytes
# 3. write the marker
# 4. rename generation files into place (not openable as this index yet)
# 5. atomically write the new registry (mapped file names)
# 6. rename the current generation into place -- only now does `INDEX` exist
# 7. drop marker and staging
#
# A crash at 1-3 leaves no visible files.  A crash at 4-5 leaves generation
# files and possibly a registry but no current file, so nothing is openable
# and the marker tells the next restore to discard the partial state.  A
# crash after 6 leaves a complete index; recovery only clears the marker.

def _restore_marker_path(target: Path) -> Path:
    return target.with_name(f".{target.name}{_INSTALL_MARKER_SUFFIX}")


def _restore_staging_path(target: Path) -> Path:
    return target.with_name(
        f".{target.name}{_INSTALL_MARKER_SUFFIX}-{os.getpid()}-{time.time_ns()}"
    )


def _write_marker(target: Path, *, staging: Path, manifest: ArchiveManifest) -> None:
    obj = {
        "version": 1,
        "target": target.name,
        "staging": str(staging),
        "manifest": manifest.as_dict(),
    }
    marker = _restore_marker_path(target)
    tmp = marker.with_name(f".{marker.name}.{os.getpid()}-{time.time_ns()}.tmp")
    tmp.write_bytes(json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))
    _fsync_file(tmp)
    os.replace(tmp, marker)
    _fsync_dir(target.parent)


def _restored_registry_obj(
    target: Path, manifest: ArchiveManifest
) -> Dict[str, object]:
    """Registry for the restored index: old file names mapped to new ones."""
    return {
        "version": 1,
        "index_id": manifest.index_id,
        "retain_old": manifest.retain_old,
        "current": {
            "generation": manifest.current_generation,
            "fingerprint": manifest.current_fingerprint,
        },
        "retained": [
            {
                "generation": g,
                "fingerprint": fp,
                "file": _gen_file_name(target, g),
            }
            for g, fp in manifest.retained
        ],
        "pending": None,
    }


def _validate_member_database(
    path: Path, member: ArchiveMember, manifest: ArchiveManifest, *, label: str
) -> None:
    """Open one extracted member as SQLite and verify its binding.

    Checks schema, ``index_id``, generation, fingerprint, SQLite integrity and
    the full text/posting binding.  Source paths stored with documents are not
    rewritten and source files are not required to exist: an archive does not
    carry external sources, and restore never edits those columns.
    """
    _reject_symlink(path, what=label)
    conn: Optional[sqlite3.Connection] = None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        conn.execute("PRAGMA query_only = ON")
        meta = _meta(conn)
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            raise ArchiveFormatError(
                f"{label}: unsupported schema_version "
                f"{meta.get('schema_version')!r}"
            )
        if meta.get("index_id") != manifest.index_id:
            raise ArchiveFormatError(
                f"{label}: index_id {meta.get('index_id')!r} does not match "
                f"the archive's {manifest.index_id!r}"
            )
        if meta.get("generation") != str(member.generation):
            raise ArchiveFormatError(
                f"{label}: file reports generation {meta.get('generation')!r}, "
                f"manifest says {member.generation}"
            )
        if meta.get("index_fingerprint") != member.fingerprint:
            raise ArchiveFormatError(
                f"{label}: file fingerprint does not match the manifest"
            )
        quick = conn.execute("PRAGMA quick_check").fetchone()
        if not quick or quick[0] != "ok":
            raise ArchiveFormatError(
                f"{label}: SQLite integrity check failed "
                f"({quick[0] if quick else 'no result'})"
            )
        rows = conn.execute(
            "SELECT doc_id, text, text_sha256, char_len, byte_len, token_count, "
            "postings_digest FROM documents ORDER BY doc_id"
        ).fetchall()
        if meta.get("doc_count") != str(len(rows)):
            raise ArchiveFormatError(
                f"{label}: meta doc_count does not match the documents table"
            )
        fp_rows: List[Tuple[str, str, str]] = []
        for doc_id, text, sha, clen, blen, tcount, pdigest in rows:
            if not isinstance(text, str):
                raise ArchiveFormatError(f"{label}: {doc_id}: text is not a string")
            if text_sha256(text) != sha:
                raise ArchiveFormatError(
                    f"{label}: {doc_id}: stored text does not match sha256"
                )
            if len(text) != clen or len(text.encode("utf-8")) != blen:
                raise ArchiveFormatError(
                    f"{label}: {doc_id}: stored length metadata mismatch"
                )
            toks = token_spans(text)
            if len(toks) != tcount:
                raise ArchiveFormatError(
                    f"{label}: {doc_id}: token_count mismatch"
                )
            digest = _postings_digest(
                (t.pos, t.norm, t.char_start, t.char_end, t.byte_start,
                 t.byte_end) for t in toks
            )
            if digest != pdigest:
                raise ArchiveFormatError(
                    f"{label}: {doc_id}: postings digest mismatch"
                )
            stored = conn.execute(
                "SELECT pos, token, char_start, char_end, byte_start, byte_end "
                "FROM postings WHERE doc_id = ? ORDER BY pos", (doc_id,)
            ).fetchall()
            recomputed = [
                (t.pos, t.norm, t.char_start, t.char_end, t.byte_start,
                 t.byte_end) for t in toks
            ]
            if [tuple(r) for r in stored] != recomputed:
                raise ArchiveFormatError(
                    f"{label}: {doc_id}: posting rows differ from stored text"
                )
            fp_rows.append((doc_id, sha, pdigest))
        if _fingerprint(fp_rows) != meta.get("index_fingerprint"):
            raise ArchiveFormatError(
                f"{label}: index fingerprint does not match its document set"
            )
    except sqlite3.DatabaseError as exc:
        raise ArchiveFormatError(
            f"{label}: not a readable SQLite index: {exc}"
        ) from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def _extract_payload(fh, parsed: _ParsedArchive, staging: Path) -> Dict[str, Path]:
    """Stream every member out of the archive into *staging*, hashing again.

    Hashing the bytes a second time while writing them is what makes the
    install consume verified bytes: a change between inspection and
    extraction (TOCTOU) is caught before anything is installed.
    """
    extracted: Dict[str, Path] = {}
    for member in parsed.manifest.members:
        fh.seek(member.offset)
        dest = staging / member.member_path
        h = hashlib.sha256()
        remaining = member.size
        with _open(dest, "wb") as out:
            while remaining:
                chunk = fh.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise ArchiveFormatError(
                        f"archive member {member.member_path!r} is truncated"
                    )
                out.write(chunk)
                h.update(chunk)
                remaining -= len(chunk)
            out.flush()
            _fsync_file(dest)
        if h.hexdigest() != member.sha256:
            raise ArchiveFormatError(
                f"archive member {member.member_path!r} sha256 mismatch during "
                "extraction"
            )
        extracted[member.member_path] = dest
    return extracted


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except OSError:
        pass


def _remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        _safe_unlink(path)
        return
    try:
        for child in path.iterdir():
            if child.is_dir() and not child.is_symlink():
                _remove_tree(child)
            else:
                _safe_unlink(child)
        path.rmdir()
    except OSError:
        pass


def _install_generations(
    target: Path, staging: Path, manifest: ArchiveManifest
) -> None:
    for member in manifest.members:
        if member.role != ROLE_GENERATION:
            continue
        dest = target.with_name(_gen_file_name(target, member.generation))
        _safe_unlink(dest)
        os.replace(staging / member.member_path, dest)
    _fsync_dir(target.parent)


def _is_complete_install(target: Path, manifest: ArchiveManifest) -> bool:
    """True when *target* already holds exactly the manifest's generations."""
    if not target.is_file() or _is_symlink(target):
        return False
    meta = _read_main_meta(target)
    if meta is None or "__corrupt__" in meta:
        return False
    if (
        meta.get("index_id") != manifest.index_id
        or meta.get("generation") != str(manifest.current_generation)
        or meta.get("index_fingerprint") != manifest.current_fingerprint
    ):
        return False
    for g, fp in manifest.retained:
        gpath = target.with_name(_gen_file_name(target, g))
        gmeta = _read_main_meta(gpath)
        if gmeta is None or "__corrupt__" in gmeta:
            return False
        if gmeta.get("index_fingerprint") != fp or \
                gmeta.get("generation") != str(g) or \
                gmeta.get("index_id") != manifest.index_id:
            return False
        member = manifest.member_for(ROLE_GENERATION, g)
        assert member is not None
        try:
            if gpath.stat().st_size != member.size:
                return False
        except OSError:
            return False
    return True


def _roll_forward(target: Path, marker_obj: Dict[str, object]) -> bool:
    """Complete an install whose current file is already in place.

    Returns True when the target is a complete restored index afterwards.
    The staging directory is consulted as a fallback source for generation
    files that were still being moved when the install was interrupted.
    """
    try:
        manifest = _validate_manifest(
            marker_obj.get("manifest"), payload_end=None
        ).manifest
    except ArchiveFormatError:
        return False
    if not target.is_file() or _is_symlink(target):
        return False
    staging_raw = marker_obj.get("staging")
    staging = Path(staging_raw) if isinstance(staging_raw, str) else None
    if not _is_complete_install(target, manifest):
        if staging is None:
            return False
        for member in manifest.members:
            if member.role != ROLE_GENERATION:
                continue
            dest = target.with_name(_gen_file_name(target, member.generation))
            if dest.is_file() and not _is_symlink(dest):
                continue
            staged = staging / member.member_path
            if not staged.is_file():
                return False
            os.replace(staged, dest)
        _fsync_dir(target.parent)
        if not _is_complete_install(target, manifest):
            return False
    _write_registry(target, _restored_registry_obj(target, manifest))
    _fsync_dir(target.parent)
    return True


def _recover_interrupted_install(target: Path) -> None:
    """Resolve a marker left by an interrupted restore of *target*.

    Roll-forward if the complete current generation is already installed and
    every retained generation matches; otherwise delete the partial artifacts
    and the marker, leaving nothing openable at the target path.  No-op when
    no marker exists.
    """
    marker = _restore_marker_path(target)
    if not marker.exists() and not marker.is_symlink():
        return
    _reject_symlink(marker, what="restore marker")
    try:
        obj = _strict_json(marker.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError):
        obj = None
    staging = None
    if isinstance(obj, dict) and isinstance(obj.get("staging"), str):
        staging = Path(obj["staging"])
    rolled = False
    if isinstance(obj, dict) and obj.get("version") == 1:
        rolled = _roll_forward(target, obj)
    if not rolled:
        if target.is_file() and not _is_symlink(target):
            # A half current file (or a current file from a mismatched
            # journal) is never a usable restored index.
            _safe_unlink(target)
        if isinstance(obj, dict) and isinstance(obj.get("manifest"), dict):
            members = obj["manifest"].get("members")
            if isinstance(members, list):
                for m in members:
                    if not isinstance(m, dict):
                        continue
                    if m.get("role") != ROLE_GENERATION:
                        continue
                    g = m.get("generation")
                    if isinstance(g, int) and not isinstance(g, bool):
                        _safe_unlink(
                            target.with_name(_gen_file_name(target, g))
                        )
        _safe_unlink(_registry_path(target))
    if staging is not None:
        _remove_tree(staging)
    _safe_unlink(marker)
    _fsync_dir(target.parent)


def _check_destination_clear(target: Path, manifest: ArchiveManifest) -> None:
    """Refuse to install over artifacts restore did not create itself."""
    if target.exists() or target.is_symlink():
        raise ArchiveExistsError(
            f"{target} already exists; restore only creates a new index"
        )
    for g in manifest.generation_fingerprints():
        dest = target.with_name(_gen_file_name(target, g))
        if dest.exists() or dest.is_symlink():
            raise ArchiveExistsError(
                f"{dest} already exists; restore only creates a new index"
            )
    reg = _registry_path(target)
    if reg.exists() or reg.is_symlink():
        raise ArchiveExistsError(
            f"{reg} already exists; restore only creates a new index"
        )
    # A leftover staging directory from an unrelated crashed restore would
    # make the install ambiguous; refuse rather than mixing two installs.
    # (Marker-based recovery above handles the ones that did journal.)


def restore_index(
    archive: os.PathLike | str, index: os.PathLike | str
) -> RestoreReport:
    """Restore *archive* into the new index path *index*.

    The destination (and its registry/generation files) must not exist.  The
    validation order is: container, manifest version and member list, payload
    length/SHA-256 of every member, the archived registry, then a full
    SQLite-level check of every generation against the manifest
    (``index_id``, generation, fingerprint, text/posting binding).  Only
    after all of that are files installed.  Original file names are mapped
    onto the new target's file names while identity, generation numbers and
    fingerprints are preserved exactly, so pre-archive single-index and
    federated cursors stay valid.
    """
    started = time.monotonic()
    archive_path = Path(archive)
    target = Path(index)

    # Finish or discard a previous interrupted restore before deciding the
    # destination state (a completed restore makes the target exist).
    _recover_interrupted_install(target)
    if target.exists() or target.is_symlink():
        raise ArchiveExistsError(
            f"{target} already exists; restore only creates a new index"
        )
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    _reject_symlink(archive_path, what="archive")
    target.parent.mkdir(parents=True, exist_ok=True)

    staging = _restore_staging_path(target)
    staging.mkdir(parents=True, exist_ok=False)
    published = False
    installing = False
    try:
        with _open(archive_path, "rb") as fh:
            parsed = _parse_archive(fh)
            _verify_payload(fh, parsed)
            manifest = parsed.manifest
            _check_destination_clear(target, manifest)
            extracted = _extract_payload(fh, parsed, staging)
        # Registry-vs-manifest cross-check from the member bytes.
        registry_member = manifest.member_for(ROLE_REGISTRY)
        if registry_member is not None:
            _validate_registry_member(
                extracted[registry_member.member_path], manifest
            )
        # SQLite-level verification of every generation, before install.
        for member in manifest.members:
            if member.role == ROLE_REGISTRY:
                continue
            label = (
                f"generation {member.generation}"
                if member.role == ROLE_GENERATION
                else "current generation"
            )
            _validate_member_database(
                extracted[member.member_path], member, manifest, label=label
            )
        # Journal the install before anything visible appears.  The marker
        # is also the install's own record of what it may write, so a failure
        # can withdraw exactly those artifacts.
        _write_marker(target, staging=staging, manifest=manifest)
        installing = True
        _install_generations(target, staging, manifest)
        # The registry is written whenever old generations were installed;
        # archives of a never-refreshed index carry no registry member and
        # restore keeps that state (no registry file).
        if manifest.retained:
            _write_registry(target, _restored_registry_obj(target, manifest))
        current_member = manifest.member_for(ROLE_CURRENT)
        assert current_member is not None
        # Current generation last: the target path appears complete.
        os.replace(extracted[current_member.member_path], target)
        published = True
        _fsync_dir(target.parent)
        _remove_tree(staging)
        _safe_unlink(_restore_marker_path(target))
        _fsync_dir(target.parent)
    except BaseException:
        if not published:
            if installing:
                # The marker records exactly what this install may have
                # written; recovery removes those artifacts only, never
                # anything that pre-dated the restore.
                _recover_interrupted_install(target)
            _remove_tree(staging)
            _safe_unlink(_restore_marker_path(target))
        raise

    return RestoreReport(
        path=str(target),
        archive=str(archive_path),
        index_id=manifest.index_id,
        current_generation=manifest.current_generation,
        current_fingerprint=manifest.current_fingerprint,
        generations=[manifest.current_generation]
        + [g for g, _fp in manifest.retained],
        registry_path=str(_registry_path(target)) if manifest.has_registry else None,
        members=[m.as_dict() for m in manifest.members],
        elapsed_seconds=time.monotonic() - started,
    )


def recover_restore(index: os.PathLike | str) -> bool:
    """Resolve an interrupted restore of *index*; return whether one existed.

    Opening an index never consults a restore journal, so this is provided
    for tooling that wants to clean up a killed restore before (or instead
    of) retrying :func:`restore_index`.
    """
    target = Path(index)
    existed = _restore_marker_path(target).exists()
    _recover_interrupted_install(target)
    return existed
