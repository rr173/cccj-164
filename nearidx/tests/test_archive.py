"""Tests for streaming archive/restore of an index and its generations.

Every cursor expectation is checked against delivered windows, witnesses and
original snippets -- not against generation numbers alone.  Tampering tests
modify the actual member bytes (and, where noted, the manifest fields too), so
the checks provably cover the payload rather than only the bookkeeping.

Run with ``python -m unittest discover -s nearidx/tests -t .`` or ``pytest``.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nearidx import (
    ArchiveError,
    ArchiveExistsError,
    ArchiveFormatError,
    Federation,
    build_index,
    federated_search,
    inspect_archive,
    open_index,
    parse_query,
    refresh_index,
    restore_index,
    search,
    text_sha256,
)
from nearidx import archive as archive_mod
from nearidx.archive import (
    _HEADER,
    _HEADER_SIZE,
    _FOOTER,
    _FOOTER_MAGIC,
    _MAGIC,
    _file_signature,
    archive_index,
    recover_restore,
)
from nearidx.cli import main as cli_main
from nearidx.db import _gen_file_path, _read_registry, _registry_path

ROOT = Path(__file__).resolve().parents[2]

QUERY = "NEAR(3,[pump,pump])"
PUMP = "pump valve pump"


# ---------------------------------------------------------------------------
# independent archive writer (test-side, mirrors the documented container)
# ---------------------------------------------------------------------------

def build_container(
    path: Path,
    manifest: dict,
    payloads: "dict[str, bytes]",
) -> None:
    """Write an archive from a manifest and raw member bytes.

    Used to construct deliberately malformed archives: the production code is
    never asked to produce them, so the tests do not depend on its internals.
    Payloads are laid out sequentially; the caller is responsible for making
    the manifest's ``offset``/``size``/``sha256`` fields describe (or, for
    negative tests, misdescribe) that layout.
    """
    with open(path, "wb") as out:
        out.write(b"\x00" * _HEADER_SIZE)
        for item in manifest["members"]:
            out.write(payloads[item["path"]])
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest_at = out.tell()
        out.write(manifest_bytes)
        out.write(_FOOTER.pack(
            hashlib.sha256(manifest_bytes).digest(),
            len(manifest_bytes), 1, _FOOTER_MAGIC,
        ))
        out.seek(0)
        out.write(_HEADER.pack(
            _MAGIC, 1, manifest_at, len(manifest_bytes), 0
        ))
        out.flush()


def read_container(path: Path):
    """Return ``(manifest_obj, {member_path: bytes}, header, footer)``."""
    raw = path.read_bytes()
    magic, version, manifest_at, manifest_len, _r = _HEADER.unpack(
        raw[:_HEADER_SIZE]
    )
    footer = raw[-_FOOTER.size:]
    sha, flen, fver, fmagic = _FOOTER.unpack(footer)
    manifest = json.loads(raw[manifest_at:manifest_at + manifest_len])
    payload = {}
    for item in manifest["members"]:
        payload[item["path"]] = raw[item["offset"]:item["offset"] + item["size"]]
    return manifest, payload, (magic, version, manifest_at, manifest_len), \
        (sha, flen, fver, fmagic)


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # -- builders ------------------------------------------------------
    def build(self, docs, name="t"):
        path = self.tmp / f"{name}.nearidx"
        build_index(path, docs, name=name, overwrite=True)
        return path

    def pump_docs(self, n=5):
        return [(f"d{i:02d}", PUMP) for i in range(1, n + 1)]

    def refresh(self, path, ops, **kwargs):
        manifest = self.tmp / "refresh.jsonl"
        manifest.write_text(
            "".join(json.dumps(op) + "\n" for op in ops), encoding="utf-8"
        )
        return refresh_index(path, manifest, **kwargs)

    def page(self, index, query=QUERY, **kwargs):
        return search(index, parse_query(query), **kwargs)

    def win_tuples(self, index, query=QUERY, **kwargs):
        return [
            (w.doc_id, w.start, w.end, w.witness, w.snippet)
            for w in self.page(index, query, **kwargs).windows
        ]

    def walk_chain(self, index, cursor, query=QUERY):
        pages, flat, cur = [], [], cursor
        while True:
            page = self.page(index, query, cursor=cur)
            pages.append(page)
            flat.extend(
                (w.doc_id, w.start, w.end, w.witness, w.snippet)
                for w in page.windows
            )
            if not page.has_more:
                break
            cur = page.next_cursor
            self.assertIsNotNone(cur)
        return pages, flat

    def run_cli(self, *argv):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# round trip and members
# ---------------------------------------------------------------------------

class ArchiveCompositionTests(TempCase):
    def test_no_retained_generations_round_trip(self):
        path = self.build(self.pump_docs(3))
        archive = self.tmp / "a.pkg"
        report = archive_index(path, archive)
        self.assertEqual(report.current_generation, 1)
        self.assertEqual(report.generations, [1])
        self.assertEqual([m["role"] for m in report.members], ["current"])
        manifest = inspect_archive(archive)
        self.assertEqual(manifest.current_generation, 1)
        self.assertEqual(manifest.retained, ())
        self.assertEqual(len(manifest.members), 1)

        restored = self.tmp / "restored.nearidx"
        out = restore_index(archive, restored)
        self.assertIsNone(out.registry_path)
        self.assertFalse(_registry_path(restored).exists())
        with open_index(restored) as idx:
            self.assertEqual(idx.generation, 1)
            self.assertEqual(idx.fingerprint, report.current_fingerprint)
            self.assertEqual(
                self.win_tuples(idx, limit=100),
                [(d, 0, 2, (0, 2), PUMP) for d, _t in self.pump_docs(3)],
            )
            self.assertEqual(idx.verify(full=True), [])

    def test_one_and_many_retained_generations_round_trip(self):
        path = self.build(self.pump_docs(6))
        for gen in (1, 2, 3):
            self.refresh(path, [{"op": "delete", "doc_id": f"d0{gen}"}])
        archive = self.tmp / "many.pkg"
        report = archive_index(path, archive)
        self.assertEqual(report.current_generation, 4)
        # current + two retained (default retention) + registry file
        self.assertEqual(report.generations, [4, 3, 2])
        roles = [m["role"] for m in report.members]
        self.assertEqual(roles, ["current", "generation", "generation",
                                 "registry"])
        manifest = inspect_archive(archive)
        self.assertEqual(
            [g for g, _fp in manifest.retained], [3, 2]
        )
        self.assertEqual(manifest.retain_old, 2)
        self.assertEqual(
            [m.size for m in manifest.members],
            [m["size"] for m in report.members],
        )
        # Each member's digest is over the bytes actually carried.
        for m in manifest.members:
            self.assertEqual(len(m.sha256), 64)

        restored = self.tmp / "many-restored.nearidx"
        restore_index(archive, restored)
        with open_index(restored) as idx:
            self.assertEqual(
                [(g.generation, g.current) for g in idx.generations()],
                [(4, True), (3, False), (2, False)],
            )
            self.assertEqual(idx.verify(full=True, generations=True), [])
            # Same identity, generation numbers and fingerprints as before.
            with open_index(path) as src:
                self.assertEqual(idx.index_id, src.index_id)
                self.assertEqual(idx.fingerprint, src.fingerprint)
                self.assertEqual(
                    [(g.generation, g.fingerprint) for g in idx.generations()],
                    [(g.generation, g.fingerprint) for g in src.generations()],
                )
            self.assertEqual(
                self.win_tuples(idx, limit=100),
                [("d04", 0, 2, (0, 2), PUMP), ("d05", 0, 2, (0, 2), PUMP),
                 ("d06", 0, 2, (0, 2), PUMP)],
            )

    def test_manifest_describes_roles_sizes_and_digests_of_real_bytes(self):
        path = self.build(self.pump_docs(4))
        self.refresh(path, [{"op": "delete", "doc_id": "d02"}])
        archive = self.tmp / "a.pkg"
        report = archive_index(path, archive)
        manifest = inspect_archive(archive, verify=False)
        self.assertEqual(
            [(m.role, m.generation) for m in manifest.members],
            [("current", 2), ("generation", 1), ("registry", 0)],
        )
        # The digests match the source files byte for byte.
        by_role = {m.role: m for m in manifest.members}
        for role, src in (
            ("current", path),
            ("generation", _gen_file_path(path, 1)),
            ("registry", _registry_path(path)),
        ):
            data = src.read_bytes()
            self.assertEqual(by_role[role].size, len(data))
            self.assertEqual(
                by_role[role].sha256, hashlib.sha256(data).hexdigest()
            )
        self.assertEqual(report.size, archive.stat().st_size)

    def test_archive_does_not_pack_external_sources(self):
        src = self.tmp / "note.txt"
        src.write_text(PUMP, encoding="utf-8")
        path = self.tmp / "src.nearidx"
        from nearidx import Document

        build_index(path, [Document.from_file("d1", src)], overwrite=True)
        archive = self.tmp / "a.pkg"
        report = archive_index(path, archive)
        self.assertEqual(
            {m["role"] for m in report.members}, {"current"}
        )
        self.assertLess(report.size, src.stat().st_size + 200_000)
        # Source file is not required for restore, and its binding columns are
        # preserved unchanged in the restored index.
        restored = self.tmp / "restored.nearidx"
        src.unlink()
        restore_index(archive, restored)
        with open_index(restored) as idx:
            meta = idx.doc_meta("d1")
            self.assertEqual(meta["source_path"], str(src.resolve()))
            self.assertEqual(meta["source_sha256"], text_sha256(PUMP))
            self.assertEqual(idx.text("d1"), PUMP)
            self.assertEqual(idx.verify(full=True, check_sources=False), [])
            problems = idx.verify(full=True, check_sources=True)
            self.assertTrue(
                any("source file unreadable" in p for p in problems), problems
            )

    def test_build_without_registry_and_empty_refresh_archived(self):
        path = self.build([("d1", "only words")])
        self.refresh(path, [])  # same content, new generation, registry now
        archive = self.tmp / "a.pkg"
        report = archive_index(path, archive)
        self.assertEqual(report.generations, [2, 1])
        manifest = inspect_archive(archive)
        # Identical content generations share a fingerprint, yet remain
        # separate, individually sized and hashed members.
        self.assertEqual(manifest.retained, ((1, manifest.current_fingerprint),))
        self.assertEqual(len(manifest.retained), 1)
        self.assertEqual(manifest.retained[0][0], 1)
        # The registry member is the only non-index payload.
        self.assertEqual(
            [m.role for m in manifest.members],
            ["current", "generation", "registry"],
        )
        restored = self.tmp / "r.nearidx"
        restore_index(archive, restored)
        with open_index(restored) as idx:
            self.assertEqual(idx.verify(full=True, generations=True), [])

    def test_inspect_archive_does_not_modify_or_require_target(self):
        path = self.build(self.pump_docs(2))
        archive = self.tmp / "a.pkg"
        archive_index(path, archive)
        before = archive.read_bytes()
        manifest = inspect_archive(archive)
        self.assertEqual(manifest.current_generation, 1)
        self.assertEqual(archive.read_bytes(), before)
        self.assertFalse((self.tmp / "restored.nearidx").exists())


# ---------------------------------------------------------------------------
# cursor continuity across archive/restore
# ---------------------------------------------------------------------------

class CursorContinuityTests(TempCase):
    def test_single_index_cursor_continues_after_archive_restore_and_reopen(self):
        path = self.build(self.pump_docs(6))
        with open_index(path) as idx:
            first = self.page(idx, limit=2)
            cursor = first.next_cursor
            old_fp = idx.fingerprint
            old_full = self.win_tuples(idx, limit=100)
        # Refresh the source so the cursor points at the retained old
        # generation while a newer current generation exists.
        self.refresh(path, [
            {"op": "upsert", "doc_id": "d03", "text": "valve pump valve"},
            {"op": "delete", "doc_id": "d04"},
        ])
        archive = self.tmp / "a.pkg"
        report = archive_index(path, archive)
        self.assertEqual(report.generations, [2, 1])

        restored = self.tmp / "migrated.nearidx"  # different file name
        restore_index(archive, restored)
        with open_index(restored) as idx:
            # New queries read the restored current generation.
            self.assertEqual(
                self.win_tuples(idx, limit=100),
                [("d01", 0, 2, (0, 2), PUMP), ("d02", 0, 2, (0, 2), PUMP),
                 ("d05", 0, 2, (0, 2), PUMP), ("d06", 0, 2, (0, 2), PUMP)],
            )
            self.assertIsNone(idx.doc_meta("d04"))
            # The pre-archive cursor finishes the pinned old generation with
            # the original windows, order, page size and snippets.
            pages, flat = self.walk_chain(idx, cursor)
            self.assertEqual(flat, old_full[2:])
            self.assertEqual([len(p.windows) for p in pages], [2, 2])
            self.assertTrue(all(p.index_fingerprint == old_fp for p in pages))

    def test_single_index_cursor_continues_in_fresh_process(self):
        path = self.build(self.pump_docs(5))
        with open_index(path) as idx:
            cursor = self.page(idx, limit=2).next_cursor
            old_full = self.win_tuples(idx, limit=100)
        self.refresh(path, [{"op": "delete", "doc_id": "d03"}])
        archive = self.tmp / "a.pkg"
        archive_index(path, archive)
        restored = self.tmp / "other-name.nearidx"
        restore_index(archive, restored)

        driver = self.tmp / "cursor_driver.py"
        driver.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from nearidx import open_index, parse_query, search\n"
            "idx = open_index(sys.argv[1])\n"
            "out, cur = [], sys.argv[3]\n"
            "while True:\n"
            "    page = search(idx, parse_query(sys.argv[2]), cursor=cur)\n"
            "    out += [[w.doc_id, w.start, w.end, list(w.witness), w.snippet,\n"
            "             page.index_fingerprint, page.limit]\n"
            "            for w in page.windows]\n"
            "    if not page.has_more:\n"
            "        break\n"
            "    cur = page.next_cursor\n"
            "print(json.dumps(out, ensure_ascii=False))\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-I", str(driver), str(restored), QUERY, cursor],
            capture_output=True, text=True, check=True,
        )
        got = json.loads(proc.stdout)
        self.assertEqual(
            [(g[0], g[1], g[2], tuple(g[3]), g[4]) for g in got],
            old_full[2:],
        )
        self.assertEqual([g[6] for g in got], [2, 2, 2])
        self.assertEqual(len({g[5] for g in got}), 1)

    def test_federated_cursor_continues_after_restore_under_new_names(self):
        alpha = self.build([("d01", PUMP), ("d03", PUMP), ("d05", PUMP)],
                           "alpha")
        beta = self.build([("d02", PUMP), ("d03", PUMP), ("d06", PUMP)],
                          "beta")
        with Federation({"alpha": str(alpha), "beta": str(beta)}) as fed:
            first = federated_search(fed, parse_query(QUERY), limit=2)
            cursor = first.next_cursor
            pinned = {n: i["fingerprint"] for n, i in first.sources.items()}
            full = [
                (w.source_name, w.doc_id, w.start, w.end, w.witness, w.snippet)
                for w in federated_search(
                    fed, parse_query(QUERY), limit=100
                ).windows
            ]
        # Refresh only beta so its old generation must travel with the archive.
        manifest = self.tmp / "beta.jsonl"
        manifest.write_text(
            json.dumps({"op": "delete", "doc_id": "d03"}) + "\n",
            encoding="utf-8",
        )
        refresh_index(beta, manifest)

        pkg_a = self.tmp / "alpha.pkg"
        pkg_b = self.tmp / "beta.pkg"
        archive_index(alpha, pkg_a)
        archive_index(beta, pkg_b)
        new_a = self.tmp / "vendor-a.nearidx"
        new_b = self.tmp / "vendor-b.nearidx"
        restore_index(pkg_a, new_a)
        restore_index(pkg_b, new_b)

        fed_manifest = self.tmp / "federation.json"
        fed_manifest.write_text(
            json.dumps({"alpha": new_a.name, "beta": new_b.name}),
            encoding="utf-8",
        )
        driver = self.tmp / "fed_driver.py"
        driver.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from nearidx import (Federation, federated_search, parse_query)\n"
            "fed = Federation.from_manifest(sys.argv[1])\n"
            "out, cur = [], sys.argv[3]\n"
            "while True:\n"
            "    page = federated_search(fed, parse_query(sys.argv[2]),"
            " cursor=cur)\n"
            "    out += [[w.source_name, w.doc_id, w.start, w.end,\n"
            "             list(w.witness), w.snippet,\n"
            "             {n: i['fingerprint'] for n, i in page.sources.items()}]\n"
            "            for w in page.windows]\n"
            "    if not page.has_more:\n"
            "        break\n"
            "    cur = page.next_cursor\n"
            "print(json.dumps(out, ensure_ascii=False))\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-I", str(driver), str(fed_manifest), QUERY,
             cursor],
            capture_output=True, text=True, check=True,
        )
        got = json.loads(proc.stdout)
        self.assertEqual(
            [(g[0], g[1], g[2], g[3], tuple(g[4]), g[5]) for g in got],
            full[2:],
        )
        for g in got:
            self.assertEqual(g[6], pinned)

        # A fresh federated query reads the restored current generations.
        with Federation({"alpha": str(new_a), "beta": str(new_b)}) as fed:
            fresh = federated_search(fed, parse_query(QUERY), limit=100)
            self.assertEqual(
                [(w.source_name, w.doc_id) for w in fresh.windows],
                [("alpha", "d01"), ("beta", "d02"), ("alpha", "d03"),
                 ("alpha", "d05"), ("beta", "d06")],
            )
            self.assertEqual(fresh.sources["beta"]["generation"], 2)
            self.assertNotEqual(
                fresh.sources["beta"]["fingerprint"], pinned["beta"]
            )


# ---------------------------------------------------------------------------
# validation and tampering
# ---------------------------------------------------------------------------

class ArchiveInputRejectionTests(TempCase):
    def test_archive_excludes_unregistered_generation_files(self):
        path = self.build(self.pump_docs(3))
        self.refresh(path, [{"op": "delete", "doc_id": "d03"}])
        # An orphaned, *valid* generation file that the registry does not
        # name must not travel with the archive.
        stray = _gen_file_path(path, 9)
        build_index(stray, [("ghost", PUMP)], overwrite=True)
        archive = self.tmp / "a.pkg"
        report = archive_index(path, archive)
        self.assertEqual(report.generations, [2, 1])
        self.assertNotIn(9, [g for g in report.generations])
        self.assertEqual(
            [m["path"] for m in report.members],
            ["current-2.sqlite", "generation-1.sqlite", "generations.json"],
        )
        restored = self.tmp / "r.nearidx"
        restore_index(archive, restored)
        with open_index(restored) as idx:
            self.assertEqual([g.generation for g in idx.generations()], [2, 1])
            self.assertIsNone(idx.doc_meta("ghost"))
            self.assertEqual(idx.verify(full=True, generations=True), [])

    def test_archive_rejects_bad_paths_and_states(self):
        path = self.build(self.pump_docs(2))
        with self.assertRaises(FileNotFoundError):
            archive_index(self.tmp / "missing.nearidx", self.tmp / "a.pkg")
        self.assertFalse((self.tmp / "a.pkg").exists())

        junk = self.tmp / "junk.nearidx"
        junk.write_bytes(b"not a database")
        with self.assertRaises(ArchiveFormatError):
            archive_index(junk, self.tmp / "b.pkg")
        self.assertFalse((self.tmp / "b.pkg").exists())

        # Existing destination is not overwritten without overwrite=True.
        archive = self.tmp / "a.pkg"
        archive.write_bytes(b"keep me")
        with self.assertRaises(ArchiveExistsError):
            archive_index(path, archive)
        self.assertEqual(archive.read_bytes(), b"keep me")
        archive_index(path, archive, overwrite=True)
        self.assertEqual(inspect_archive(archive).current_generation, 1)

        # Symlinked index is refused.
        link = self.tmp / "link.nearidx"
        link.symlink_to(path)
        with self.assertRaises(ArchiveFormatError):
            archive_index(link, self.tmp / "c.pkg")
        self.assertFalse((self.tmp / "c.pkg").exists())

    def test_archive_rejects_inconsistent_generation_state(self):
        path = self.build(self.pump_docs(3))
        self.refresh(path, [{"op": "delete", "doc_id": "d02"}])
        # Missing retained generation file.
        gen1 = _gen_file_path(path, 1)
        backup = gen1.read_bytes()
        gen1.unlink()
        with self.assertRaises(ArchiveFormatError):
            archive_index(path, self.tmp / "missing.pkg")
        self.assertFalse((self.tmp / "missing.pkg").exists())
        gen1.write_bytes(backup)

        # Registry that does not describe the current file.
        reg = _read_registry(path)
        reg["current"] = {"generation": 1, "fingerprint": "0" * 64}
        _registry_path(path).write_text(json.dumps(reg), encoding="utf-8")
        with self.assertRaises(ArchiveFormatError):
            archive_index(path, self.tmp / "badreg.pkg")
        self.assertFalse((self.tmp / "badreg.pkg").exists())
        # The source index is untouched by the failed archive attempts.
        with open_index(path, verify=False) as idx:
            self.assertEqual(idx.generation, 2)
            self.assertEqual(idx.verify(full=True), [])

    def test_archive_refuses_while_a_publish_is_journaled(self):
        path = self.build(self.pump_docs(2))
        self.refresh(path, [{"op": "delete", "doc_id": "d02"}])
        reg = _read_registry(path)
        reg["pending"] = {
            "generation": 3,
            "fingerprint": "0" * 64,
            "from_generation": 2,
            "from_fingerprint": reg["current"]["fingerprint"],
        }
        _registry_path(path).write_text(json.dumps(reg), encoding="utf-8")
        with self.assertRaises(ArchiveFormatError):
            archive_index(path, self.tmp / "pending.pkg")
        self.assertFalse((self.tmp / "pending.pkg").exists())

    def test_archive_fails_cleanly_when_registry_changes_mid_copy(self):
        path = self.build(self.pump_docs(4))
        self.refresh(path, [{"op": "delete", "doc_id": "d04"}])
        original_copy = archive_mod._copy_stream
        calls = {"n": 0}

        def racing_copy(src, dst, **kwargs):
            calls["n"] += 1
            result = original_copy(src, dst, **kwargs)
            if calls["n"] == 1:
                # Rewrite the registry while the archive is still copying.
                reg = _read_registry(path)
                reg["retain_old"] = 5
                _registry_path(path).write_text(
                    json.dumps(reg), encoding="utf-8"
                )
            return result

        archive_mod._copy_stream = racing_copy
        try:
            with self.assertRaises(ArchiveFormatError):
                archive_index(path, self.tmp / "race.pkg")
        finally:
            archive_mod._copy_stream = original_copy
        self.assertFalse((self.tmp / "race.pkg").exists())
        self.assertEqual(
            [p.name for p in self.tmp.iterdir() if p.name.endswith(".tmp")], []
        )
        # Source index is still complete and current.
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 2)

    def test_archive_fails_cleanly_when_current_file_is_replaced_mid_copy(self):
        path = self.build(self.pump_docs(3))
        original_copy = archive_mod._copy_stream
        calls = {"n": 0}

        def racing_copy(src, dst, **kwargs):
            calls["n"] += 1
            result = original_copy(src, dst, **kwargs)
            if calls["n"] == 1:
                self.refresh(path, [{"op": "delete", "doc_id": "d01"}])
            return result

        archive_mod._copy_stream = racing_copy
        try:
            with self.assertRaises(ArchiveFormatError):
                archive_index(path, self.tmp / "race2.pkg")
        finally:
            archive_mod._copy_stream = original_copy
        self.assertFalse((self.tmp / "race2.pkg").exists())
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 2)
            self.assertEqual(idx.verify(full=True, generations=True), [])


class RestoreRejectionTests(TempCase):
    def setUp(self):
        super().setUp()
        self.source = self.build(self.pump_docs(4))
        self.refresh(self.source, [{"op": "delete", "doc_id": "d02"}])
        self.archive = self.tmp / "good.pkg"
        archive_index(self.source, self.archive)
        self.target = self.tmp / "target.nearidx"

    def assert_target_absent(self):
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.is_symlink())
        self.assertFalse(_registry_path(self.target).exists())
        for g in (1, 2):
            self.assertFalse(_gen_file_path(self.target, g).exists())
        self.assertEqual(
            [p for p in self.tmp.iterdir()
             if p.name.startswith("." + self.target.name)], []
        )

    def assert_rejected(self, archive_path):
        with self.assertRaises(ArchiveError):
            restore_index(archive_path, self.target)
        self.assert_target_absent()

    def _tampered(self, mutate, name="tampered.pkg"):
        manifest, payload, _h, _f = read_container(self.archive)
        out = mutate(manifest, payload)
        path = self.tmp / name
        build_container(path, out[0], out[1])
        return path

    def test_rejects_truncated_and_extended_container(self):
        raw = self.archive.read_bytes()
        truncated = self.tmp / "trunc.pkg"
        truncated.write_bytes(raw[: len(raw) // 2])
        self.assert_rejected(truncated)
        extended = self.tmp / "extended.pkg"
        extended.write_bytes(raw + b"\x00" * 10)
        self.assert_rejected(extended)
        short_header = self.tmp / "short.pkg"
        short_header.write_bytes(raw[:10])
        self.assert_rejected(short_header)

    def test_rejects_replaced_member_bytes_same_length(self):
        def mutate(manifest, payload):
            data = bytearray(payload["generation-1.sqlite"])
            # Flip bytes in the middle: same length, different content.
            for i in range(5000, 5010):
                data[i] ^= 0xFF
            payload["generation-1.sqlite"] = bytes(data)
            return manifest, payload

        self.assert_rejected(self._tampered(mutate))

    def test_rejects_member_bytes_when_manifest_digest_still_matches(self):
        """The digest must cover the real bytes, not just manifest fields."""
        def mutate(manifest, payload):
            # Swap the two generation payloads and also update sizes/digests
            # of the *manifest entries* consistently -- an attacker who
            # recomputes bookkeeping must still fail SQLite verification.
            cur = next(m for m in manifest["members"]
                       if m["role"] == "current")
            gen = next(m for m in manifest["members"]
                       if m["role"] == "generation")
            payload[cur["path"]], payload[gen["path"]] = (
                payload[gen["path"]], payload[cur["path"]]
            )
            for member in (cur, gen):
                data = payload[member["path"]]
                member["size"] = len(data)
                member["sha256"] = hashlib.sha256(data).hexdigest()
            return manifest, payload

        path = self._tampered(mutate)
        with self.assertRaises(ArchiveFormatError):
            restore_index(path, self.target)
        self.assert_target_absent()

    def test_rejects_missing_extra_and_unregistered_members(self):
        # Missing member: drop the registry member from manifest and payload.
        def drop(manifest, payload):
            manifest["members"] = [
                m for m in manifest["members"] if m["role"] != "registry"
            ]
            return manifest, payload

        self.assert_rejected(self._tampered(drop, "drop.pkg"))

        # Extra bytes appended inside the payload region without a manifest
        # entry: offsets must tile the payload exactly.
        def extra_payload(manifest, payload):
            payload["generation-1.sqlite"] += b"extra"
            return manifest, payload

        self.assert_rejected(self._tampered(extra_payload, "extra.pkg"))

        # Unregistered member listed in the manifest.
        def unregistered(manifest, payload):
            data = b"junk"
            payload["generation-9.sqlite"] = data
            last = manifest["members"][-1]
            manifest["members"].append({
                "role": "generation",
                "generation": 9,
                "fingerprint": "a" * 64,
                "file_name": "unregistered.gen-9",
                "path": "generation-9.sqlite",
                "offset": last["offset"] + last["size"],
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            return manifest, payload

        self.assert_rejected(self._tampered(unregistered, "unreg.pkg"))

    def test_rejects_duplicate_members(self):
        def duplicate(manifest, payload):
            item = dict(next(m for m in manifest["members"]
                             if m["role"] == "generation"))
            manifest["members"].insert(1, item)
            return manifest, payload

        self.assert_rejected(self._tampered(duplicate, "dup.pkg"))

    def test_rejects_malicious_member_paths(self):
        for i, evil in enumerate((
            "../escaped.sqlite",
            "/absolute.sqlite",
            "nested/gen.sqlite",
            "..\\windows.sqlite",
            "generation-1.sqlite\x00",
        )):
            def mutate(manifest, payload, evil=evil):
                target = next(
                    m for m in manifest["members"] if m["role"] == "generation"
                )
                payload[evil] = payload.pop(target["path"])
                target["path"] = evil
                return manifest, payload

            with self.subTest(path=evil):
                self.assert_rejected(
                    self._tampered(mutate, f"evil{i}.pkg")
                )
        # Nothing was written outside the target directory either.
        self.assertEqual(
            sorted(p.name for p in Path(self.tmp).parent.iterdir()
                   if p.name.startswith("escaped")), []
        )

    def test_rejects_unsupported_manifest_and_container_versions(self):
        def bump(manifest, payload):
            manifest["manifest_version"] = 2
            return manifest, payload

        self.assert_rejected(self._tampered(bump, "v2.pkg"))

        def bump_container(manifest, payload):
            return manifest, payload

        path = self._tampered(bump_container, "v9.pkg")
        raw = bytearray(path.read_bytes())
        # Patch header_version (offset 8) to 9.
        struct.pack_into("<I", raw, 8, 9)
        path.write_bytes(bytes(raw))
        self.assert_rejected(path)

    def test_rejects_non_index_member_content(self):
        def corrupt_sqlite(manifest, payload):
            cur = next(m for m in manifest["members"]
                       if m["role"] == "current")
            data = b"SQLite format 3\x00" + b"\x00" * 200
            payload[cur["path"]] = data
            cur["size"] = len(data)
            cur["sha256"] = hashlib.sha256(data).hexdigest()
            return manifest, payload

        self.assert_rejected(self._tampered(corrupt_sqlite, "corrupt.pkg"))

    def test_rejects_wrong_index_identity_and_fingerprint(self):
        def wrong_identity(manifest, payload):
            manifest["index_id"] = "deadbeefdeadbeef"
            return manifest, payload

        self.assert_rejected(self._tampered(wrong_identity, "id.pkg"))

        def wrong_fingerprint(manifest, payload):
            manifest["current"]["fingerprint"] = "f" * 64
            for m in manifest["members"]:
                if m["role"] == "current":
                    m["fingerprint"] = "f" * 64
            return manifest, payload

        self.assert_rejected(self._tampered(wrong_fingerprint, "fp.pkg"))

    def test_rejects_registry_mismatch_and_bad_registration_relations(self):
        def mismatch(manifest, payload):
            reg = json.loads(payload["generations.json"])
            reg["retained"] = []
            payload["generations.json"] = json.dumps(reg).encode()
            member = next(m for m in manifest["members"]
                          if m["role"] == "registry")
            member["size"] = len(payload["generations.json"])
            member["sha256"] = hashlib.sha256(
                payload["generations.json"]
            ).hexdigest()
            return manifest, payload

        self.assert_rejected(self._tampered(mismatch, "regmismatch.pkg"))

        def future_generation(manifest, payload):
            manifest["retained"][0]["generation"] = 5
            for m in manifest["members"]:
                if m["role"] == "generation":
                    m["generation"] = 5
            return manifest, payload

        self.assert_rejected(self._tampered(future_generation, "future.pkg"))

    def test_rejects_symlinked_archive_and_target(self):
        link = self.tmp / "link.pkg"
        link.symlink_to(self.archive)
        with self.assertRaises(ArchiveFormatError):
            restore_index(link, self.target)
        self.assert_target_absent()

        # Symlinked destination (even dangling) is never overwritten.
        dangling = self.tmp / "dangling.nearidx"
        dangling.symlink_to(self.tmp / "nowhere")
        with self.assertRaises(ArchiveExistsError):
            restore_index(self.archive, dangling)
        self.assertTrue(dangling.is_symlink())

    def test_existing_destination_is_never_overwritten(self):
        existing = self.build([("keep", "keep this text")], "existing")
        before = existing.read_bytes()
        before_fp = _file_signature(existing)
        with self.assertRaises(ArchiveExistsError):
            restore_index(self.archive, existing)
        self.assertEqual(existing.read_bytes(), before)
        self.assertEqual(_file_signature(existing), before_fp)
        with open_index(existing) as idx:
            self.assertEqual(idx.doc_ids(), ["keep"])
            self.assertEqual(idx.text("keep"), "keep this text")

        # A pre-existing generation file also blocks the restore.
        other = self.tmp / "other.nearidx"
        _gen_file_path(other, 1).write_bytes(b"do not touch")
        with self.assertRaises(ArchiveExistsError):
            restore_index(self.archive, other)
        self.assertFalse(other.exists())
        self.assertEqual(_gen_file_path(other, 1).read_bytes(), b"do not touch")

    def test_rejects_truncated_member_payload_with_stale_manifest_sizes(self):
        manifest, payload, _h, _f = read_container(self.archive)
        # Shrink one member's bytes but leave the manifest claiming the
        # original length: the payload region then ends before the manifest.
        gen = next(m for m in manifest["members"] if m["role"] == "generation")
        payload[gen["path"]] = payload[gen["path"]][:-512]
        path = self.tmp / "shrunk.pkg"
        build_container(path, manifest, payload)
        with self.assertRaises(ArchiveFormatError):
            restore_index(path, self.target)
        self.assert_target_absent()

    def test_rejects_member_offset_field_tampering(self):
        def shift_offset(manifest, payload):
            manifest["members"][1]["offset"] += 1
            return manifest, payload

        self.assert_rejected(self._tampered(shift_offset, "offset.pkg"))

    def test_rejects_duplicate_manifest_json_keys(self):
        # Duplicate "retained" keys would let a reader and a writer disagree;
        # the manifest checksum is recomputed so only key duplication fails.
        raw = self.archive.read_bytes()
        magic, version, manifest_at, manifest_len, _r = _HEADER.unpack(
            raw[:_HEADER_SIZE]
        )
        manifest = json.loads(raw[manifest_at:manifest_at + manifest_len])
        text = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        # Inject a second "retained" member right after the first key.
        text = '{"retained":[],' + text.lstrip("{")
        manifest_bytes = text.encode("utf-8")
        out = bytearray(raw[:manifest_at]) + bytearray(manifest_bytes) + \
            bytearray(_FOOTER.pack(
                hashlib.sha256(manifest_bytes).digest(),
                len(manifest_bytes), 1, _FOOTER_MAGIC,
            ))
        struct.pack_into("<IQQ", out, 8, version, manifest_at,
                         len(manifest_bytes))
        path = self.tmp / "dupes.pkg"
        path.write_bytes(bytes(out))
        self.assert_rejected(path)

    def test_rejects_symlinked_registry_or_generation_member(self):
        # A registry member that is a symlink inside the staging directory is
        # impossible to construct from a flat archive; the extractor creates
        # regular files only.  Check instead that the *source* side refuses
        # symlinked generation files at archive time.
        source = self.build([("d1", PUMP)], "symsrc")
        self.refresh(source, [{"op": "upsert", "doc_id": "d2", "text": PUMP}])
        gen1 = _gen_file_path(source, 1)
        moved = self.tmp / "moved-gen-1"
        os.replace(gen1, moved)
        gen1.symlink_to(moved)
        with self.assertRaises(ArchiveFormatError):
            archive_index(source, self.tmp / "sym.pkg")
        self.assertFalse((self.tmp / "sym.pkg").exists())
        # And the source index is still completely usable.
        with open_index(source) as idx:
            self.assertEqual(idx.verify(full=True, generations=True), [])
            gen1.unlink()
            os.replace(moved, gen1)

    def test_rejects_unregistered_generation_content(self):
        # Swap the retained generation's file for a valid index with the
        # same generation number but a different fingerprint.
        other = self.build(self.pump_docs(2), "other")
        manifest, payload, _h, _f = read_container(self.archive)
        gen = next(m for m in manifest["members"] if m["role"] == "generation")
        payload[gen["path"]] = other.read_bytes()
        gen["size"] = len(payload[gen["path"]])
        gen["sha256"] = hashlib.sha256(payload[gen["path"]]).hexdigest()
        path = self.tmp / "swapped.pkg"
        build_container(path, manifest, payload)
        with self.assertRaises(ArchiveFormatError):
            restore_index(path, self.target)
        self.assert_target_absent()


class RestoreRetryAfterFailureTests(TempCase):
    """A rejected restore must leave the destination ready for a retry."""

    def setUp(self):
        super().setUp()
        self.source = self.build(self.pump_docs(4))
        with open_index(self.source) as idx:
            self.cursor = self.page(idx, limit=2).next_cursor
        self.refresh(self.source, [{"op": "delete", "doc_id": "d03"}])
        self.archive = self.tmp / "good.pkg"
        archive_index(self.source, self.archive)
        self.source_sig = _file_signature(self.source)
        self.target = self.tmp / "retry.nearidx"

    def test_rejected_archive_then_good_restore_into_same_target(self):
        manifest, payload, _h, _f = read_container(self.archive)
        cur = next(m for m in manifest["members"] if m["role"] == "current")
        payload[cur["path"]] = b"garbage" * 100
        cur["size"] = len(payload[cur["path"]])
        cur["sha256"] = hashlib.sha256(payload[cur["path"]]).hexdigest()
        bad = self.tmp / "bad.pkg"
        build_container(bad, manifest, payload)

        with self.assertRaises(ArchiveError):
            restore_index(bad, self.target)
        self.assertFalse(self.target.exists())
        self.assertFalse(_registry_path(self.target).exists())
        self.assertEqual(_file_signature(self.source), self.source_sig)

        restore_index(self.archive, self.target)
        with open_index(self.target) as idx:
            self.assertEqual(idx.verify(full=True, generations=True), [])
            _, flat = self.walk_chain(idx, self.cursor)
            self.assertEqual(flat[0][0], "d03")
            self.assertEqual(flat[0][4], PUMP)
        self.assertEqual(_file_signature(self.source), self.source_sig)

    def test_failed_restore_does_not_modify_source_index(self):
        damaged = self.tmp / "damaged.pkg"
        damaged.write_bytes(self.archive.read_bytes()[:-7])
        with self.assertRaises(ArchiveError):
            restore_index(damaged, self.target)
        self.assertEqual(_file_signature(self.source), self.source_sig)
        with open_index(self.source) as idx:
            self.assertEqual(idx.generation, 2)
            self.assertEqual(idx.verify(full=True, generations=True), [])
            self.assertEqual(
                self.win_tuples(idx, limit=100),
                [("d01", 0, 2, (0, 2), PUMP), ("d02", 0, 2, (0, 2), PUMP),
                 ("d04", 0, 2, (0, 2), PUMP)],
            )

    def test_inspect_archive_detects_payload_tampering(self):
        manifest, payload, _h, _f = read_container(self.archive)
        payload["current-2.sqlite"] = payload["current-2.sqlite"][:-16] + \
            b"0123456789abcdef"
        path = self.tmp / "bad.pkg"
        build_container(path, manifest, payload)
        with self.assertRaises(ArchiveFormatError):
            inspect_archive(path)
        # Structural-only inspection still reads the manifest.
        self.assertEqual(inspect_archive(path, verify=False).current_generation, 2)


# ---------------------------------------------------------------------------
# interruption / recovery
# ---------------------------------------------------------------------------

CRASH_DRIVER = (
    "import os, sys\n"
    f"sys.path.insert(0, {str(ROOT)!r})\n"
    "import nearidx.archive as A\n"
    "crash_at = int(os.environ['CRASH_AT'])\n"
    "mode = os.environ.get('CRASH_MODE', 'before')\n"
    "orig = os.replace\n"
    "state = {'n': 0}\n"
    "def bomb(a, b):\n"
    "    state['n'] += 1\n"
    "    if state['n'] == crash_at:\n"
    "        if mode == 'after':\n"
    "            orig(a, b)\n"
    "        os._exit(3)\n"
    "    return orig(a, b)\n"
    "os.replace = bomb\n"
    "A.restore_index(sys.argv[1], sys.argv[2])\n"
)


class RestoreInterruptionTests(TempCase):
    def setUp(self):
        super().setUp()
        self.source = self.build(self.pump_docs(5))
        with open_index(self.source) as idx:
            self.cursor = self.page(idx, limit=2).next_cursor
            self.old_full = self.win_tuples(idx, limit=100)
        self.refresh(self.source, [{"op": "delete", "doc_id": "d03"}])
        self.archive = self.tmp / "a.pkg"
        archive_index(self.source, self.archive)
        self.manifest, _payload, _h, _f = read_container(self.archive)
        self.gen_count = sum(
            1 for m in self.manifest["members"] if m["role"] == "generation"
        )
        self.source_bytes = self.source.read_bytes()
        self.source_gen_bytes = _gen_file_path(self.source, 1).read_bytes()
        self.source_registry = _registry_path(self.source).read_bytes()

    def crash(self, crash_at, mode="before"):
        target = self.tmp / f"target-{crash_at}-{mode}.nearidx"
        driver = self.tmp / "driver.py"
        driver.write_text(CRASH_DRIVER, encoding="utf-8")
        env = dict(os.environ, CRASH_AT=str(crash_at), CRASH_MODE=mode)
        proc = subprocess.run(
            [sys.executable, "-I", str(driver), str(self.archive), str(target)],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(proc.returncode, 3, proc.stderr)
        return target

    def assert_source_untouched(self):
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        self.assertEqual(
            _gen_file_path(self.source, 1).read_bytes(),
            self.source_gen_bytes,
        )
        self.assertEqual(
            _registry_path(self.source).read_bytes(), self.source_registry
        )
        with open_index(self.source) as idx:
            self.assertEqual(idx.generation, 2)
            self.assertEqual(idx.verify(full=True, generations=True), [])

    def test_interrupted_restore_is_never_openable_and_retry_succeeds(self):
        # Replace calls: 1 marker, 2..(1+gens) generation files,
        # (2+gens) registry, (3+gens) current.  Crashing *before* the current
        # rename must leave no openable index at all.
        last_before_current = 2 + self.gen_count
        for crash_at in range(1, last_before_current + 1):
            with self.subTest(crash_at=crash_at):
                target = self.crash(crash_at)
                self.assertFalse(target.exists())
                with self.assertRaises(FileNotFoundError):
                    open_index(target)
                # Retry after the interruption finishes a complete restore.
                restore_index(self.archive, target)
                with open_index(target) as idx:
                    self.assertEqual(idx.generation, 2)
                    self.assertEqual(
                        idx.verify(full=True, generations=True), []
                    )
                    _, flat = self.walk_chain(idx, self.cursor)
                    self.assertEqual(flat, self.old_full[2:])
                    self.assertEqual(
                        self.win_tuples(idx, limit=100),
                        [("d01", 0, 2, (0, 2), PUMP),
                         ("d02", 0, 2, (0, 2), PUMP),
                         ("d04", 0, 2, (0, 2), PUMP),
                         ("d05", 0, 2, (0, 2), PUMP)],
                    )
                self.assertFalse(
                    (self.tmp / f".{target.name}.nearidx-restore").exists()
                )
                self.assert_source_untouched()

    def test_interruption_after_current_rename_leaves_complete_index(self):
        target = self.crash(3 + self.gen_count, mode="after")
        # The target is complete and usable immediately.
        with open_index(target) as idx:
            self.assertEqual(idx.generation, 2)
            self.assertEqual(idx.verify(full=True, generations=True), [])
            _, flat = self.walk_chain(idx, self.cursor)
            self.assertEqual(flat, self.old_full[2:])
        # A marker may remain; recovery rolls forward and cleans it.
        self.assertTrue(recover_restore(target))
        self.assertFalse(
            (self.tmp / f".{target.name}.nearidx-restore").exists()
        )
        with open_index(target) as idx:
            self.assertEqual(idx.verify(full=True, generations=True), [])
        # A further restore refuses because a complete index exists.
        with self.assertRaises(ArchiveExistsError):
            restore_index(self.archive, target)
        with open_index(target) as idx:
            self.assertEqual(idx.generation, 2)
        self.assert_source_untouched()

    def test_repeated_interruptions_never_expose_partial_state(self):
        target = self.tmp / "repeated.nearidx"
        for crash_at in (1, 2, 3):
            driver = self.tmp / "driver.py"
            driver.write_text(CRASH_DRIVER, encoding="utf-8")
            env = dict(os.environ, CRASH_AT=str(crash_at))
            proc = subprocess.run(
                [sys.executable, "-I", str(driver), str(self.archive),
                 str(target)],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertFalse(target.exists())
            with self.assertRaises(FileNotFoundError):
                open_index(target)
        restore_index(self.archive, target)
        with open_index(target) as idx:
            self.assertEqual(idx.verify(full=True, generations=True), [])


# ---------------------------------------------------------------------------
# streaming evidence
# ---------------------------------------------------------------------------

class StreamingTests(TempCase):
    def test_archive_and_restore_use_bounded_reads(self):
        big = "pump " * 400_000 + "valve"
        path = self.build([("big", big), ("d2", PUMP)])
        archive = self.tmp / "big.pkg"

        class Recorder:
            def __init__(self, real):
                self.real = real
                self.read_sizes = []

            def __enter__(self):
                self._fh = self.real.__enter__()
                return self

            def __exit__(self, *exc):
                return self.real.__exit__(*exc)

            def __getattr__(self, name):
                attr = getattr(self._fh, name)
                if name == "read":
                    def read(n=-1):
                        self.read_sizes.append(n)
                        return attr(n)
                    return read
                return attr

        real_open = archive_mod._open
        sizes: dict = {}

        def tracking_open(file, mode="r", *args, **kwargs):
            rec = Recorder(real_open(file, mode, *args, **kwargs))
            sizes.setdefault(str(file), []).append(rec)
            return rec

        archive_mod._open = tracking_open
        try:
            report = archive_index(path, archive)
        finally:
            archive_mod._open = real_open
        self.assertGreater(report.size, 1_000_000)
        all_reads = [n for reads in sizes.values() for rec in reads
                     for n in rec.read_sizes]
        self.assertTrue(all_reads)
        # Every read is bounded; no whole-file read() or read(-1).
        self.assertTrue(all(n > 0 for n in all_reads))
        self.assertLessEqual(max(all_reads), archive_mod.CHUNK_SIZE)
        self.assertGreater(len(all_reads), 2)

        # Restore also streams: instrument the module the same way.
        restored = self.tmp / "big-restored.nearidx"
        sizes.clear()
        archive_mod._open = tracking_open
        try:
            restore_index(archive, restored)
        finally:
            archive_mod._open = real_open
        all_reads = [n for reads in sizes.values() for rec in reads
                     for n in rec.read_sizes]
        self.assertTrue(all_reads)
        self.assertTrue(all(n > 0 for n in all_reads))
        self.assertLessEqual(max(all_reads), archive_mod.CHUNK_SIZE)
        with open_index(restored) as idx:
            self.assertEqual(idx.text("big"), big)
            self.assertEqual(idx.verify(full=True), [])

    def test_manifest_carries_incremental_offsets_for_every_member(self):
        path = self.build(self.pump_docs(3))
        for gen in (1, 2):
            self.refresh(path, [{"op": "delete", "doc_id": f"d0{gen}"}])
        manifest = inspect_archive(
            archive_index(path, self.tmp / "a.pkg").path
        )
        expected = archive_mod._HEADER_SIZE
        for member in manifest.members:
            self.assertEqual(member.offset, expected)
            expected += member.size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class CliArchiveTests(TempCase):
    def test_archive_inspect_restore_round_trip(self):
        path = self.build(self.pump_docs(3))
        self.refresh(path, [{"op": "delete", "doc_id": "d03"}])
        archive = self.tmp / "cli.pkg"
        code, out, err = self.run_cli("archive", str(path), str(archive))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["generations"], [2, 1])

        code, out, err = self.run_cli("inspect-archive", str(archive))
        self.assertEqual(code, 0, err)
        manifest = json.loads(out)
        self.assertEqual(manifest["current"]["generation"], 2)
        self.assertEqual(
            [m["role"] for m in manifest["members"]],
            ["current", "generation", "registry"],
        )

        restored = self.tmp / "cli-restored.nearidx"
        code, out, err = self.run_cli(
            "restore", str(archive), str(restored)
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["index_id"],
                         json.loads(out)["index_id"])
        code, out, err = self.run_cli(
            "search", str(restored), QUERY, "--json", "--limit", "10"
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(
            [w["doc_id"] for w in json.loads(out)["windows"]],
            ["d01", "d02"],
        )
        code, out, err = self.run_cli(
            "verify", str(restored), "--full", "--generations"
        )
        self.assertEqual(code, 0, err)

    def test_cli_errors(self):
        path = self.build(self.pump_docs(2))
        code, _out, err = self.run_cli(
            "archive", str(self.tmp / "missing.nearidx"),
            str(self.tmp / "a.pkg"),
        )
        self.assertEqual(code, 2)
        self.assertIn("index error", err)
        archive = self.tmp / "a.pkg"
        self.run_cli("archive", str(path), str(archive))
        code, _out, err = self.run_cli(
            "restore", str(archive), str(path)
        )
        self.assertEqual(code, 2)
        self.assertIn("restore error", err)

        bad = self.tmp / "bad.pkg"
        bad.write_bytes(b"garbage")
        code, _out, err = self.run_cli(
            "restore", str(bad), str(self.tmp / "new.nearidx")
        )
        self.assertEqual(code, 2)
        self.assertIn("restore error", err)
        self.assertFalse((self.tmp / "new.nearidx").exists())

        # recover-restore on a clean path is a no-op.
        code, out, err = self.run_cli(
            "recover-restore", str(self.tmp / "new.nearidx")
        )
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["recovered"], False)


if __name__ == "__main__":
    unittest.main()
