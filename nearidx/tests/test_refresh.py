"""Tests for batch refresh: generational publish, cursor binding, recovery.

Every expectation here is checked against windows, witnesses and original
text snippets -- never only against a returned generation number.

Run with ``python -m unittest discover -s nearidx/tests -t .`` or ``pytest``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nearidx import (
    CursorError,
    GenerationExpiredError,
    ManifestError,
    StaleSourceError,
    build_index,
    open_index,
    parse_manifest,
    parse_query,
    rebuild_document,
    refresh_index,
    search,
    text_sha256,
)
from nearidx.cli import main as cli_main
from nearidx.db import _gen_file_path, _read_registry, _registry_path

ROOT = Path(__file__).resolve().parents[2]

QUERY = "NEAR(3,[pump,pump])"
PUMP_TEXT = "pump valve pump"
PUMP_WINDOW = (0, 2, (0, 2))  # start, end, witness


class TempIndexCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, docs, name="t"):
        path = self.tmp / f"{name}.nearidx"
        build_index(path, docs, name=name, overwrite=True)
        return path

    def page(self, index, query=QUERY, **kwargs):
        return search(index, parse_query(query), **kwargs)

    def win_tuples(self, index, query=QUERY, **kwargs):
        return [
            (w.doc_id, w.start, w.end, w.witness)
            for w in self.page(index, query, **kwargs).windows
        ]

    def write_manifest(self, ops, name="manifest.jsonl"):
        path = self.tmp / name
        path.write_text(
            "".join(json.dumps(op) + "\n" for op in ops), encoding="utf-8"
        )
        return path

    def pump_docs(self, n=5):
        return [(f"d{i:02d}", PUMP_TEXT) for i in range(1, n + 1)]

    def expected_pump_windows(self, doc_ids):
        return [(d, 0, 2, (0, 2)) for d in doc_ids]

    def walk_chain(self, index, cursor, query=QUERY):
        """Follow a cursor to the end; return (pages, flat window tuples)."""
        pages = []
        cur = cursor
        while True:
            page = self.page(index, query, cursor=cur)
            pages.append(page)
            if not page.has_more:
                break
            cur = page.next_cursor
            self.assertIsNotNone(cur)
        flat = [
            (w.doc_id, w.start, w.end, w.witness)
            for p in pages
            for w in p.windows
        ]
        return pages, flat


class RefreshPublishTests(TempIndexCase):
    """The reviewer workflow: page 1, publish a batch, keep reading."""

    def test_old_cursor_finishes_old_generation_after_reopen(self):
        path = self.build(self.pump_docs())
        with open_index(path) as idx:
            first = self.page(idx, limit=2)
            self.assertEqual(
                [(w.doc_id, w.start, w.end, w.witness) for w in first.windows],
                self.expected_pump_windows(["d01", "d02"]),
            )
            self.assertTrue(first.has_more)
            cursor = first.next_cursor
            gen1_fp = idx.fingerprint
            self.assertEqual(idx.generation, 1)
            # Full old-generation result, captured before the publish.
            old_full = self.win_tuples(idx, limit=100)

        manifest = self.write_manifest(
            [
                {"op": "upsert", "doc_id": "d03", "text": "valve pump valve"},
                {"op": "upsert", "doc_id": "d06", "text": PUMP_TEXT},
                {"op": "delete", "doc_id": "d04"},
            ]
        )
        report = refresh_index(path, manifest)
        self.assertEqual(report.generation, 2)
        self.assertNotEqual(report.index_fingerprint, gen1_fp)
        self.assertEqual((report.upserted, report.deleted), (2, 1))
        self.assertEqual(report.retained_generations, [1])

        # Close and reopen: the old cursor still reads generation 1.
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 2)
            pages = []
            cur = cursor
            while True:
                page = self.page(idx, cursor=cur)
                pages.append(page)
                if not page.has_more:
                    break
                cur = page.next_cursor
                self.assertIsNotNone(cur)
            got = [
                (w.doc_id, w.start, w.end, w.witness)
                for p in pages
                for w in p.windows
            ]
            # Remaining old-generation windows, in the original order, with
            # the original page boundaries: d03 keeps its *original* text and
            # d04 is still there even though the batch replaced/deleted them.
            self.assertEqual(got, old_full[2:])
            self.assertEqual(
                [len(p.windows) for p in pages], [2, 1]  # original page size
            )
            for p in pages:
                self.assertEqual(p.index_fingerprint, gen1_fp)
            snippets = [w.snippet for p in pages for w in p.windows]
            self.assertEqual(snippets, [PUMP_TEXT] * 3)
            self.assertEqual(
                [w.doc_id for p in pages for w in p.windows],
                ["d03", "d04", "d05"],
            )

    def test_new_query_sees_exactly_the_new_generation(self):
        path = self.build(self.pump_docs())
        manifest = self.write_manifest(
            [
                {"op": "upsert", "doc_id": "d03", "text": "valve pump valve"},
                {"op": "upsert", "doc_id": "d06", "text": PUMP_TEXT},
                {"op": "delete", "doc_id": "d04"},
            ]
        )
        refresh_index(path, manifest)
        with open_index(path) as idx:
            # d03 replaced (no longer matches), d04 deleted, d06 added.
            self.assertEqual(
                self.win_tuples(idx, limit=100),
                self.expected_pump_windows(["d01", "d02", "d05", "d06"]),
            )
            self.assertEqual(idx.text("d03"), "valve pump valve")
            self.assertIsNone(idx.doc_meta("d04"))
            self.assertEqual(
                idx.doc_ids(), ["d01", "d02", "d03", "d05", "d06"]
            )
            self.assertEqual(idx.verify(full=True, generations=True), [])

    def test_old_cursor_used_on_same_open_handle_stays_in_old_generation(self):
        path = self.build(self.pump_docs())
        idx = open_index(path)
        try:
            cursor = self.page(idx, limit=2).next_cursor
            old_fp = idx.fingerprint
            refresh_index(
                path, self.write_manifest(
                    [{"op": "upsert", "doc_id": "d03", "text": "valve pump valve"}]
                )
            )
            # The handle keeps serving the generation it was opened on.
            page = self.page(idx, cursor=cursor)
            self.assertEqual(
                [(w.doc_id, w.start, w.end, w.witness) for w in page.windows],
                self.expected_pump_windows(["d03", "d04"]),
            )
            self.assertEqual(page.index_fingerprint, old_fp)
            self.assertEqual(page.windows[0].snippet, PUMP_TEXT)
        finally:
            idx.close()

    def test_cursor_chain_does_not_rebind_to_current_generation(self):
        path = self.build(self.pump_docs(6))
        with open_index(path) as idx:
            cursor = self.page(idx, limit=2).next_cursor
            old_fp = idx.fingerprint
        refresh_index(
            path,
            self.write_manifest([{"op": "delete", "doc_id": "d04"}]),
        )
        refresh_index(
            path,
            self.write_manifest([{"op": "delete", "doc_id": "d05"}]),
        )
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 3)
            self.assertNotEqual(idx.fingerprint, old_fp)
            # Every page of the old chain still reports the old fingerprint.
            seen = []
            cur = cursor
            while True:
                page = self.page(idx, cursor=cur)
                self.assertEqual(page.index_fingerprint, old_fp)
                seen.extend(w.doc_id for w in page.windows)
                if not page.has_more:
                    break
                cur = page.next_cursor
            self.assertEqual(seen, ["d03", "d04", "d05", "d06"])

    def test_cursor_still_validates_query_and_page_size(self):
        path = self.build(self.pump_docs())
        with open_index(path) as idx:
            cursor = self.page(idx, limit=2).next_cursor
        refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d05"}])
        )
        with open_index(path) as idx:
            with self.assertRaises(CursorError):
                self.page(idx, "NEAR(3,[valve,valve])", cursor=cursor)
            with self.assertRaises(CursorError):
                self.page(idx, cursor="not-a-cursor")
            # A well-formed cursor keeps its own page size.
            page = self.page(idx, cursor=cursor, limit=99)
            self.assertEqual(len(page.windows), 2)
            self.assertEqual(page.limit, 2)

    def test_empty_manifest_publishes_identical_content_generation(self):
        path = self.build(self.pump_docs(3))
        with open_index(path) as idx:
            before_fp = idx.fingerprint
            cursor = self.page(idx, limit=2).next_cursor
        report = refresh_index(path, self.write_manifest([]))
        self.assertEqual(report.generation, 2)
        self.assertEqual(report.index_fingerprint, before_fp)
        with open_index(path) as idx:
            self.assertEqual(
                self.win_tuples(idx, limit=100),
                self.expected_pump_windows(["d01", "d02", "d03"]),
            )
            page = self.page(idx, cursor=cursor)
            self.assertEqual(
                [(w.doc_id,) for w in page.windows], [("d03",)]
            )

    def test_upsert_from_path_binds_source_file(self):
        src = self.tmp / "note.txt"
        src.write_text("pump valve pump", encoding="utf-8")
        path = self.build([("d1", "unrelated words")])
        report = refresh_index(
            path,
            self.write_manifest(
                [{"op": "upsert", "doc_id": "d2", "path": str(src)}]
            ),
        )
        self.assertEqual(report.upserted, 1)
        with open_index(path) as idx:
            self.assertEqual(idx.text("d2"), "pump valve pump")
            meta = idx.doc_meta("d2")
            self.assertEqual(meta["source_path"], str(src.resolve()))
            self.assertEqual(
                meta["source_sha256"], text_sha256("pump valve pump")
            )
            idx.check_source("d2")  # not stale
            self.assertEqual(idx.verify(full=True, generations=True), [])
            src.write_text("pump pump valve", encoding="utf-8")
            with self.assertRaises(StaleSourceError):
                idx.check_source("d2")

    def test_refresh_after_rebuild_continues_generation_sequence(self):
        path = self.build(self.pump_docs(2))
        rebuild_document(path, "d01", "pump pump valve")  # in-place: gen 2
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 2)
            cursor = self.page(idx, "pump", limit=1).next_cursor
            gen2_fp = idx.fingerprint
        report = refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d02"}])
        )
        self.assertEqual(report.generation, 3)
        with open_index(path) as idx:
            self.assertEqual([g.generation for g in idx.generations()], [3, 2])
            # The rebuilt content is the retained generation 2.
            pages, flat = self.walk_chain(idx, cursor, query="pump")
            self.assertTrue(all(p.index_fingerprint == gen2_fp for p in pages))
            self.assertEqual(
                flat,
                [("d01", 1, 1, (1,)), ("d02", 0, 0, (0,)), ("d02", 2, 2, (2,))],
            )
            self.assertEqual(pages[0].windows[0].snippet, "pump")
            self.assertEqual(idx.verify(full=True, generations=True), [])

    def test_refresh_report_and_stats_expose_generation(self):
        path = self.build(self.pump_docs(2))
        report = refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d02"}])
        )
        with open_index(path) as idx:
            stats = idx.stats()
            self.assertEqual(stats["generation"], report.generation)
            self.assertEqual(stats["retained_generations"], [1])
            gens = idx.generations()
            self.assertEqual(
                [(g.generation, g.current) for g in gens], [(2, True), (1, False)]
            )
            self.assertTrue(Path(gens[1].path).is_file())


class ManifestValidationTests(TempIndexCase):
    """Every invalid batch is rejected as a whole and changes nothing."""

    def setUp(self):
        super().setUp()
        self.path = self.build([("d1", "pump valve pump"), ("d2", "alpha beta")])
        with open_index(self.path) as idx:
            self.fp_before = idx.fingerprint
            self.cursor = search(
                idx, parse_query("pump"), limit=1
            ).next_cursor

    def assert_unchanged(self):
        with open_index(self.path) as idx:
            self.assertEqual(idx.fingerprint, self.fp_before)
            self.assertEqual(idx.generation, 1)
            self.assertEqual(idx.doc_ids(), ["d1", "d2"])
            self.assertEqual(
                self.win_tuples(idx, "pump", limit=10),
                [("d1", 0, 0, (0,)), ("d1", 2, 2, (2,))],
            )
            # The pre-existing cursor still works.
            page = search(idx, parse_query("pump"), cursor=self.cursor)
            self.assertEqual([w.doc_id for w in page.windows], ["d1"])
            self.assertEqual(idx.verify(full=True), [])
        self.assertFalse(_registry_path(self.path).exists())
        self.assertFalse(_gen_file_path(self.path, 2).exists())

    def assert_rejected(self, ops_or_lines):
        if isinstance(ops_or_lines, str):
            manifest = self.tmp / "bad.jsonl"
            manifest.write_text(ops_or_lines, encoding="utf-8")
        else:
            manifest = self.write_manifest(ops_or_lines, name="bad.jsonl")
        with self.assertRaises(ManifestError):
            refresh_index(self.path, manifest)
        self.assert_unchanged()

    def test_bad_json_rejected(self):
        self.assert_rejected('{"op":"upsert","doc_id":"d3","text":"x"\n')

    def test_non_object_line_rejected(self):
        self.assert_rejected('["upsert","d3"]\n')

    def test_unknown_op_rejected(self):
        self.assert_rejected([{"op": "replace", "doc_id": "d1", "text": "x"}])

    def test_unknown_key_rejected(self):
        self.assert_rejected(
            [{"op": "upsert", "doc_id": "d3", "text": "x", "test": "y"}]
        )

    def test_missing_and_empty_doc_id_rejected(self):
        self.assert_rejected([{"op": "delete"}])
        self.assert_rejected([{"op": "delete", "doc_id": ""}])

    def test_upsert_needs_exactly_one_source(self):
        self.assert_rejected([{"op": "upsert", "doc_id": "d3"}])
        self.assert_rejected(
            [{"op": "upsert", "doc_id": "d3", "text": "x", "path": "y"}]
        )

    def test_text_must_be_clean_string(self):
        self.assert_rejected([{"op": "upsert", "doc_id": "d3", "text": 42}])
        self.assert_rejected([{"op": "upsert", "doc_id": "d3", "text": "a\x00b"}])

    def test_delete_with_payload_rejected(self):
        self.assert_rejected(
            [{"op": "delete", "doc_id": "d1", "text": "x"}]
        )

    def test_delete_of_missing_document_rejected(self):
        self.assert_rejected([{"op": "delete", "doc_id": "nope"}])

    def test_duplicate_doc_id_rejected(self):
        self.assert_rejected(
            [
                {"op": "upsert", "doc_id": "d3", "text": "a"},
                {"op": "upsert", "doc_id": "d3", "text": "b"},
            ]
        )
        self.assert_rejected(
            [
                {"op": "upsert", "doc_id": "d1", "text": "a"},
                {"op": "delete", "doc_id": "d1"},
            ]
        )
        self.assert_rejected(
            [
                {"op": "delete", "doc_id": "d1"},
                {"op": "delete", "doc_id": "d1"},
            ]
        )

    def test_unreadable_and_undecodable_sources_rejected(self):
        self.assert_rejected(
            [{"op": "upsert", "doc_id": "d3", "path": str(self.tmp / "missing.txt")}]
        )
        blob = self.tmp / "blob.bin"
        blob.write_bytes(b"\xff\xfe\x00\x01")
        self.assert_rejected(
            [{"op": "upsert", "doc_id": "d3", "path": str(blob)}]
        )

    def test_missing_manifest_file_rejected(self):
        with self.assertRaises(ManifestError):
            refresh_index(self.path, self.tmp / "no-such-manifest.jsonl")
        self.assert_unchanged()

    def test_one_bad_line_rejects_whole_batch(self):
        # The valid first op must not be applied either.
        self.assert_rejected(
            [
                {"op": "upsert", "doc_id": "d3", "text": "pump"},
                {"op": "delete", "doc_id": "ghost"},
            ]
        )
        with open_index(self.path) as idx:
            self.assertIsNone(idx.doc_meta("d3"))

    def test_parse_manifest_accepts_mapping_iterable(self):
        ops = parse_manifest(
            [
                {"op": "upsert", "doc_id": "x", "text": "a b"},
                {"op": "delete", "doc_id": "y"},
            ],
            current_ids={"y"},
        )
        self.assertEqual([op.op for op in ops], ["upsert", "delete"])
        self.assertEqual(ops[0].text, "a b")


class RetentionTests(TempIndexCase):
    def refresh(self, path, ops, **kwargs):
        return refresh_index(path, self.write_manifest(ops), **kwargs)

    def test_default_keeps_current_plus_two_old_generations(self):
        path = self.build(self.pump_docs(5))
        cursors, fps = {}, {}
        for gen, victim in ((1, "d01"), (2, "d02"), (3, "d03")):
            with open_index(path) as idx:
                self.assertEqual(idx.generation, gen)
                fps[gen] = idx.fingerprint
                cursors[gen] = self.page(idx, limit=1).next_cursor
                self.assertIsNotNone(cursors[gen])
            self.refresh(path, [{"op": "delete", "doc_id": victim}])
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 4)
            self.assertEqual(
                [g.generation for g in idx.generations()], [4, 3, 2]
            )
            # Generation 1 fell out of retention: its file is gone and its
            # cursor reports expiry instead of silently reading current data.
            self.assertFalse(_gen_file_path(path, 1).exists())
            self.assertTrue(_gen_file_path(path, 2).exists())
            self.assertTrue(_gen_file_path(path, 3).exists())
            with self.assertRaises(GenerationExpiredError):
                self.page(idx, cursor=cursors[1])
            # Generations 2 and 3 are still fully queryable: the chains
            # continue with the original windows, order and page size.
            pages2, flat2 = self.walk_chain(idx, cursors[2])
            self.assertEqual(
                flat2, self.expected_pump_windows(["d03", "d04", "d05"])
            )
            self.assertTrue(all(p.index_fingerprint == fps[2] for p in pages2))
            self.assertEqual([len(p.windows) for p in pages2], [1, 1, 1])
            pages3, flat3 = self.walk_chain(idx, cursors[3])
            self.assertEqual(flat3, self.expected_pump_windows(["d04", "d05"]))
            self.assertTrue(all(p.index_fingerprint == fps[3] for p in pages3))

    def test_retain_old_setting_is_enforced_and_persisted(self):
        path = self.build(self.pump_docs(3))
        with self.assertRaises(ValueError):
            self.refresh(path, [{"op": "delete", "doc_id": "d01"}], retain_old=0)
        self.refresh(path, [{"op": "delete", "doc_id": "d01"}], retain_old=1)
        self.refresh(path, [{"op": "delete", "doc_id": "d02"}])
        with open_index(path) as idx:
            # retain_old=1 persisted: only generation 2 is kept.
            self.assertEqual([g.generation for g in idx.generations()], [3, 2])
            self.assertFalse(_gen_file_path(path, 1).exists())
        reg = _read_registry(path)
        self.assertEqual(reg["retain_old"], 1)

    def test_expiry_is_rechecked_against_the_live_registry(self):
        path = self.build(self.pump_docs(3))
        with open_index(path) as idx:
            cursor1 = self.page(idx, limit=1).next_cursor
        self.refresh(path, [{"op": "delete", "doc_id": "d01"}], retain_old=1)
        idx2 = open_index(path)
        try:
            # Generation 1 is retained: resolves through the registry.
            page = self.page(idx2, cursor=cursor1)
            self.assertEqual([w.doc_id for w in page.windows], ["d02"])
        finally:
            idx2.close()
        self.refresh(path, [{"op": "delete", "doc_id": "d02"}], retain_old=1)
        # Generation 1 has now fallen out of retention: the same cursor on a
        # reopened handle must expire, not read current or cached data.
        with open_index(path) as idx3:
            self.assertEqual([g.generation for g in idx3.generations()], [3, 2])
            with self.assertRaises(GenerationExpiredError):
                self.page(idx3, cursor=cursor1)

    def test_expired_cursor_is_a_cursor_error(self):
        path = self.build(self.pump_docs(2))
        with open_index(path) as idx:
            cursor = self.page(idx, limit=1).next_cursor
        self.refresh(path, [{"op": "delete", "doc_id": "d01"}], retain_old=1)
        self.refresh(path, [{"op": "delete", "doc_id": "d02"}], retain_old=1)
        with open_index(path) as idx:
            with self.assertRaises(CursorError):
                self.page(idx, cursor=cursor)


class CrashRecoveryTests(TempIndexCase):
    """A killed publish leaves a complete old or a complete new generation."""

    DRIVER = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import nearidx.refresh as R\n"
        "crash_at = int(os.environ['NEARIDX_CRASH_AT'])\n"
        "orig = os.replace\n"
        "state = {'n': 0}\n"
        "def bomb(a, b):\n"
        "    state['n'] += 1\n"
        "    if state['n'] == crash_at:\n"
        "        os._exit(3)\n"
        "    return orig(a, b)\n"
        "os.replace = bomb\n"
        "R.refresh_index(sys.argv[1], sys.argv[2])\n"
    )

    def test_interrupted_publish_recovers_at_every_step(self):
        # os.replace calls inside one refresh: 1) tmp->staged,
        # 2) registry(pending), 3) main->old gen, 4) staged->main,
        # 5) registry(final).  Crashing at 1-4 must keep generation 1,
        # crashing at 5 must recover to the complete generation 2.
        for crash_at, expected_gen in ((1, 1), (2, 1), (3, 1), (4, 1), (5, 2)):
            with self.subTest(crash_at=crash_at):
                path = self.build(self.pump_docs(3), name=f"crash{crash_at}")
                with open_index(path) as idx:
                    gen1_fp = idx.fingerprint
                    old_cursor = self.page(idx, limit=1).next_cursor
                manifest = self.write_manifest(
                    [
                        {"op": "upsert", "doc_id": "d04", "text": PUMP_TEXT},
                        {"op": "delete", "doc_id": "d02"},
                    ],
                    name=f"crash{crash_at}.jsonl",
                )
                driver = self.tmp / f"driver{crash_at}.py"
                driver.write_text(self.DRIVER, encoding="utf-8")
                env = dict(os.environ, NEARIDX_CRASH_AT=str(crash_at))
                proc = subprocess.run(
                    [sys.executable, "-I", str(driver), str(path), str(manifest)],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(proc.returncode, 3, proc.stderr)
                if crash_at == 4:
                    # Really died mid-swap: the main file is missing.
                    self.assertFalse(path.exists())
                with open_index(path) as idx:
                    self.assertEqual(idx.generation, expected_gen)
                    self.assertEqual(idx.verify(full=True), [])
                    wins = self.win_tuples(idx, limit=100)
                    if expected_gen == 1:
                        self.assertEqual(idx.fingerprint, gen1_fp)
                        self.assertEqual(
                            wins, self.expected_pump_windows(["d01", "d02", "d03"])
                        )
                        # The pre-crash cursor still paginates generation 1.
                        _, flat = self.walk_chain(idx, old_cursor)
                        self.assertEqual(
                            flat, self.expected_pump_windows(["d02", "d03"])
                        )
                    else:
                        self.assertNotEqual(idx.fingerprint, gen1_fp)
                        self.assertEqual(
                            wins,
                            self.expected_pump_windows(["d01", "d03", "d04"]),
                        )
                        # Old generation retained and still queryable.
                        _, flat = self.walk_chain(idx, old_cursor)
                        self.assertEqual(
                            flat, self.expected_pump_windows(["d02", "d03"])
                        )
                reg = _read_registry(path)
                if reg is not None:
                    self.assertIsNone(reg.get("pending"))

    def test_orphaned_unpublished_file_is_never_served(self):
        path = self.build(self.pump_docs(2))
        with open_index(path) as idx:
            gen1_fp = idx.fingerprint
        # Simulate a crash that left a fully built but unpublished file.
        staged = _gen_file_path(path, 2)
        build_index(staged, [("d01", "pump valve pump"), ("d09", "pump valve pump")],
                    overwrite=True)
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 1)
            self.assertEqual(idx.fingerprint, gen1_fp)
            self.assertEqual(idx.doc_ids(), ["d01", "d02"])
        # The next refresh cleans the orphan up.
        refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d02"}])
        )
        with open_index(path) as idx:
            self.assertEqual(idx.doc_ids(), ["d01"])
            self.assertEqual(idx.generation, 2)


class CrossProcessTests(TempIndexCase):
    def test_old_cursor_works_in_a_fresh_process_after_refresh(self):
        path = self.build(self.pump_docs())
        with open_index(path) as idx:
            cursor = self.page(idx, limit=2).next_cursor
            old_full = self.win_tuples(idx, limit=100)
        refresh_index(
            path,
            self.write_manifest(
                [
                    {"op": "upsert", "doc_id": "d03", "text": "valve pump valve"},
                    {"op": "delete", "doc_id": "d04"},
                    {"op": "upsert", "doc_id": "d06", "text": PUMP_TEXT},
                ]
            ),
        )
        driver = self.tmp / "cursor_driver.py"
        driver.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from nearidx import open_index, parse_query, search\n"
            "idx = open_index(sys.argv[1])\n"
            "out = []\n"
            "cur = sys.argv[3]\n"
            "while True:\n"
            "    page = search(idx, parse_query(sys.argv[2]), cursor=cur)\n"
            "    out += [[w.doc_id, w.start, w.end, list(w.witness), w.snippet]\n"
            "            for w in page.windows]\n"
            "    if not page.has_more:\n"
            "        break\n"
            "    cur = page.next_cursor\n"
            "print(json.dumps(out, ensure_ascii=False))\n",
            encoding="utf-8",
        )
        out = subprocess.run(
            [sys.executable, "-I", str(driver), str(path), QUERY, cursor],
            capture_output=True,
            text=True,
            check=True,
        )
        got = json.loads(out.stdout)
        self.assertEqual(
            [(g[0], g[1], g[2], tuple(g[3])) for g in got],
            old_full[2:],
        )
        self.assertEqual([g[4] for g in got], [PUMP_TEXT] * 3)


class RebuildAndIntegrityTests(TempIndexCase):
    def test_single_document_rebuild_still_works_and_invalidates_cursor(self):
        path = self.build(self.pump_docs(3))
        with open_index(path) as idx:
            cursor = self.page(idx, limit=1).next_cursor
        refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d03"}])
        )
        with open_index(path) as idx:
            gen2_cursor = self.page(idx, limit=1).next_cursor
            gen2_fp = idx.fingerprint
        new_fp = rebuild_document(path, "d01", "pump pump valve")
        with open_index(path) as idx:
            self.assertEqual(idx.fingerprint, new_fp)
            self.assertEqual(idx.generation, 3)
            self.assertEqual(
                self.win_tuples(idx, "SEQ(0,[pump,valve])", limit=10),
                [("d01", 1, 2, (1, 2)), ("d02", 0, 1, (0, 1))],
            )
            self.assertEqual(idx.verify(full=True, generations=True), [])
            # The cursor of the just-replaced generation is expired...
            with self.assertRaises(GenerationExpiredError):
                self.page(idx, cursor=gen2_cursor)
            # ...but the retained older generation still paginates.
            _, flat = self.walk_chain(idx, cursor)
            self.assertEqual(flat, self.expected_pump_windows(["d02", "d03"]))

    def test_verify_checks_all_retained_generations(self):
        path = self.build(self.pump_docs(2))
        refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d02"}])
        )
        gen1 = _gen_file_path(path, 1)
        self.assertTrue(gen1.is_file())
        conn = sqlite3.connect(gen1)
        conn.execute(
            "UPDATE documents SET text = 'pump pump valve' WHERE doc_id = 'd01'"
        )
        conn.commit()
        conn.close()
        with open_index(path, verify=False) as idx:
            self.assertEqual(idx.verify(full=True), [])  # current only
            problems = idx.verify(full=True, generations=True)
            self.assertTrue(
                any("generation 1" in p and "sha256" in p for p in problems),
                problems,
            )

    def test_missing_retained_file_is_reported(self):
        path = self.build(self.pump_docs(2))
        refresh_index(
            path, self.write_manifest([{"op": "delete", "doc_id": "d02"}])
        )
        _gen_file_path(path, 1).unlink()
        with open_index(path, verify=False) as idx:
            problems = idx.verify(full=False, generations=True)
            self.assertTrue(any("missing" in p for p in problems), problems)

    def test_normal_queries_snippets_and_spans_unchanged_after_refresh(self):
        text = "STRASSE straße ÄRGER ärger"
        path = self.build([("fold", text), ("main", "pump valve pump")])
        refresh_index(
            path,
            self.write_manifest([{"op": "upsert", "doc_id": "n", "text": "a b c"}]),
        )
        with open_index(path) as idx:
            window = self.page(idx, "SEQ(0,[strasse,ärger])").windows[0]
            self.assertEqual(window.snippet, "straße ÄRGER")
            self.assertEqual((window.char_start, window.char_end), (8, 20))
            near = self.page(idx, "NEAR(3,[pump,pump])").windows[0]
            self.assertEqual(
                (near.doc_id, near.start, near.end, near.witness),
                ("main", 0, 2, (0, 2)),
            )
            self.assertEqual(near.snippet, "pump valve pump")
            self.assertEqual(idx.snippet("fold", 8, 20), "straße ÄRGER")


class CliRefreshTests(TempIndexCase):
    def run_cli(self, *argv):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_refresh_command_publishes_and_reports(self):
        path = self.build(self.pump_docs(3))
        manifest = self.write_manifest(
            [
                {"op": "upsert", "doc_id": "d04", "text": PUMP_TEXT},
                {"op": "delete", "doc_id": "d01"},
            ]
        )
        code, out, _ = self.run_cli("refresh", str(path), str(manifest))
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["generation"], 2)
        self.assertEqual((report["upserted"], report["deleted"]), (1, 1))
        code, out, _ = self.run_cli("search", str(path), QUERY, "--json")
        self.assertEqual(code, 0)
        hits = json.loads(out)["windows"]
        self.assertEqual(
            [w["doc_id"] for w in hits], ["d02", "d03", "d04"]
        )
        code, out, _ = self.run_cli("verify", str(path), "--full", "--generations")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["problems"], [])

    def test_refresh_command_rejects_bad_manifest(self):
        path = self.build(self.pump_docs(2))
        manifest = self.tmp / "bad.jsonl"
        manifest.write_text('{"op":"delete","doc_id":"ghost"}\n', encoding="utf-8")
        code, _out, err = self.run_cli("refresh", str(path), str(manifest))
        self.assertEqual(code, 2)
        self.assertIn("manifest error", err)
        with open_index(path) as idx:
            self.assertEqual(idx.generation, 1)

    def test_refresh_command_rejects_bad_retain_old(self):
        path = self.build(self.pump_docs(2))
        manifest = self.write_manifest([{"op": "delete", "doc_id": "d01"}])
        code, _out, err = self.run_cli(
            "refresh", str(path), str(manifest), "--retain-old", "0"
        )
        self.assertEqual(code, 2)
        self.assertIn("retain_old", err)


if __name__ == "__main__":
    unittest.main()
