"""Tests for federated search across several independent indexes.

Every expectation is checked against delivered windows -- source name,
witness, snippet and spans -- never only against page metadata.

Run with ``python -m unittest discover -s nearidx/tests -t .`` or ``pytest``.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nearidx import (
    CursorError,
    FederatedWindow,
    Federation,
    FederationError,
    GenerationExpiredError,
    IndexFormatError,
    build_index,
    federated_search,
    open_index,
    parse_query,
    refresh_index,
    search,
)
from nearidx.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[2]

QUERY = "NEAR(3,[pump,pump])"
PUMP = "pump valve pump"


class TempFederationCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, docs, name):
        path = self.tmp / f"{name}.nearidx"
        build_index(path, docs, name=name, overwrite=True)
        return path

    def federation(self, mapping, **kwargs):
        return Federation(
            {name: str(path) for name, path in mapping.items()}, **kwargs
        )

    def fpage(self, fed, query=QUERY, **kwargs):
        return federated_search(fed, parse_query(query), **kwargs)

    @staticmethod
    def ftuples(page):
        return [
            (w.source_name, w.doc_id, w.start, w.end, w.witness)
            for w in page.windows
        ]

    def walk(self, fed, query=QUERY, limit=20, cursor=None):
        """Follow the cursor chain to the end; return (pages, flat tuples)."""
        pages = []
        cur = cursor
        while True:
            page = self.fpage(fed, query, limit=limit, cursor=cur)
            pages.append(page)
            if not page.has_more:
                break
            cur = page.next_cursor
            self.assertIsNotNone(cur)
        flat = [t for page in pages for t in self.ftuples(page)]
        return pages, flat

    def write_manifest(self, mapping, name="federation.json"):
        path = self.tmp / name
        path.write_text(json.dumps(mapping), encoding="utf-8")
        return path

    def pump_federation(self):
        """Three sources with interleaved and shared doc_ids."""
        paths = {
            "alpha": self.build(
                [("d01", PUMP), ("d03", PUMP), ("d05", PUMP)], "alpha"
            ),
            "beta": self.build(
                [("d02", PUMP), ("d03", PUMP), ("d06", PUMP)], "beta"
            ),
            "gamma": self.build([("d03", PUMP), ("d04", PUMP)], "gamma"),
        }
        return paths


class GlobalOrderTests(TempFederationCase):
    """The merged stream follows (doc_id, source_name, start, end)."""

    def test_interleaved_sources_merge_in_global_order(self):
        paths = self.pump_federation()
        with self.federation(paths) as fed:
            page = self.fpage(fed, limit=100)
            self.assertEqual(
                self.ftuples(page),
                [
                    ("alpha", "d01", 0, 2, (0, 2)),
                    ("beta", "d02", 0, 2, (0, 2)),
                    ("alpha", "d03", 0, 2, (0, 2)),
                    ("beta", "d03", 0, 2, (0, 2)),
                    ("gamma", "d03", 0, 2, (0, 2)),
                    ("gamma", "d04", 0, 2, (0, 2)),
                    ("alpha", "d05", 0, 2, (0, 2)),
                    ("beta", "d06", 0, 2, (0, 2)),
                ],
            )
            self.assertFalse(page.has_more)
            self.assertIsNone(page.next_cursor)

    def test_same_doc_id_in_two_sources_stays_two_results(self):
        paths = {
            "one": self.build([("shared", PUMP)], "one"),
            "two": self.build([("shared", PUMP)], "two"),
        }
        with self.federation(paths) as fed:
            page = self.fpage(fed, limit=10)
            self.assertEqual(
                self.ftuples(page),
                [
                    ("one", "shared", 0, 2, (0, 2)),
                    ("two", "shared", 0, 2, (0, 2)),
                ],
            )
            # Both keep their own original text span and snippet.
            for w in page.windows:
                self.assertEqual(w.snippet, PUMP)
                self.assertEqual((w.window.char_start, w.window.char_end), (0, 15))

    def test_windows_within_one_doc_order_by_start_end(self):
        paths = {
            "alpha": self.build(
                [("d00", "pump valve pump pump valve pump"), ("d02", PUMP)],
                "alpha",
            ),
            "beta": self.build([("d01", PUMP)], "beta"),
        }
        with self.federation(paths) as fed:
            page = self.fpage(fed, limit=100)
            self.assertEqual(
                self.ftuples(page),
                [
                    ("alpha", "d00", 0, 2, (0, 2)),
                    ("alpha", "d00", 2, 3, (2, 3)),
                    ("alpha", "d00", 3, 5, (3, 5)),
                    ("beta", "d01", 0, 2, (0, 2)),
                    ("alpha", "d02", 0, 2, (0, 2)),
                ],
            )

    def test_tiny_pages_are_a_stable_partition_of_the_big_page(self):
        paths = self.pump_federation()
        with self.federation(paths) as fed:
            full = self.ftuples(self.fpage(fed, limit=1000))
            self.assertEqual(len(full), 8)
            for limit in (1, 2, 3, 5):
                pages, flat = self.walk(fed, limit=limit)
                self.assertEqual(flat, full, msg=f"limit={limit}")
                # No duplicates and no gaps across page boundaries.
                self.assertEqual(len(flat), len(set(flat)))
                for page in pages[:-1]:
                    self.assertEqual(len(page.windows), limit)
                self.assertLessEqual(len(pages[-1].windows), limit)
                self.assertEqual(
                    [p.offset for p in pages], list(range(0, len(full), limit))
                )
                self.assertTrue(all(p.limit == limit for p in pages))

    def test_source_with_no_match_contributes_nothing(self):
        paths = self.pump_federation()
        paths["empty"] = self.build([("x1", "nothing relevant")], "empty")
        with self.federation(paths) as fed:
            page = self.fpage(fed, limit=100)
            self.assertEqual(len(page.windows), 8)
            self.assertNotIn("empty", [w.source_name for w in page.windows])
            self.assertEqual(page.trace["empty"]["candidate_docs"], 0)
            self.assertEqual(page.trace["empty"]["body_reads"], [])

    def test_windows_match_single_index_search_item_for_item(self):
        """The federation layer must not change matching or witnesses."""
        paths = {
            "alpha": self.build(
                [
                    ("m1", PUMP),
                    ("m2", "STRASSE straße ÄRGER ärger"),
                    ("m3", "a b c"),
                ],
                "alpha",
            ),
            "beta": self.build(
                [
                    ("m1", "pump pump valve pump"),
                    ("z9", "emoji \U0001f600 pump valve pump tail"),
                ],
                "beta",
            ),
        }
        queries = (
            "NEAR(3,[pump,pump])",
            "SEQ(0,[strasse,ärger])",
            "OR(pump,ärger)",
            "NEAR(3,[SEQ(1,[a,c]),b])",
        )
        with self.federation(paths) as fed:
            for query in queries:
                with self.subTest(query=query):
                    fed_page = self.fpage(fed, query, limit=1000)
                    by_source = {}
                    for w in fed_page.windows:
                        self.assertIsInstance(w, FederatedWindow)
                        by_source.setdefault(w.source_name, []).append(w)
                    for name, path in paths.items():
                        with open_index(path) as idx:
                            single = search(idx, parse_query(query), limit=1000)
                        expected = [w.as_dict() for w in single.windows]
                        got = [
                            w.window.as_dict()
                            for w in by_source.get(name, [])
                        ]
                        self.assertEqual(got, expected, msg=(query, name))


class GenerationBindingTests(TempFederationCase):
    """Cursors keep reading the generations the first page pinned."""

    def two_source_setup(self):
        paths = {
            "alpha": self.build(
                [(f"d{i:02d}", PUMP) for i in range(1, 5)], "alpha"
            ),
            "beta": self.build(
                [(f"d{i:02d}", PUMP) for i in range(1, 5)], "beta"
            ),
        }
        return paths

    def refresh_beta(self, paths, ops, **kwargs):
        manifest = self.tmp / "refresh.jsonl"
        manifest.write_text(
            "".join(json.dumps(op) + "\n" for op in ops), encoding="utf-8"
        )
        return refresh_index(paths["beta"], manifest, **kwargs)

    def test_cursor_reads_pinned_generations_after_refresh_and_reopen(self):
        paths = self.two_source_setup()
        fed = self.federation(paths)
        first = self.fpage(fed, limit=2)
        self.assertEqual(
            self.ftuples(first),
            [("alpha", "d01", 0, 2, (0, 2)), ("beta", "d01", 0, 2, (0, 2))],
        )
        cursor = first.next_cursor
        pinned = {name: info["fingerprint"] for name, info in
                  first.sources.items()}
        fed.close()

        # Refresh ONLY beta: d03 replaced by a non-match, d04 deleted,
        # d05 added.  Alpha stays untouched.
        report = self.refresh_beta(
            paths,
            [
                {"op": "upsert", "doc_id": "d03", "text": "valve pump valve"},
                {"op": "delete", "doc_id": "d04"},
                {"op": "upsert", "doc_id": "d05", "text": PUMP},
            ],
        )
        self.assertEqual(report.generation, 2)

        # Close everything and reopen: the old cursor still reads the
        # original generations of BOTH sources.
        with self.federation(paths) as fed:
            pages, flat = self.walk(fed, limit=2, cursor=cursor)
            self.assertEqual(
                flat,
                [
                    ("alpha", "d02", 0, 2, (0, 2)),
                    ("beta", "d02", 0, 2, (0, 2)),
                    ("alpha", "d03", 0, 2, (0, 2)),
                    ("beta", "d03", 0, 2, (0, 2)),  # old text still matches
                    ("alpha", "d04", 0, 2, (0, 2)),
                    ("beta", "d04", 0, 2, (0, 2)),  # deleted in gen 2
                ],
            )
            # Every page of the chain reports the pinned old generations --
            # never a mixture of old and new.
            for page in pages:
                for name, info in page.sources.items():
                    self.assertEqual(info["fingerprint"], pinned[name])
                self.assertEqual(page.sources["beta"]["generation"], 1)
            # The old-generation snippets come from the old text.
            beta_d03 = [
                w for p in pages for w in p.windows
                if w.source_name == "beta" and w.doc_id == "d03"
            ]
            self.assertEqual(len(beta_d03), 1)
            self.assertEqual(beta_d03[0].snippet, PUMP)

            # A fresh query sees the current generations of all sources.
            fresh = self.fpage(fed, limit=100)
            self.assertEqual(
                self.ftuples(fresh),
                [
                    ("alpha", "d01", 0, 2, (0, 2)),
                    ("beta", "d01", 0, 2, (0, 2)),
                    ("alpha", "d02", 0, 2, (0, 2)),
                    ("beta", "d02", 0, 2, (0, 2)),
                    ("alpha", "d03", 0, 2, (0, 2)),
                    ("alpha", "d04", 0, 2, (0, 2)),
                    ("beta", "d05", 0, 2, (0, 2)),
                ],
            )
            self.assertEqual(fresh.sources["beta"]["generation"], 2)
            self.assertNotEqual(
                fresh.sources["beta"]["fingerprint"], pinned["beta"]
            )
            self.assertEqual(fresh.sources["alpha"]["fingerprint"], pinned["alpha"])

    def test_first_page_pins_the_generation_the_handle_opened(self):
        paths = self.two_source_setup()
        with self.federation(paths) as fed:
            first = self.fpage(fed, limit=2)
            cursor = first.next_cursor
            beta_fp = first.sources["beta"]["fingerprint"]
            # Publish a new beta generation while the federation stays open.
            self.refresh_beta(paths, [{"op": "delete", "doc_id": "d04"}])
            # The open federation keeps serving the generation it opened.
            pages, flat = self.walk(fed, limit=2, cursor=cursor)
            self.assertEqual(
                [t for t in flat if t[0] == "beta"],
                [("beta", "d02", 0, 2, (0, 2)),
                 ("beta", "d03", 0, 2, (0, 2)),
                 ("beta", "d04", 0, 2, (0, 2))],
            )
            for page in pages:
                self.assertEqual(page.sources["beta"]["fingerprint"], beta_fp)
        # A reopened federation sees the new generation.
        with self.federation(paths) as fed:
            fresh = self.fpage(fed, limit=100)
            self.assertNotIn(
                ("beta", "d04", 0, 2, (0, 2)), self.ftuples(fresh)
            )
            self.assertEqual(fresh.sources["beta"]["generation"], 2)

    def test_old_cursor_works_in_a_fresh_process_after_refresh(self):
        paths = self.two_source_setup()
        manifest_path = self.write_manifest(
            {"alpha": str(paths["alpha"]), "beta": str(paths["beta"])}
        )
        with Federation.from_manifest(manifest_path) as fed:
            first = self.fpage(fed, limit=2)
            cursor = first.next_cursor
            pinned = {n: i["fingerprint"] for n, i in first.sources.items()}
        self.refresh_beta(
            paths,
            [
                {"op": "upsert", "doc_id": "d03", "text": "valve pump valve"},
                {"op": "delete", "doc_id": "d04"},
            ],
        )
        driver = self.tmp / "fed_driver.py"
        driver.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from nearidx import Federation, federated_search, parse_query\n"
            "fed = Federation.from_manifest(sys.argv[1])\n"
            "out, sources = [], []\n"
            "cur = sys.argv[3]\n"
            "while True:\n"
            "    page = federated_search(fed, parse_query(sys.argv[2]), cursor=cur)\n"
            "    sources.append(page.sources)\n"
            "    out += [[w.source_name, w.doc_id, w.start, w.end,\n"
            "             list(w.witness), w.snippet] for w in page.windows]\n"
            "    if not page.has_more:\n"
            "        break\n"
            "    cur = page.next_cursor\n"
            "print(json.dumps({'windows': out, 'sources': sources}))\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-I", str(driver), str(manifest_path), QUERY, cursor],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        got = json.loads(proc.stdout)
        self.assertEqual(
            [tuple(w[:5]) for w in got["windows"]],
            [
                ("alpha", "d02", 0, 2, [0, 2]),
                ("beta", "d02", 0, 2, [0, 2]),
                ("alpha", "d03", 0, 2, [0, 2]),
                ("beta", "d03", 0, 2, [0, 2]),
                ("alpha", "d04", 0, 2, [0, 2]),
                ("beta", "d04", 0, 2, [0, 2]),
            ],
        )
        self.assertEqual(
            [w[5] for w in got["windows"]], [PUMP] * 6
        )
        for page_sources in got["sources"]:
            for name, info in page_sources.items():
                self.assertEqual(info["fingerprint"], pinned[name])

    def test_expired_generation_in_any_source_fails_the_whole_call(self):
        paths = self.two_source_setup()
        with self.federation(paths) as fed:
            cursor = self.fpage(fed, limit=2).next_cursor
        # Two refreshes with retain_old=1 push beta's generation 1 out of
        # retention; alpha is never touched and its pinned generation is
        # still perfectly valid.
        self.refresh_beta(paths, [{"op": "delete", "doc_id": "d04"}], retain_old=1)
        self.refresh_beta(paths, [{"op": "delete", "doc_id": "d03"}], retain_old=1)
        with self.federation(paths) as fed:
            self.assertEqual(fed.handle("beta").generation, 3)
            with self.assertRaises(GenerationExpiredError):
                self.fpage(fed, cursor=cursor)
            # GenerationExpiredError is a CursorError, and no partial page
            # is produced: the exception is the whole outcome.
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=cursor)

    def test_cursor_chain_never_rebinds_to_current_generations(self):
        paths = self.two_source_setup()
        with self.federation(paths) as fed:
            cursor = self.fpage(fed, limit=3).next_cursor
            pinned = {n: i["fingerprint"] for n, i in
                      fed.sources_info().items()}
        self.refresh_beta(paths, [{"op": "delete", "doc_id": "d04"}])
        self.refresh_beta(paths, [{"op": "delete", "doc_id": "d03"}])
        with self.federation(paths) as fed:
            pages, flat = self.walk(fed, limit=3, cursor=cursor)
            self.assertEqual(
                [t for t in flat if t[0] == "beta"],
                [("beta", "d02", 0, 2, (0, 2)),
                 ("beta", "d03", 0, 2, (0, 2)),
                 ("beta", "d04", 0, 2, (0, 2))],
            )
            for page in pages:
                for name, info in page.sources.items():
                    self.assertEqual(info["fingerprint"], pinned[name])


class CursorRejectionTests(TempFederationCase):
    """A cursor is bound to query, page size, source names and identities."""

    def setUp(self):
        super().setUp()
        self.paths = {
            "alpha": self.build([("d01", PUMP), ("d02", PUMP), ("d03", PUMP)],
                                "alpha"),
            "beta": self.build([("d01", PUMP), ("d02", PUMP), ("d03", PUMP)],
                               "beta"),
        }
        with self.federation(self.paths) as fed:
            self.cursor = self.fpage(fed, limit=2).next_cursor
        self.assertIsNotNone(self.cursor)

    def test_changed_source_set_rejected(self):
        # A source dropped.
        with self.federation({"alpha": self.paths["alpha"]}) as fed:
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=self.cursor)
        # A source added.
        extra = dict(self.paths)
        extra["gamma"] = self.build([("d01", PUMP)], "gamma")
        with self.federation(extra) as fed:
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=self.cursor)

    def test_source_name_repointed_to_another_index_rejected(self):
        other = self.build([("d01", PUMP), ("d02", PUMP), ("d03", PUMP)],
                           "other")
        repointed = {"alpha": self.paths["alpha"], "beta": other}
        with self.federation(repointed) as fed:
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=self.cursor)

    def test_different_query_rejected(self):
        with self.federation(self.paths) as fed:
            with self.assertRaises(CursorError):
                self.fpage(fed, "NEAR(3,[valve,valve])", cursor=self.cursor)

    def test_cursor_keeps_its_own_page_size(self):
        with self.federation(self.paths) as fed:
            page = self.fpage(fed, cursor=self.cursor, limit=99)
            self.assertEqual(page.limit, 2)
            self.assertEqual(len(page.windows), 2)

    def test_garbage_cursor_rejected(self):
        with self.federation(self.paths) as fed:
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor="not-a-cursor")

    def test_single_index_cursor_rejected(self):
        with open_index(self.paths["alpha"]) as idx:
            single_cursor = search(idx, parse_query(QUERY), limit=1).next_cursor
        with self.federation(self.paths) as fed:
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=single_cursor)

    def test_federated_cursor_rejected_by_single_index_search(self):
        with open_index(self.paths["alpha"]) as idx:
            with self.assertRaises(CursorError):
                search(idx, parse_query(QUERY), cursor=self.cursor)

    def _tamper(self, mutate):
        pad = "=" * (-len(self.cursor) % 4)
        obj = json.loads(
            base64.urlsafe_b64decode(self.cursor + pad).decode("utf-8")
        )
        mutate(obj)
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def test_tampered_resume_state_rejected(self):
        with self.federation(self.paths) as fed:
            # A skip that exceeds the document's actual window count.
            def blow_up_skip(obj):
                obj["s"][0]["skip"] = 999
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=self._tamper(blow_up_skip))
            # A resume document that is not a candidate in this generation.
            def wrong_doc(obj):
                obj["s"][0]["doc"] = "no-such-doc"
                obj["s"][0]["skip"] = 1
            with self.assertRaises(CursorError):
                self.fpage(fed, cursor=self._tamper(wrong_doc))

    def test_structurally_invalid_cursors_rejected(self):
        with self.federation(self.paths) as fed:
            for bad in (
                {},
                {"v": "nearidx-federated-v1"},
                {"v": "nearidx-federated-v1", "q": "x", "n": 1, "e": 0, "s": []},
                {"v": "nearidx-federated-v1", "q": "x", "n": 0, "e": 0,
                 "s": [{"name": "a", "id": "i", "fp": "f", "doc": None,
                        "skip": 0}]},
                {"v": "nearidx-federated-v1", "q": "x", "n": 1, "e": 0,
                 "s": [{"name": "a", "id": "i", "fp": "f", "doc": None,
                        "skip": 1}]},
            ):
                raw = json.dumps(bad).encode()
                token = base64.urlsafe_b64encode(raw).decode().rstrip("=")
                with self.assertRaises(CursorError, msg=bad):
                    self.fpage(fed, cursor=token)


class ManifestAndSourceFailureTests(TempFederationCase):
    """Bad manifests and bad sources fail the whole call, never partially."""

    def test_missing_index_fails_construction(self):
        good = self.build([("d1", PUMP)], "good")
        with self.assertRaises(FileNotFoundError):
            Federation({"alpha": good, "ghost": self.tmp / "ghost.nearidx"})

    def test_corrupt_index_fails_construction(self):
        good = self.build([("d1", PUMP)], "good")
        broken = self.tmp / "broken.nearidx"
        broken.write_bytes(b"this is not a sqlite database at all")
        with self.assertRaises(IndexFormatError):
            Federation({"alpha": good, "beta": broken})

    def test_one_bad_source_fails_even_if_others_are_fine(self):
        good = self.build([("d1", PUMP)], "good")
        manifest = self.write_manifest(
            {"alpha": str(good), "ghost": "ghost.nearidx"}
        )
        with self.assertRaises(FileNotFoundError):
            Federation.from_manifest(manifest)

    def test_duplicate_source_names_rejected(self):
        one = self.build([("d1", PUMP)], "one")
        two = self.build([("d2", PUMP)], "two")
        # In the JSON file (a naive parser would keep only the last key).
        manifest = self.tmp / "dup.json"
        manifest.write_text(
            '{"alpha": %s, "alpha": %s}'
            % (json.dumps(str(one)), json.dumps(str(two))),
            encoding="utf-8",
        )
        with self.assertRaises(FederationError):
            Federation.from_manifest(manifest)
        # And through the pair-list API.
        with self.assertRaises(FederationError):
            Federation([("alpha", one), ("alpha", two)])

    def test_empty_manifest_rejected(self):
        manifest = self.write_manifest({}, name="empty.json")
        with self.assertRaises(FederationError):
            Federation.from_manifest(manifest)
        with self.assertRaises(FederationError):
            Federation({})

    def test_invalid_manifests_rejected(self):
        one = self.build([("d1", PUMP)], "one")
        cases = {
            "notjson.json": "{not json",
            "list.json": '["alpha", "beta"]',
            "emptyname.json": json.dumps({"": str(one)}),
            "nonstring.json": json.dumps({"alpha": 42}),
            "emptypath.json": json.dumps({"alpha": ""}),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                manifest = self.tmp / name
                manifest.write_text(text, encoding="utf-8")
                with self.assertRaises(FederationError):
                    Federation.from_manifest(manifest)

    def test_missing_manifest_file_rejected(self):
        with self.assertRaises(FederationError):
            Federation.from_manifest(self.tmp / "no-such-manifest.json")

    def test_relative_paths_resolve_against_the_manifest_directory(self):
        one = self.build([("d1", PUMP)], "one")
        manifest = self.write_manifest({"alpha": "one.nearidx"})
        with Federation.from_manifest(manifest) as fed:
            page = self.fpage(fed, limit=10)
            self.assertEqual(self.ftuples(page), [("alpha", "d1", 0, 2, (0, 2))])
            self.assertEqual(
                page.sources["alpha"]["path"], str(self.tmp / "one.nearidx")
            )


class ApiSurfaceTests(TempFederationCase):
    def test_limit_must_be_positive(self):
        paths = {"alpha": self.build([("d1", PUMP)], "alpha")}
        with self.federation(paths) as fed:
            from nearidx import QueryError

            with self.assertRaises(QueryError):
                self.fpage(fed, limit=0)

    def test_closed_federation_rejects_search(self):
        paths = {"alpha": self.build([("d1", PUMP)], "alpha")}
        fed = self.federation(paths)
        fed.close()
        with self.assertRaises(FederationError):
            self.fpage(fed)

    def test_federation_search_method_and_sources_info(self):
        paths = {"alpha": self.build([("d1", PUMP)], "alpha")}
        with self.federation(paths) as fed:
            page = fed.search(parse_query(QUERY), limit=5)
            self.assertEqual(self.ftuples(page), [("alpha", "d1", 0, 2, (0, 2))])
            info = fed.sources_info()
            self.assertEqual(sorted(info), ["alpha"])
            self.assertEqual(info["alpha"]["generation"], 1)
            self.assertEqual(info["alpha"]["path"], str(paths["alpha"]))
            self.assertEqual(fed.source_names, ["alpha"])
            self.assertEqual(fed.source_path("alpha"), str(paths["alpha"]))

    def test_mapping_and_pair_list_are_equivalent(self):
        one = self.build([("d1", PUMP)], "one")
        two = self.build([("d2", PUMP)], "two")
        with Federation({"a": one, "b": two}) as fed_map, Federation(
            [("b", two), ("a", one)]
        ) as fed_pairs:
            self.assertEqual(fed_map.source_names, fed_pairs.source_names)
            page_a = self.fpage(fed_map, limit=10)
            page_b = self.fpage(fed_pairs, limit=10)
            self.assertEqual(self.ftuples(page_a), self.ftuples(page_b))


class StreamingEvidenceTests(TempFederationCase):
    """The first page never collects every hit before delivering."""

    def test_first_page_pulls_only_what_it_needs(self):
        n = 1500
        paths = {
            "alpha": self.build(
                [(f"d{i:04d}", PUMP) for i in range(n)], "alpha"
            ),
            "beta": self.build([("e0", PUMP), ("e1", PUMP)], "beta"),
        }
        with self.federation(paths) as fed:
            page = self.fpage(fed, limit=3)
            self.assertTrue(page.has_more)
            self.assertEqual(
                self.ftuples(page),
                [
                    ("alpha", "d0000", 0, 2, (0, 2)),
                    ("alpha", "d0001", 0, 2, (0, 2)),
                    ("alpha", "d0002", 0, 2, (0, 2)),
                ],
            )
            # The candidate list comes from the inverted table (that is the
            # architecture), but only a handful of documents were solved and
            # only the delivered ones had their text read.  Collecting all
            # 1500 hits before sorting would show up here.
            self.assertEqual(page.trace["alpha"]["candidate_docs"], n)
            self.assertLessEqual(page.trace["alpha"]["docs_solved"], 5)
            self.assertLessEqual(page.trace["beta"]["docs_solved"], 2)
            self.assertEqual(
                page.trace["alpha"]["body_reads"], ["d0000", "d0001", "d0002"]
            )
            self.assertEqual(page.trace["beta"]["body_reads"], [])

            # Continuation pages stay just as bounded: only the boundary
            # document is re-solved, never everything before the page.
            page2 = self.fpage(fed, cursor=page.next_cursor)
            self.assertEqual(
                self.ftuples(page2),
                [
                    ("alpha", "d0003", 0, 2, (0, 2)),
                    ("alpha", "d0004", 0, 2, (0, 2)),
                    ("alpha", "d0005", 0, 2, (0, 2)),
                ],
            )
            self.assertLessEqual(page2.trace["alpha"]["docs_solved"], 6)
            self.assertLessEqual(page2.trace["beta"]["docs_solved"], 2)
            self.assertEqual(
                page2.trace["alpha"]["body_reads"], ["d0003", "d0004", "d0005"]
            )
            self.assertEqual(page2.offset, 3)

    def test_full_walk_over_many_windows_is_exact(self):
        n = 300
        paths = {
            "alpha": self.build([(f"d{i:04d}", PUMP) for i in range(n)], "a"),
            "beta": self.build([(f"d{i:04d}", PUMP) for i in range(n)], "b"),
        }
        with self.federation(paths) as fed:
            pages, flat = self.walk(fed, limit=7)
            self.assertEqual(len(flat), 2 * n)
            self.assertEqual(len(flat), len(set(flat)))
            expected = [
                (name, f"d{i:04d}", 0, 2, (0, 2))
                for i in range(n)
                for name in ("alpha", "beta")
            ]
            self.assertEqual(flat, expected)


class CliFederationTests(TempFederationCase):
    def run_cli(self, *argv):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def setUp(self):
        super().setUp()
        self.paths = {
            "alpha": self.build([("d01", PUMP), ("d03", PUMP)], "alpha"),
            "beta": self.build([("d02", PUMP), ("d03", PUMP)], "beta"),
        }
        self.manifest = self.write_manifest(
            {"alpha": "alpha.nearidx", "beta": "beta.nearidx"}
        )

    def test_fsearch_json_and_cursor_chain(self):
        code, out, err = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--limit", "2", "--json"
        )
        self.assertEqual(code, 0, err)
        page = json.loads(out)
        self.assertEqual(
            [(w["source_name"], w["doc_id"]) for w in page["windows"]],
            [("alpha", "d01"), ("beta", "d02")],
        )
        self.assertTrue(page["has_more"])
        self.assertEqual(page["windows"][0]["snippet"], PUMP)
        self.assertEqual(page["windows"][0]["char_span"], [0, 15])
        self.assertEqual(
            sorted(page["sources"]), ["alpha", "beta"]
        )
        cursor = page["next_cursor"]
        code, out, err = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--cursor", cursor, "--json"
        )
        self.assertEqual(code, 0, err)
        page2 = json.loads(out)
        self.assertEqual(
            [(w["source_name"], w["doc_id"]) for w in page2["windows"]],
            [("alpha", "d03"), ("beta", "d03")],
        )
        self.assertFalse(page2["has_more"])
        self.assertIsNone(page2["next_cursor"])
        self.assertEqual(page2["offset"], 2)

    def test_fsearch_human_output_names_sources(self):
        code, out, err = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--limit", "2"
        )
        self.assertEqual(code, 0, err)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("alpha\td01\t0-2\t"))
        self.assertTrue(lines[1].startswith("beta\td02\t0-2\t"))
        self.assertIn("has_more=True", err)
        self.assertIn("next_cursor", err)

    def test_fsearch_cursor_survives_refresh_of_one_source(self):
        code, out, _ = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--limit", "1", "--json"
        )
        cursor = json.loads(out)["next_cursor"]
        pinned = json.loads(out)["sources"]
        refresh_manifest = self.tmp / "r.jsonl"
        refresh_manifest.write_text(
            json.dumps({"op": "delete", "doc_id": "d03"}) + "\n",
            encoding="utf-8",
        )
        code, _out, err = self.run_cli(
            "refresh", str(self.paths["beta"]), str(refresh_manifest)
        )
        self.assertEqual(code, 0, err)
        # The old cursor still reads beta's original generation.  The cursor
        # carries its own page size (1), so follow the chain to the end.
        seen, sources_seen = [], []
        while cursor is not None:
            code, out, err = self.run_cli(
                "fsearch", str(self.manifest), QUERY, "--cursor", cursor,
                "--json",
            )
            self.assertEqual(code, 0, err)
            page = json.loads(out)
            self.assertEqual(page["limit"], 1)
            seen.extend(
                (w["source_name"], w["doc_id"]) for w in page["windows"]
            )
            sources_seen.append(page["sources"])
            cursor = page["next_cursor"]
        self.assertEqual(
            seen,
            [("beta", "d02"), ("alpha", "d03"), ("beta", "d03")],
        )
        for page_sources in sources_seen:
            for name in ("alpha", "beta"):
                self.assertEqual(
                    page_sources[name]["fingerprint"],
                    pinned[name]["fingerprint"],
                )
        # A fresh query sees beta's current generation (d03 deleted).
        code, out, err = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--json"
        )
        self.assertEqual(code, 0, err)
        fresh = json.loads(out)
        self.assertEqual(
            [(w["source_name"], w["doc_id"]) for w in fresh["windows"]],
            [("alpha", "d01"), ("beta", "d02"), ("alpha", "d03")],
        )

    def test_fsearch_rejects_changed_manifest(self):
        code, out, _ = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--limit", "1", "--json"
        )
        cursor = json.loads(out)["next_cursor"]
        # Repoint beta at a different index and drop/re-add nothing else.
        other = self.build([("d01", PUMP)], "other")
        self.write_manifest({"alpha": "alpha.nearidx", "beta": str(other)})
        code, _out, err = self.run_cli(
            "fsearch", str(self.manifest), QUERY, "--cursor", cursor
        )
        self.assertEqual(code, 2)
        self.assertIn("query error", err)

    def test_fsearch_rejects_missing_manifest_and_missing_index(self):
        code, _out, err = self.run_cli(
            "fsearch", str(self.tmp / "nope.json"), QUERY
        )
        self.assertEqual(code, 2)
        self.assertIn("manifest error", err)
        manifest = self.write_manifest(
            {"alpha": "alpha.nearidx", "ghost": "ghost.nearidx"},
            name="with-ghost.json",
        )
        code, _out, err = self.run_cli("fsearch", str(manifest), QUERY)
        self.assertEqual(code, 2)
        self.assertIn("index error", err)

    def test_fsearch_rejects_bad_query(self):
        code, _out, err = self.run_cli(
            "fsearch", str(self.manifest), "NEAR(3,[pump)"
        )
        self.assertEqual(code, 2)
        self.assertIn("query error", err)


if __name__ == "__main__":
    unittest.main()
