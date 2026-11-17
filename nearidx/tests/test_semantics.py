"""Tests: main example, semantics, oracle comparison, paging, reopening.

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
    AccessTrace,
    Document,
    Index,
    IndexFormatError,
    QuerySyntaxError,
    StaleSourceError,
    build_index,
    open_index,
    parse_query,
    rebuild_document,
    search,
    select_root_windows,
    text_sha256,
    token_spans,
)
from nearidx.solver import Hit, solve_document
from nearidx.tests import oracle

ROOT = Path(__file__).resolve().parents[2]


def norm_witness(witness):
    """Oracle uses None for unused leaves; the solver uses -1."""
    return tuple(-1 if p is None else p for p in witness)


def norm_windows(windows):
    return [(d, s, e, norm_witness(w)) for d, s, e, w in windows]


def oracle_windows(doc_id, query, text):
    """Oracle root windows, shaped like the solver's tuples."""
    return [(doc_id, s, e, w) for s, e, w in oracle.root_windows(parse_query(query), text)]


class TempIndexCase(unittest.TestCase):
    """Helpers for building a throwaway index per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, docs, name="t", overwrite=True):
        path = self.tmp / f"{name}.nearidx"
        build_index(path, docs, name=name, overwrite=overwrite)
        return path

    def wins(self, index, query, **kwargs):
        page = search(index, parse_query(query), **kwargs)
        return [(w.doc_id, w.start, w.end, w.witness, w.snippet) for w in page.windows]

    def win_tuples(self, index, query, **kwargs):
        page = search(index, parse_query(query), **kwargs)
        return [(w.doc_id, w.start, w.end, w.witness) for w in page.windows]


class MainExampleTests(TempIndexCase):
    """The specification's pump/valve example."""

    def test_near_two_pumps_is_empty_but_near_three_uses_both(self):
        path = self.build([("d", "pump valve pump")])
        with open_index(path) as idx:
            self.assertEqual(self.win_tuples(idx, "NEAR(2,[pump,pump])"), [])
            self.assertEqual(
                self.win_tuples(idx, "NEAR(3,[pump,pump])"),
                [("d", 0, 2, (0, 2))],
            )
            # 0 is never reused for both children: distinct leaf witnesses.
            page = search(idx, parse_query("NEAR(3,[pump,pump])"))
            self.assertEqual(page.windows[0].leaves[0].positions, [0])
            self.assertEqual(page.windows[0].leaves[1].positions, [2])

    def test_seq_zero_pump_valve(self):
        path = self.build([("d", "pump valve pump")])
        with open_index(path) as idx:
            self.assertEqual(
                self.win_tuples(idx, "SEQ(0,[pump,valve])"),
                [("d", 0, 1, (0, 1))],
            )

    def test_seq_gap_counts_unoccupied_positions(self):
        path = self.build([("d", "a b c d")])
        with open_index(path) as idx:
            self.assertEqual(self.win_tuples(idx, "SEQ(0,[a,c])"), [])
            self.assertEqual(self.win_tuples(idx, "SEQ(1,[a,c])"), [("d", 0, 2, (0, 2))])
            self.assertEqual(self.win_tuples(idx, "SEQ(2,[a,d])"), [("d", 0, 3, (0, 3))])
            self.assertEqual(self.win_tuples(idx, "SEQ(1,[a,d])"), [])

    def test_seq_has_no_self_overlap(self):
        # The second child may not start at or before the first child's end.
        path = self.build([("d", "x a y")])
        with open_index(path) as idx:
            self.assertEqual(self.win_tuples(idx, "SEQ(5,[x,x])"), [])

    def test_or_witness_marks_unused_leaves_as_minus_one(self):
        path = self.build([("d", "pump valve")])
        with open_index(path) as idx:
            self.assertEqual(
                self.win_tuples(idx, "OR(pump,valve)"),
                [
                    ("d", 0, 0, (0, -1)),
                    ("d", 1, 1, (-1, 1)),
                ],
            )


class SemanticsTests(TempIndexCase):
    def test_near_does_not_require_child_order(self):
        path = self.build([("d", "valve pump")])
        with open_index(path) as idx:
            self.assertEqual(
                self.win_tuples(idx, "NEAR(2,[pump,valve])"),
                [("d", 0, 1, (1, 0))],
            )

    def test_near_children_may_have_overlapping_intervals(self):
        # left = pump at 0 (+ valve at 2 => interval 0..2)
        # right = valve at 2 (+ pump at 3 => interval 2..3)
        # Intervals overlap at 2, but used positions {0,2} and {2,3} share 2,
        # so this must fail; the inner distinct assignment {0,2}/{1,3} works.
        text = "pump valve pump seal"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            got = self.win_tuples(
                idx, "NEAR(4,[SEQ(0,[pump,valve]),SEQ(0,[pump,seal])])"
            )
            self.assertEqual(got, [("d", 0, 3, (0, 1, 2, 3))])
            oracle_hits = oracle.all_hits(
                parse_query("NEAR(4,[SEQ(0,[pump,valve]),SEQ(0,[pump,seal])])"),
                text,
            )
            self.assertIn((0, 3, (0, 1, 2, 3)), oracle_hits)

    def test_near_rejects_reused_positions_across_children(self):
        # Both children can only be pump@1, so they collide.
        path = self.build([("d", "a pump b")])
        with open_index(path) as idx:
            self.assertEqual(self.win_tuples(idx, "NEAR(1,[pump,pump])"), [])

    def test_overlapping_child_intervals_disjoint_positions(self):
        # Child 1 = SEQ(1,[a,c]) uses {0,2}, hull (0,2).  Child 2 = b uses
        # {1}, hull (1,1).  The intervals overlap at position 1, but the used
        # positions {0,2} and {1} are disjoint, so this is a valid NEAR hit.
        # Treating "overlapping intervals" as a conflict would lose it.
        text = "a b c"
        q = "NEAR(3,[SEQ(1,[a,c]), b])"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            self.assertEqual(got, [("d", 0, 2, (0, 2, 1))])
            self.assertEqual(got, oracle_windows("d", q, text))

    def test_overlapping_intervals_with_disjoint_positions(self):
        # a@0,b@2 -> interval (0,2) and b@1,c@2 -> interval (1,2).  The
        # intervals overlap, the used positions {0,2} and {1} do not.
        text = "a b c"
        path = self.build([("d", text)])
        q = "NEAR(3,[SEQ(0,[a,b]), SEQ(0,[b,c])])"
        with open_index(path) as idx:
            # second child needs b,c adjacent: b@1,c@2. first needs a@0,b@1,
            # which collides on b@1. No solution here, by construction.
            self.assertEqual(self.win_tuples(idx, q), [])
            self.assertEqual(oracle_windows("d", q, text), [])

        text = "a x b c"
        path2 = self.build([("d", text)], name="t2")
        q = "NEAR(4,[SEQ(0,[a,x]), SEQ(0,[b,c])])"
        with open_index(path2) as idx:
            got = self.win_tuples(idx, q)
            self.assertEqual(got, [("d", 0, 3, (0, 1, 2, 3))])

    def test_near_width_bounds_the_hull_not_each_interval(self):
        text = "a b c d e"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            # children at 0,1,2,3 -> hull width 4
            q = "NEAR(4,[a,b,c,d])"
            self.assertEqual(self.win_tuples(idx, q), [("d", 0, 3, (0, 1, 2, 3))])
            q3 = "NEAR(3,[a,b,c,d])"
            self.assertEqual(self.win_tuples(idx, q3), [])


class RepetitionTests(TempIndexCase):
    def test_repeated_query_token_needs_distinct_positions(self):
        path = self.build([("d", "term term term")])
        with open_index(path) as idx:
            self.assertEqual(
                self.win_tuples(idx, "NEAR(2,[term,term])"),
                [("d", 0, 1, (0, 1)), ("d", 1, 2, (1, 2))],
            )

    def test_repeated_subtree_needs_distinct_positions(self):
        # Two occurrences of the *same* NEAR subtree.  With "a b a b" the only
        # hull-4 solution is (0,1,2,3): reused positions are rejected, so the
        # two subtrees cannot both take a@0,b@1.
        text = "a b a b"
        path = self.build([("d", text)])
        q = "NEAR(4,[NEAR(2,[a,b]), NEAR(2,[a,b])])"
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            self.assertEqual(got, oracle_windows("d", q, text))
            self.assertEqual(got, [("d", 0, 3, (0, 1, 2, 3))])

        # A repeated subtree under OR: both OR branches are fully used, so the
        # witness assigns four distinct positions.
        text = "a q w b"
        path2 = self.build([("d", text)], name="t2")
        q2 = "NEAR(4,[OR(a,b), OR(a,b)])"
        with open_index(path2) as idx:
            got = self.win_tuples(idx, q2)
            self.assertEqual(got, oracle_windows("d", q2, text))
            self.assertEqual(got, [("d", 0, 3, (-1, 3, 0, -1))])
            used = [p for p in got[0][3] if p >= 0]
            self.assertEqual(len(used), len(set(used)))


class RootWindowTests(TempIndexCase):
    def test_strictly_contained_window_is_removed(self):
        # pump@0 valve@1, pump@2 valve@1
        text = "pump valve pump"
        path = self.build([("d", text)])
        q = "OR(SEQ(0,[pump,valve]), pump)"
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            # (0,1) and (2,2) survive; nothing strictly contains them here.
            # (0,1) from SEQ strictly contains the OR's bare pump at (0,0),
            # so the larger (0,1) window is the one removed.
            self.assertEqual(
                got,
                [
                    ("d", 0, 0, (-1, -1, 0)),
                    ("d", 2, 2, (-1, -1, 2)),
                ],
            )
            expected = oracle_windows("d", q, text)
            self.assertEqual(got, expected)

    def test_minimum_windows_are_per_root_and_containment_is_strict(self):
        text = "a b a b a"
        path = self.build([("d", text)])
        q = "NEAR(5,[a,b])"
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            expected = oracle_windows("d", q, text)
            self.assertEqual(got, expected)
            # a@0 with b@1 is contained in a@0..a@2 etc.  Only minimal survive.
            for doc, start, end, witness in got:
                self.assertFalse(
                    any(
                        s2 >= start and e2 <= end and (s2, e2) != (start, end)
                        for _, s2, e2, _ in expected
                    )
                )

    def test_equal_intervals_keep_one_witness_by_leaf_order(self):
        # OR: a@0 OR b@0 both yield interval (0,0); the lexicographic witness
        # (0,-1) must win over (-1,0).
        text = "a b"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            got = self.win_tuples(idx, "OR(a,b)")
            self.assertEqual(got[0], ("d", 0, 0, (0, -1)))
            expected = oracle_windows("d", "OR(a,b)", text)
            self.assertEqual(got, expected)

    def test_witness_tie_break_uses_unused_minus_one(self):
        # Two ways to hit interval (0,1): leaf order prefers 0 over 1.
        text = "a a b"
        path = self.build([("d", text)])
        q = "NEAR(2,[a,OR(a,b)])"
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            expected = oracle_windows("d", q, text)
            self.assertEqual(got, expected)
            self.assertIn(("d", 0, 1, (0, 1, -1)), got)
            # (1,0,-1) is the same interval with a lexicographically larger
            # witness and must be dropped.
            self.assertNotIn(("d", 0, 1, (1, 0, -1)), got)


class NestedWitnessTests(TempIndexCase):
    """A sibling conflict inside one interval must not hide the other witness."""

    def test_sibling_conflict_keeps_alternative_witness(self):
        # q = NEAR(3,[OR(a, SEQ(0,[b,c])), b])
        #   This is constructed so that leaf1=b and leaf0=a conflict at some
        #   positions but the SEq branch uses different positions.
        text = "a b c b"
        path = self.build([("d", text)])
        q = "NEAR(4,[OR(a, SEQ(0,[b,c])), b])"
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            expected = oracle_windows("d", q, text)
            self.assertEqual(got, expected)
            # used positions must be pairwise distinct per witness
            for _doc, _s, _e, witness in got:
                used = [p for p in witness if p >= 0]
                self.assertEqual(len(used), len(set(used)))

    def test_internal_dedup_does_not_collapse_distinct_witnesses(self):
        text = "a b a b"
        path = self.build([("d", text)])
        q = "OR(NEAR(2,[a,b]), NEAR(2,[a,b]))"
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            expected = oracle_windows("d", q, text)
            self.assertEqual(got, expected)

    def test_internal_same_interval_keeps_both_witnesses(self):
        # Inner NEAR produces the *same* interval (0,3) with two different
        # used-position sets: {0,1,3} (OR takes z@1) and {0,2,3} (OR takes
        # y@2).  The root's sibling z can only use position 1, so the first
        # witness conflicts and the second succeeds.  Deduplicating the inner
        # node by interval alone would keep only one and drop the root hit.
        text = "a z y b"
        q = "NEAR(4,[NEAR(4,[a,OR(z,y),b]), z])"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            self.assertEqual(got, oracle_windows("d", q, text))
            self.assertEqual(got, [("d", 0, 3, (0, -1, 2, 3, 1))])
            used = sorted(p for p in got[0][3] if p >= 0)
            self.assertEqual(used, [0, 1, 2, 3])

    def test_internal_shortest_interval_cannot_be_the_only_one_kept(self):
        # Inner OR has hits (0,1) using {0,1} and (0,2) using {0,2}.  The
        # root sibling b sits at position 1, so the shortest internal interval
        # conflicts and must not be the only witness retained.
        text = "a b c"
        q = "NEAR(3,[OR(SEQ(0,[a,b]), SEQ(1,[a,c])), b])"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            self.assertEqual(got, oracle_windows("d", q, text))
            self.assertEqual(got, [("d", 0, 2, (-1, -1, 0, 2, 1))])

    def test_sequence_of_or_alternatives(self):
        text = "alpha x beta y gamma"
        q = "SEQ(2,[OR(alpha,beta), OR(x,y), gamma])"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            got = self.win_tuples(idx, q)
            expected = oracle_windows("d", q, text)
            self.assertEqual(got, expected)
            self.assertTrue(got)


class CaseFoldingTests(TempIndexCase):
    def test_full_casefold_and_original_slice(self):
        text = "STRASSE straße ÄRGER ärger"
        path = self.build([("d", text)])
        self.assertEqual(
            [t.norm for t in token_spans(text)],
            ["strasse", "strasse", "ärger", "ärger"],
        )
        with open_index(path) as idx:
            got = self.win_tuples(idx, "NEAR(3,[straße,ÄRGER])", limit=10)
            expected = oracle_windows("d", "NEAR(3,[straße,ÄRGER])", text)
            self.assertEqual(got, expected)
            # strasse@0 + ärger@2 is the wider window and is removed because
            # it strictly contains strasse@1 + ärger@2.
            self.assertEqual(got, [("d", 1, 2, (1, 2))])
            windows = search(idx, parse_query("NEAR(2,[strasse,ärger])"), limit=10).windows
            self.assertEqual(
                [(w.start, w.end, w.witness, w.snippet) for w in windows],
                [(1, 2, (1, 2), "straße ÄRGER")],
            )

    def test_multibyte_char_and_byte_spans(self):
        text = "😀 ÄRGER pump"
        path = self.build([("d", text)])
        with open_index(path) as idx:
            window = search(idx, parse_query("pump")).windows[0]
            # chars: emoji=0, space=1, "ÄRGER"=2..7, space=7, "pump"=8..12
            self.assertEqual((window.char_start, window.char_end), (8, 12))
            # bytes: emoji=4, space=1, "ÄRGER"=6, space=1 -> pump at 12..16
            self.assertEqual((window.byte_start, window.byte_end), (12, 16))
            self.assertEqual(window.snippet, "pump")
            window = search(idx, parse_query("ärger")).windows[0]
            self.assertEqual(window.snippet, "ÄRGER")
            self.assertEqual((window.char_start, window.char_end), (2, 7))
            self.assertEqual((window.byte_start, window.byte_end), (5, 11))

    def test_query_token_that_is_not_exactly_one_token_is_rejected(self):
        with self.assertRaises(QuerySyntaxError):
            parse_query("a-b")
        with self.assertRaises(QuerySyntaxError):
            parse_query("a b")
        with self.assertRaises(QuerySyntaxError):
            parse_query("a_b c")


class EmptyAndMultibyteDocTests(TempIndexCase):
    def test_empty_document_and_separator_only_document(self):
        path = self.build(
            [("empty", ""), ("seps", " !!! ??? "), ("real", "pump")]
        )
        with open_index(path) as idx:
            self.assertEqual(self.win_tuples(idx, "pump"), [("real", 0, 0, (0,))])
            # Empty docs simply have no postings; the index is valid.
            self.assertEqual(idx.verify(full=True), [])
            meta = idx.doc_meta("empty")
            self.assertEqual(meta["token_count"], 0)
            self.assertEqual(idx.text("empty"), "")

    def test_snippet_is_cut_not_rejoined(self):
        text = "این  متن\tاست pump"
        path = self.build([("f", text)])
        with open_index(path) as idx:
            window = search(idx, parse_query("pump")).windows[0]
            self.assertEqual(window.snippet, "pump")
            full = search(idx, parse_query("SEQ(0,[این,متن])")).windows[0]
            self.assertEqual(full.snippet, "این  متن")
            self.assertIn("\t", search(idx, parse_query("SEQ(0,[متن,است])")).windows[0].snippet)


class SolverOracleTests(TempIndexCase):
    """Exhaustive assignment oracle on tiny texts and random trees."""

    TEXTS = [
        "a b c",
        "a a b",
        "a b a b",
        "x y x y x",
        "a b c a b c",
        "pump valve pump seal pump",
        "a a a",
    ]
    QUERIES = [
        "a",
        "OR(a, b)",
        "NEAR(1,[a,b])",
        "NEAR(2,[a,b])",
        "NEAR(3,[a,a])",
        "NEAR(4,[a,OR(a,b)])",
        "NEAR(3,[OR(a,b), OR(a,b)])",
        "NEAR(3,[SEQ(0,[a,b]), a])",
        "SEQ(0,[a,b])",
        "SEQ(1,[a,c])",
        "SEQ(2,[a,OR(b,c)])",
        "OR(SEQ(0,[a,b]), NEAR(2,[a,b]))",
        "NEAR(5,[OR(a,b), OR(a,b), a])",
        "NEAR(3,[NEAR(2,[a,b]), a])",
        "NEAR(2,[a,SEQ(0,[b,c])])",
        "SEQ(1,[a, NEAR(2,[b,c])])",
        "NEAR(4,[SEQ(0,[a,b]), SEQ(0,[a,b])])",
        "OR(a, a)",
        "SEQ(0,[a,a])",
        "NEAR(2,[SEQ(0,[a,a]), a])",
    ]

    def test_every_query_matches_oracle_on_every_text(self):
        docs = [(f"d{i}", text) for i, text in enumerate(self.TEXTS)]
        path = self.build(docs)
        with open_index(path) as idx:
            for query in self.QUERIES:
                root = parse_query(query)
                got_all = self.win_tuples(idx, query, limit=10_000)
                expected_all = []
                for doc_id, text in docs:
                    for start, end, witness in oracle.root_windows(root, text):
                        expected_all.append((doc_id, start, end, witness))
                self.assertEqual(got_all, expected_all, msg=query)

    def test_raw_solver_hits_match_oracle_assignments(self):
        for text in self.TEXTS:
            toks = token_spans(text)
            postings: dict = {}
            for t in toks:
                postings.setdefault(t.norm, []).append(
                    (t.pos, t.char_start, t.char_end, t.byte_start, t.byte_end)
                )
            for query in self.QUERIES:
                root = parse_query(query)
                hits = solve_document(root, postings)
                converted = set()
                for hit in hits:
                    w = [-1] * len(oracle._leaves_in(root))
                    for leaf_id, pos in hit.slots:
                        w[leaf_id] = pos
                    used = [p for p in w if p >= 0]
                    converted.add((min(used), max(used), tuple(w)))
                expected = set(oracle.all_hits(root, text))
                self.assertEqual(converted, expected, msg=(query, text))


class RandomisedOracleTests(TempIndexCase):
    """Deterministic random trees and texts against the brute-force oracle."""

    TEXT_ALPHABET = ["a", "b", "c", "a b", "  "]

    def _random_text(self, rng):
        parts = []
        for _ in range(rng.randint(1, 6)):
            parts.append(rng.choice(["a", "b", "c"]))
            parts.append(rng.choice([" ", "  ", "_", "!", "😀 "]))
        return "".join(parts)

    def _random_tree(self, rng, depth=0, leaves=None):
        from nearidx.query import NearNode, OrNode, SeqNode, TokenNode

        # Keep the exhaustive oracle tractable: at most five leaves.
        room = 6 - leaves[0]
        if depth >= 2 or room <= 1 or rng.random() < 0.4:
            leaves[0] += 1
            return TokenNode(token=rng.choice("abc"), spelling="x")
        kind = rng.choice(["OR", "SEQ", "NEAR"])
        count = min(room, rng.choice([2, 2, 3]))
        children = []
        for _ in range(count):
            children.append(self._random_tree(rng, depth + 1, leaves))
        if kind == "OR":
            return OrNode(children=children)
        if kind == "SEQ":
            return SeqNode(gap=rng.randint(0, 3), children=children)
        return NearNode(near_width=rng.randint(1, 4), children=children)

    def test_random_trees_match_oracle(self):
        import random

        rng = random.Random(20261117)
        for trial in range(150):
            text = self._random_text(rng)
            root = self._random_tree(rng, leaves=[0])
            from nearidx.query import _number_leaves

            _number_leaves(root)
            toks = token_spans(text)
            postings: dict = {}
            for t in toks:
                postings.setdefault(t.norm, []).append(
                    (t.pos, t.char_start, t.char_end, t.byte_start, t.byte_end)
                )
            hits = solve_document(root, postings)
            converted = set()
            for hit in hits:
                w = [-1] * len(oracle._leaves_in(root))
                for leaf_id, pos in hit.slots:
                    w[leaf_id] = pos
                used = [p for p in w if p >= 0]
                if not used:
                    continue
                converted.add((min(used), max(used), tuple(w)))
            expected = set(oracle.all_hits(root, text))
            self.assertEqual(converted, expected, msg=(trial, text))
            kept = select_root_windows(hits, len(oracle._leaves_in(root)))
            self.assertEqual(
                [(k[0], k[1], k[2]) for k in kept],
                list(oracle.root_windows(root, text)),
                msg=(trial, text),
            )


class MultiDocCandidateTests(TempIndexCase):
    def test_irrelevant_docs_are_never_solved_or_read(self):
        docs = [("relevant", "pump valve pump")]
        docs += [(f"noise{i:03d}", f"unrelated filler words number {i}") for i in range(300)]
        path = self.build(docs)
        trace = AccessTrace()
        with open_index(path) as idx:
            page = search(idx, parse_query("NEAR(3,[pump,pump])"), trace=trace)
            self.assertEqual(len(page.windows), 1)
            self.assertEqual(page.trace["candidate_docs"], 1)
            self.assertEqual(page.trace["docs_solved"], 1)
            self.assertEqual(page.trace["body_reads"], ["relevant"])
            self.assertGreaterEqual(page.trace["missing_token_docs"], 300)

    def test_body_reads_only_for_docs_with_delivered_windows(self):
        docs = [(f"d{i:02d}", "alpha beta") for i in range(10)]
        path = self.build(docs)
        with open_index(path) as idx:
            page = search(idx, parse_query("NEAR(2,[alpha,beta])"), limit=2)
            self.assertEqual(len(page.windows), 2)
            self.assertEqual(
                [w.doc_id for w in page.windows], ["d00", "d01"]
            )
            self.assertEqual(page.trace["body_reads"], ["d00", "d01"])
            # Docs after the page are never solved, and bodies are fetched
            # only for documents that actually contribute a delivered window.
            self.assertEqual(page.trace["docs_solved"], 3)
            self.assertNotIn("d02", page.trace["body_reads"])
            page = search(idx, parse_query("NEAR(2,[alpha,beta])"), limit=1,
                          offset=1)
            self.assertEqual([w.doc_id for w in page.windows], ["d01"])
            self.assertEqual(page.trace["body_reads"], ["d01"])

    def test_extra_document_solved_for_has_more_is_not_body_read(self):
        docs = [("d0", "alpha beta"), ("d1", "alpha beta"), ("d2", "alpha beta")]
        path = self.build(docs)
        with open_index(path) as idx:
            page = search(idx, parse_query("NEAR(2,[alpha,beta])"), limit=1)
            self.assertTrue(page.has_more)
            self.assertEqual([w.doc_id for w in page.windows], ["d0"])
            self.assertEqual(page.trace["body_reads"], ["d0"])
            self.assertEqual(page.trace["docs_solved"], 2)

    def test_no_required_token_no_body_read(self):
        docs = [(f"d{i}", f"word{i}") for i in range(5)]
        path = self.build(docs)
        with open_index(path) as idx:
            page = search(idx, parse_query("NEAR(2,[absent,word1])"))
            self.assertEqual(page.windows, [])
            self.assertEqual(page.trace["body_reads"], [])
            self.assertEqual(page.trace["candidate_docs"], 0)


class PaginationTests(TempIndexCase):
    def _all_pages(self, idx, query, limit):
        pages = []
        cursor = None
        while True:
            page = search(idx, parse_query(query), limit=limit, cursor=cursor)
            pages.append(page)
            if not page.has_more:
                break
            cursor = page.next_cursor
            self.assertIsNotNone(cursor)
        return pages

    def test_tiny_pages_are_a_stable_partition(self):
        docs = [(f"d{i:02d}", "pump valve pump pump") for i in range(7)]
        docs.append(("z", "pump"))
        path = self.build(docs)
        with open_index(path) as idx:
            q = "OR(pump, valve)"
            full = self.win_tuples(idx, q, limit=10_000)
            for limit in (1, 2, 3):
                pages = self._all_pages(idx, q, limit)
                flat = [
                    (w.doc_id, w.start, w.end, w.witness)
                    for page in pages
                    for w in page.windows
                ]
                self.assertEqual(flat, full, msg=f"limit={limit}")
                self.assertEqual(len(flat), len(set(flat)))
                for page in pages[:-1]:
                    self.assertEqual(len(page.windows), limit)
                self.assertLessEqual(len(pages[-1].windows), limit)
                # each page starts exactly where the previous ended
                offsets = [p.offset for p in pages]
                self.assertEqual(offsets, list(range(0, len(full), limit)))

    def test_cursor_rejects_other_query_and_rebuild(self):
        path = self.build([("d", "pump valve pump"), ("e", "pump")])
        with open_index(path) as idx:
            page = search(idx, parse_query("OR(pump,valve)"), limit=1)
            self.assertTrue(page.has_more)
            with self.assertRaises(Exception):
                search(idx, parse_query("OR(valve,pump)"), cursor=page.next_cursor)
        rebuild_document(path, "d", "pump valve pump valve pump")
        with open_index(path) as idx:
            with self.assertRaises(Exception):
                search(idx, parse_query("OR(pump,valve)"), cursor=page.next_cursor)

    def test_offset_and_limit_consistency(self):
        docs = [(f"d{i}", "a b") for i in range(6)]
        path = self.build(docs)
        with open_index(path) as idx:
            q = "OR(a,b)"
            full = self.win_tuples(idx, q, limit=100)
            for offset in (0, 1, 3):
                page = search(idx, parse_query(q), limit=2, offset=offset)
                self.assertEqual(
                    [(w.doc_id, w.start, w.end, w.witness) for w in page.windows],
                    full[offset : offset + 2],
                )


class ReopenTests(TempIndexCase):
    def test_index_reopens_in_a_fresh_process(self):
        docs = [(f"d{i}", "alpha beta gamma") for i in range(3)]
        path = self.build(docs)
        with open_index(path) as idx:
            expected = self.win_tuples(idx, "NEAR(2,[alpha,beta])")
        driver = self.tmp / "reopen_driver.py"
        driver.write_text(
            "import json, sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from nearidx import open_index, parse_query, search\n"
            "idx = open_index(sys.argv[1])\n"
            "page = search(idx, parse_query(sys.argv[2]), limit=50)\n"
            "print(json.dumps([[w.doc_id, w.start, w.end, list(w.witness),\n"
            "                   w.char_start, w.char_end, w.byte_start,\n"
            "                   w.byte_end, w.snippet] for w in page.windows],\n"
            "                  ensure_ascii=False))\n",
            encoding="utf-8",
        )
        out = subprocess.run(
            [sys.executable, "-I", str(driver), str(path),
             "NEAR(2,[alpha,beta])"],
            cwd=str(self.tmp),
            capture_output=True,
            text=True,
            check=True,
        )
        got = json.loads(out.stdout)
        self.assertEqual(
            [(g[0], g[1], g[2], tuple(g[3])) for g in got],
            expected,
        )

    def test_verify_detects_tampered_text(self):
        path = self.build([("d", "pump valve")])
        conn = sqlite3.connect(path)
        conn.execute("UPDATE documents SET text = 'pump pump valve' WHERE doc_id='d'")
        conn.commit()
        conn.close()
        with open_index(path, verify=False) as idx:
            problems = idx.verify(full=True)
            self.assertTrue(any("sha256" in p for p in problems))
            self.assertTrue(any("postings" in p for p in problems))

    def test_stale_source_is_reported_not_silently_used(self):
        src = self.tmp / "doc.txt"
        src.write_text("pump valve", encoding="utf-8")
        docs = [Document.from_file("d", src)]
        path = self.tmp / "src.nearidx"
        build_index(path, docs, overwrite=True)
        src.write_text("pump ignored valve", encoding="utf-8")
        with open_index(path) as idx:
            with self.assertRaises(StaleSourceError):
                idx.check_source("d")
            problems = idx.verify(full=True)
            self.assertTrue(any("source file changed" in p for p in problems))

    def test_rebuild_rebinds_source_and_clears_stale_report(self):
        src = self.tmp / "doc.txt"
        src.write_text("pump valve", encoding="utf-8")
        path = self.tmp / "src.nearidx"
        build_index(path, [Document.from_file("d", src)], overwrite=True)
        src.write_text("pump pump valve", encoding="utf-8")
        with open_index(path) as idx:
            with self.assertRaises(StaleSourceError):
                idx.check_source("d")
        rebuild_document(
            path,
            "d",
            src.read_text(encoding="utf-8"),
            source_path=str(src.resolve()),
            source_sha256=text_sha256(src.read_text(encoding="utf-8")),
        )
        with open_index(path) as idx:
            idx.check_source("d")  # no longer stale
            self.assertEqual(idx.verify(full=True, check_sources=True), [])
            self.assertEqual(
                self.win_tuples(idx, "SEQ(0,[pump,valve])"),
                [("d", 1, 2, (1, 2))],
            )

    def test_rebuild_changes_hits_and_fingerprint(self):
        path = self.build([("d", "pump valve")])
        with open_index(path) as idx:
            before_fp = idx.fingerprint
            self.assertEqual(self.win_tuples(idx, "SEQ(0,[pump,valve])"), [("d", 0, 1, (0, 1))])
        new_fp = rebuild_document(path, "d", "pump pump valve")
        self.assertNotEqual(before_fp, new_fp)
        with open_index(path) as idx:
            self.assertEqual(idx.fingerprint, new_fp)
            self.assertEqual(
                self.win_tuples(idx, "SEQ(0,[pump,valve])"),
                [("d", 1, 2, (1, 2))],
            )
            self.assertEqual(idx.verify(full=True), [])


class ParserTests(unittest.TestCase):
    def test_bad_queries(self):
        bad = [
            "",
            "   ",
            "OR(a)",
            "NEAR(2,[a])",
            "SEQ(0,[a,b]",
            "NEAR(0,[a,b])",
            "SEQ(-1,[a,b])",
            "FOO(a,b)",
            "NEAR(2,[a,b]) trailing",
            "a-b",
        ]
        for q in bad:
            with self.assertRaises(QuerySyntaxError, msg=q):
                parse_query(q)

    def test_leaf_numbering_is_query_preorder(self):
        root = parse_query("NEAR(3,[c, OR(a, b), SEQ(0,[d, e])])")
        leaves = []
        from nearidx.query import iter_leaves

        for leaf in iter_leaves(root):
            leaves.append((leaf.leaf, leaf.token))
        self.assertEqual(
            leaves, [(0, "c"), (1, "a"), (2, "b"), (3, "d"), (4, "e")]
        )


class RootSelectionUnitTests(unittest.TestCase):
    def test_containment_is_strict_and_per_interval(self):
        hits = [
            Hit(slots=((0, 0),), used=frozenset((0,)), start=0, end=0),
            Hit(slots=((0, 0), (1, 1)), used=frozenset((0, 1)), start=0, end=1),
            Hit(slots=((0, 0), (1, 2)), used=frozenset((0, 2)), start=0, end=2),
            Hit(slots=((0, 3), (1, 3)), used=frozenset((3,)), start=3, end=3),
        ]
        kept = select_root_windows(hits, total_leaves=2)
        self.assertEqual([(k[0], k[1]) for k in kept], [(0, 0), (3, 3)])

    def test_linear_selection_matches_naive_containment(self):
        import random

        rng = random.Random(7)
        for _trial in range(200):
            leaves = rng.randint(1, 4)
            hits = []
            for _ in range(rng.randint(1, 20)):
                positions = sorted(
                    {rng.randint(0, 8) for _ in range(rng.randint(1, leaves))}
                )
                slots = tuple(enumerate(positions))
                hits.append(
                    Hit(
                        slots=slots,
                        used=frozenset(positions),
                        start=positions[0],
                        end=positions[-1],
                    )
                )
            got = [(k[0], k[1], k[2]) for k in select_root_windows(hits, leaves)]
            # naive O(n^2) reference over the same interval set
            best = {}
            for h in hits:
                w = [-1] * leaves
                for leaf_id, pos in h.slots:
                    w[leaf_id] = pos
                key = (h.start, h.end)
                if key not in best or tuple(w) < best[key]:
                    best[key] = tuple(w)
            items = sorted((s, e, w) for (s, e), w in best.items())
            expected = []
            for i, (s, e, w) in enumerate(items):
                if not any(
                    i != j and s2 >= s and e2 <= e
                    for j, (s2, e2, _) in enumerate(items)
                ):
                    expected.append((s, e, w))
            self.assertEqual(got, expected)

    def test_equal_intervals_choose_one_witness(self):
        hits = [
            Hit(slots=((0, 1),), used=frozenset((1,)), start=1, end=1),
            Hit(slots=((0, 0),), used=frozenset((0,)), start=0, end=0),
            Hit(slots=((0, 0),), used=frozenset((0,)), start=0, end=0),
        ]
        kept = select_root_windows(hits, total_leaves=1)
        self.assertEqual([(k[0], k[1], k[2]) for k in kept],
                         [(0, 0, (0,)), (1, 1, (1,))])


if __name__ == "__main__":
    unittest.main()
