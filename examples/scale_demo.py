"""Scale example: mostly irrelevant documents, with access evidence recorded.

Builds N irrelevant docs plus a few relevant ones, then runs pinned queries and
prints what the solver actually touched (inverted-table candidates, solved
documents, body reads).  It also demonstrates tiny-page pagination.

Usage: python examples/scale_demo.py [N]
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nearidx import (  # noqa: E402
    AccessTrace,
    build_index,
    open_index,
    parse_query,
    search,
)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    tmp = Path(tempfile.mkdtemp(prefix="nearidx-scale-"))
    path = tmp / "scale.nearidx"
    docs = []
    for i in range(n):
        docs.append(
            (
                f"noise-{i:05d}",
                f"report {i} mentions filler item {i * 7 % 97} and nothing else",
            )
        )
    docs.append(("rel-0001", "pump valve pump and then a pump valve"))
    docs.append(("rel-0002", "valve pump seal pump valve"))
    print(f"building {len(docs)} documents ...")
    t0 = time.monotonic()
    report = build_index(path, docs, name="scale", overwrite=True)
    print(
        f"built in {report.elapsed_seconds:.2f}s "
        f"({report.token_count} tokens, fp={report.index_fingerprint[:12]})"
    )
    print(f"python build wall time {time.monotonic() - t0:.2f}s")

    with open_index(path) as index:
        print("stats:", json.dumps(index.stats(), ensure_ascii=False))
        for query in (
            "NEAR(3,[pump,pump])",
            "SEQ(0,[pump,valve])",
            "NEAR(3,[OR(pump,seal), valve, pump])",
            "NEAR(2,[absent,pump])",
        ):
            trace = AccessTrace()
            t0 = time.monotonic()
            page = search(index, parse_query(query), limit=10, trace=trace)
            dt = time.monotonic() - t0
            print(
                f"\n{query}\n  windows={len(page.windows)} "
                f"docs={[w.doc_id for w in page.windows]}"
                f"  elapsed={dt * 1000:.1f}ms"
            )
            print("  trace:", json.dumps(page.trace, ensure_ascii=False))
            print(
                "  snippets:",
                [(w.doc_id, w.start, w.end, w.snippet) for w in page.windows],
            )
            assert trace.missing_token_docs > 0
            if query == "NEAR(2,[absent,pump])":
                assert page.trace["body_reads"] == []

        # Tiny pages over a query with many hits: stable, non-overlapping.
        flat = []
        cursor = None
        pages = 0
        while True:
            page = search(index, parse_query("OR(pump,valve)"), limit=2,
                          cursor=cursor)
            pages += 1
            flat.extend((w.doc_id, w.start, w.end, w.witness) for w in page.windows)
            if not page.has_more:
                break
            cursor = page.next_cursor
        full_page = search(index, parse_query("OR(pump,valve)"), limit=1000)
        full = [(w.doc_id, w.start, w.end, w.witness) for w in full_page.windows]
        assert flat == full, "pagination was not a stable partition"
        assert len(flat) == len(set(flat)), "duplicate window across pages"
        print(
            f"\npagination: {len(flat)} windows over {pages} pages of 2, "
            "no duplicates or gaps"
        )
    print(f"\nindex kept at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
