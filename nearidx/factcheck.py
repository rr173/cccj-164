"""A short, executable walkthrough of the delivered behaviour.

Run ``python -m nearidx.factcheck``.  It builds a temporary index, prints the
specification's main example, a nested witness case, paging evidence and an
integrity check, and exits non-zero on any mismatch.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .db import build_index, open_index
from .query import parse_query
from .solver import search


def show(index, query: str, *, limit: int = 20):
    page = search(index, parse_query(query), limit=limit)
    return {
        "query": query,
        "windows": [
            {
                "doc_id": w.doc_id,
                "positions": [w.start, w.end],
                "leaves": [leaf.positions for leaf in w.leaves],
                "snippet": w.snippet,
            }
            for w in page.windows
        ],
        "trace": page.trace,
    }


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="nearidx-factcheck-"))
    path = tmp / "demo.nearidx"
    build_index(
        path,
        [
            ("main", "pump valve pump"),
            ("nested", "a b c"),
            ("fold", "STRASSE straße ÄRGER"),
            ("noise", "nothing relevant here"),
        ],
        name="demo",
        overwrite=True,
    )
    out = {}
    with open_index(path) as index:
        out["NEAR(2,[pump,pump])"] = show(index, "NEAR(2,[pump,pump])")
        out["NEAR(3,[pump,pump])"] = show(index, "NEAR(3,[pump,pump])")
        out["SEQ(0,[pump,valve])"] = show(index, "SEQ(0,[pump,valve])")
        out["overlap"] = show(index, "NEAR(3,[SEQ(1,[a,c]), b])")
        out["fold"] = show(index, "SEQ(0,[strasse,ärger])")
        out["verify"] = index.verify(full=True)

    assert out["NEAR(2,[pump,pump])"]["windows"] == []
    assert out["NEAR(3,[pump,pump])"]["windows"][0]["leaves"] == [[0], [2]]
    assert out["SEQ(0,[pump,valve])"]["windows"][0]["positions"] == [0, 1]
    assert out["overlap"]["windows"][0]["leaves"] == [[0], [2], [1]]
    assert out["fold"]["windows"][0]["snippet"] == "straße ÄRGER"
    assert out["verify"] == []
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
