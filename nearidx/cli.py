"""Command line interface: build, search, fsearch, snippet, stats, verify, rebuild, refresh.

Examples
--------
::

    python -m nearidx build corpus.nearidx docs/*.txt --doc-id-from stem
    python -m nearidx search corpus.nearidx 'NEAR(3,[pump,pump])' --limit 5
    python -m nearidx fsearch vendors.json 'NEAR(3,[pump,pump])' --limit 5
    python -m nearidx snippet corpus.nearidx d1 --char 0:15
    python -m nearidx verify corpus.nearidx --full
    python -m nearidx rebuild corpus.nearidx d1 --file docs/d1.txt
    python -m nearidx refresh corpus.nearidx batch.jsonl --retain-old 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .db import (
    Document,
    IndexFormatError,
    StaleSourceError,
    build_index,
    open_index,
    rebuild_document,
    text_sha256,
)
from .federate import Federation, FederationError, federated_search
from .query import CursorError, QuerySyntaxError, parse_query, serialize_query
from .refresh import ManifestError, refresh_index
from .solver import QueryError, search


def _emit(obj: object) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _load_documents(args: argparse.Namespace) -> List[Document]:
    docs: List[Document] = []
    for raw in args.files:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
        if args.doc_id_from == "stem":
            doc_id = path.stem
        else:
            doc_id = path.name
        docs.append(Document.from_file(doc_id, path))
    return docs


def cmd_build(args: argparse.Namespace) -> int:
    docs = _load_documents(args)
    report = build_index(
        args.index,
        docs,
        name=args.name,
        overwrite=args.overwrite,
    )
    _emit(
        {
            "path": report.path,
            "doc_count": report.doc_count,
            "token_count": report.token_count,
            "index_fingerprint": report.index_fingerprint,
            "elapsed_seconds": round(report.elapsed_seconds, 4),
        }
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        root = parse_query(args.query)
    except QuerySyntaxError as exc:
        sys.stderr.write(f"query error: {exc}\n")
        return 2
    try:
        index = open_index(args.index)
    except (FileNotFoundError, IndexFormatError) as exc:
        sys.stderr.write(f"index error: {exc}\n")
        return 2
    try:
        page = search(
            index,
            root,
            limit=args.limit,
            cursor=args.cursor,
            offset=0 if args.cursor else args.offset,
        )
        result = page.as_dict()
        result["query"] = serialize_query(root)
        if args.json:
            _emit(result)
            return 0
        for win in page.windows:
            print(
                f"{win.doc_id}\t{win.start}-{win.end}\t"
                f"chars {win.char_start}:{win.char_end}\t"
                f"bytes {win.byte_start}:{win.byte_end}\t"
                f"witness {list(win.witness)}\t{win.snippet}"
            )
        tr = page.trace
        print(
            f"# candidates={tr['candidate_docs']} solved={tr['docs_solved']} "
            f"bodies={len(tr['body_reads'])} has_more={page.has_more}",
            file=sys.stderr,
        )
        if page.next_cursor:
            print(f"# next_cursor={page.next_cursor}", file=sys.stderr)
        return 0
    except (CursorError, QueryError) as exc:
        sys.stderr.write(f"query error: {exc}\n")
        return 2
    finally:
        index.close()


def cmd_fsearch(args: argparse.Namespace) -> int:
    try:
        root = parse_query(args.query)
    except QuerySyntaxError as exc:
        sys.stderr.write(f"query error: {exc}\n")
        return 2
    try:
        federation = Federation.from_manifest(args.manifest)
    except FederationError as exc:
        sys.stderr.write(f"manifest error: {exc}\n")
        return 2
    except (FileNotFoundError, IndexFormatError) as exc:
        sys.stderr.write(f"index error: {exc}\n")
        return 2
    try:
        page = federated_search(
            federation,
            root,
            limit=args.limit,
            cursor=args.cursor,
        )
        result = page.as_dict()
        result["query"] = serialize_query(root)
        if args.json:
            _emit(result)
            return 0
        for fed_window in page.windows:
            win = fed_window.window
            print(
                f"{fed_window.source_name}\t{win.doc_id}\t{win.start}-{win.end}\t"
                f"chars {win.char_start}:{win.char_end}\t"
                f"bytes {win.byte_start}:{win.byte_end}\t"
                f"witness {list(win.witness)}\t{win.snippet}"
            )
        print(
            f"# sources={len(page.sources)} has_more={page.has_more}",
            file=sys.stderr,
        )
        if page.next_cursor:
            print(f"# next_cursor={page.next_cursor}", file=sys.stderr)
        return 0
    except (CursorError, QueryError) as exc:
        sys.stderr.write(f"query error: {exc}\n")
        return 2
    finally:
        federation.close()


def cmd_snippet(args: argparse.Namespace) -> int:
    index = open_index(args.index)
    try:
        if args.char:
            start_s, end_s = args.char.split(":", 1)
            char_start, char_end = int(start_s), int(end_s)
            if char_start < 0:
                sys.stderr.write("--char start must be >= 0\n")
                return 2
            text = index.text(args.doc_id)
            snippet = index.snippet(args.doc_id, char_start, char_end)
        elif args.positions:
            pos_start, pos_end = (int(x) for x in args.positions.split(":", 1))
            rows = index.token_spans(args.doc_id, pos_start, pos_end)
            if not rows:
                sys.stderr.write("no tokens in that position range\n")
                return 2
            char_start = rows[0][1]
            char_end = rows[-1][2]
            text = index.text(args.doc_id)
            snippet = index.snippet(args.doc_id, char_start, char_end)
        else:
            sys.stderr.write("give --char A:B or --positions A:B\n")
            return 2
        _emit(
            {
                "doc_id": args.doc_id,
                "char_span": [char_start, char_end],
                "snippet": snippet,
                "char_ok": text[char_start:char_end] == snippet,
            }
        )
        return 0
    finally:
        index.close()


def cmd_stats(args: argparse.Namespace) -> int:
    index = open_index(args.index)
    try:
        _emit(index.stats())
        return 0
    finally:
        index.close()


def cmd_verify(args: argparse.Namespace) -> int:
    index = open_index(args.index, verify=False)
    try:
        problems = index.verify(full=args.full, generations=args.generations)
        _emit({"ok": not problems, "problems": problems})
        return 0 if not problems else 1
    finally:
        index.close()


def cmd_refresh(args: argparse.Namespace) -> int:
    try:
        report = refresh_index(
            args.index, args.manifest, retain_old=args.retain_old
        )
    except FileNotFoundError as exc:
        sys.stderr.write(f"index error: {exc}\n")
        return 2
    except ManifestError as exc:
        sys.stderr.write(f"manifest error: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"refresh error: {exc}\n")
        return 2
    _emit(report.as_dict())
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    raw = Path(args.file).read_bytes()
    text = raw.decode("utf-8")
    sha = text_sha256(text)
    fingerprint = rebuild_document(
        args.index,
        args.doc_id,
        text,
        source_path=str(Path(args.file).resolve()),
        source_sha256=sha,
    )
    _emit({"doc_id": args.doc_id, "index_fingerprint": fingerprint})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nearidx", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a fresh index")
    b.add_argument("index")
    b.add_argument("files", nargs="+")
    b.add_argument("--doc-id-from", choices=("stem", "name"), default="stem")
    b.add_argument("--name", default=None)
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="run a query")
    s.add_argument("index")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--cursor", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    f = sub.add_parser(
        "fsearch",
        help="federated search: run one query over every index named in a "
        "JSON manifest and merge the windows in (doc_id, source, start, "
        "end) order",
    )
    f.add_argument(
        "manifest",
        help="JSON object mapping stable source names to index files "
        "(relative paths resolve against the manifest's directory)",
    )
    f.add_argument("query")
    f.add_argument("--limit", type=int, default=20)
    f.add_argument("--cursor", default=None)
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_fsearch)

    sn = sub.add_parser("snippet", help="cut original text by char or token span")
    sn.add_argument("index")
    sn.add_argument("doc_id")
    sn.add_argument("--char", default=None)
    sn.add_argument("--positions", default=None)
    sn.set_defaults(func=cmd_snippet)

    st = sub.add_parser("stats", help="index statistics")
    st.add_argument("index")
    st.set_defaults(func=cmd_stats)

    v = sub.add_parser("verify", help="check text/posting binding")
    v.add_argument("index")
    v.add_argument("--full", action="store_true")
    v.add_argument(
        "--generations",
        action="store_true",
        help="also verify every retained old generation against the registry",
    )
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("rebuild", help="explicitly replace one document")
    r.add_argument("index")
    r.add_argument("doc_id")
    r.add_argument("--file", required=True)
    r.set_defaults(func=cmd_rebuild)

    rf = sub.add_parser(
        "refresh",
        help="validate a JSONL batch (upsert/delete) and publish it as one "
        "new index generation",
    )
    rf.add_argument("index")
    rf.add_argument("manifest", help="JSON Lines manifest: one op per line")
    rf.add_argument(
        "--retain-old",
        type=int,
        default=None,
        help="old generations to keep queryable for in-flight cursors "
        "(default: keep the registry setting, initially 2; minimum 1)",
    )
    rf.set_defaults(func=cmd_refresh)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StaleSourceError as exc:
        sys.stderr.write(f"stale source: {exc}\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
