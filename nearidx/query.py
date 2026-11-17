"""Query AST, parser, and the fixed semantics of the near-neighbour language.

Language
--------
::

    expr    := TOKEN | NAME '(' args ')'
    TOKEN   := one lexical token (letters/digits/underscore, defines a leaf)
    OR      '(' expr (',' expr)+ ')'
    SEQ     '(' INT ',' children ')'
    NEAR    '(' INT ',' children ')'
    children:= expr (',' expr)+ | '[' expr (',' expr)+ ']'

``OR`` chooses exactly one child hit; every leaf below the other children is
unused and its witness slot is ``-1``.  ``SEQ(gap, ...)`` requires the children
in the given order, the previous child's closed interval ending strictly before
the next one starts, and at most ``gap`` unoccupied token positions between
them.  ``NEAR(width, ...)`` does not order the children, but the used position
sets of all children must be pairwise disjoint and the hull of all used
positions must be at most ``width`` positions wide.

A hit is a used-position set plus its closed hull interval.  Intervals may
overlap between children; only the *used positions* must be disjoint.  Chinese
whitespace is irrelevant: every token occurrence takes one position.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, List

from .lexer import single_query_token


class QuerySyntaxError(ValueError):
    pass


@dataclass(slots=True)
class TokenNode:
    token: str          # folded lookup key
    spelling: str       # token as written in the query
    leaf: int = -1
    first_leaf: int = -1
    leaf_count: int = 1
    width: int = 1


@dataclass(slots=True)
class OrNode:
    children: List["Node"]
    first_leaf: int = -1
    leaf_count: int = 0
    width: int = 0


@dataclass(slots=True)
class SeqNode:
    gap: int
    children: List["Node"]
    first_leaf: int = -1
    leaf_count: int = 0
    width: int = 0


@dataclass(slots=True)
class NearNode:
    near_width: int
    children: List["Node"]
    first_leaf: int = -1
    leaf_count: int = 0
    width: int = 0


Node = TokenNode | OrNode | SeqNode | NearNode


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

_OPS = {"OR", "SEQ", "NEAR"}
_WS = " \t\r\n\f\v"


class _Parser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0
        self.n = len(text)
        self.next_leaf = 0

    # -- helpers ---------------------------------------------------------
    def error(self, msg: str) -> QuerySyntaxError:
        return QuerySyntaxError(f"{msg} at offset {self.i} in {self.s!r}")

    def skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in _WS:
            self.i += 1

    def peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def take(self, ch: str) -> None:
        if self.peek() != ch:
            raise self.error(f"expected {ch!r}")
        self.i += 1

    def read_token_chars(self) -> str:
        from .lexer import is_token_char

        start = self.i
        while self.i < self.n and is_token_char(self.s[self.i]):
            self.i += 1
        return self.s[start:self.i]

    def parse_int(self) -> int:
        start = self.i
        digits = ""
        while self.i < self.n and self.s[self.i].isdigit():
            digits += self.s[self.i]
            self.i += 1
        if not digits:
            self.i = start
            raise self.error("expected a non-negative integer")
        return int(digits)

    # -- grammar ---------------------------------------------------------
    def parse_expr(self) -> Node:
        self.skip_ws()
        if self.peek() == "":
            raise self.error("expected expression")
        word = self.read_token_chars()
        if not word:
            raise self.error("expected token or OR/SEQ/NEAR")
        self.skip_ws()
        if self.peek() == "(" and word.upper() in _OPS:
            return self.parse_op(word.upper())
        if self.peek() == "(":
            raise self.error(f"unknown operator {word!r}")
        # A bare word must be exactly one token under the shared lexer.
        folded = single_query_token(word)
        node = TokenNode(token=folded, spelling=word)
        node.leaf = self.next_leaf
        self.next_leaf += 1
        return node

    def parse_op(self, name: str) -> Node:
        self.take("(")
        if name == "OR":
            children = self.parse_children()
            node: Node = OrNode(children=children)
        else:
            self.skip_ws()
            value = self.parse_int()
            self.skip_ws()
            if self.peek() == ",":
                self.take(",")
            elif self.peek() != "]":
                raise self.error("expected ',' after gap/width")
            children = self.parse_children()
            if name == "SEQ":
                if value < 0:
                    raise self.error("SEQ gap must be non-negative")
                node = SeqNode(gap=value, children=children)
            else:
                if value < 1:
                    raise self.error("NEAR width must be at least 1")
                node = NearNode(near_width=value, children=children)
        if len(children) < 2:
            raise self.error(f"{name} needs at least two children")
        return node

    def parse_children(self) -> List[Node]:
        self.skip_ws()
        bracketed = self.peek() == "["
        if bracketed:
            self.take("[")
        out: List[Node] = []
        while True:
            self.skip_ws()
            if bracketed and self.peek() == "]":
                self.take("]")
                break
            if not bracketed and self.peek() == ")":
                break
            out.append(self.parse_expr())
            self.skip_ws()
            if self.peek() == ",":
                self.take(",")
                continue
            if bracketed:
                if self.peek() == "]":
                    self.take("]")
                    break
                raise self.error("expected ',' or ']'")
            break
        self.skip_ws()
        self.take(")")
        return out


def parse_query(text: str) -> Node:
    """Parse a query and number its leaves in pre-order from 0."""
    if not isinstance(text, str) or not text.strip():
        raise QuerySyntaxError("empty query")
    p = _Parser(text)
    root = p.parse_expr()
    p.skip_ws()
    if p.i != p.n:
        raise QuerySyntaxError(
            f"unexpected trailing input at offset {p.i} in {text!r}"
        )
    _number_leaves(root)
    return root


def _number_leaves(node: Node) -> None:
    """Assign contiguous pre-order leaf ids and leaf ranges.

    Leaves are numbered left-to-right in depth-first order, from 0.  This is
    also used by tests that build trees directly instead of parsing text.
    """
    counter = 0

    def visit(n: Node) -> None:
        nonlocal counter
        if isinstance(n, TokenNode):
            n.leaf = counter
            n.first_leaf = counter
            n.leaf_count = 1
            n.width = 1
            counter += 1
            return
        first = counter
        for child in n.children:
            visit(child)
        n.first_leaf = first
        n.leaf_count = counter - first

    visit(node)


def iter_leaves(node: Node) -> Iterable[TokenNode]:
    if isinstance(node, TokenNode):
        yield node
    else:
        for child in node.children:
            yield from iter_leaves(child)


def leaf_tokens(root: Node) -> List[str]:
    leaves = list(iter_leaves(root))
    assert [leaf.leaf for leaf in leaves] == list(range(len(leaves)))
    return [leaf.token for leaf in leaves]


def query_signature(root: Node) -> str:
    """Stable signature used to bind pagination cursors to a query."""
    return hashlib.sha256(serialize_query(root).encode("utf-8")).hexdigest()[:16]


def serialize_query(node: Node) -> str:
    if isinstance(node, TokenNode):
        return f"T({node.token})"
    if isinstance(node, OrNode):
        return "OR(" + ",".join(serialize_query(c) for c in node.children) + ")"
    if isinstance(node, SeqNode):
        return (
            f"SEQ({node.gap};"
            + ",".join(serialize_query(c) for c in node.children)
            + ")"
        )
    return (
        f"NEAR({node.near_width};"
        + ",".join(serialize_query(c) for c in node.children)
        + ")"
    )


# ---------------------------------------------------------------------------
# pagination cursor
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Cursor:
    offset: int
    page_size: int
    query_sig: str
    index_sig: str


class CursorError(ValueError):
    pass


def encode_cursor(
    offset: int, page_size: int, signature: str, index_sig: str
) -> str:
    raw = json.dumps(
        {"o": offset, "n": page_size, "q": signature, "i": index_sig},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(token: str) -> Cursor:
    import base64

    try:
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(token + pad)
        obj = json.loads(raw.decode("utf-8"))
        offset = obj["o"]
        page_size = obj["n"]
        signature = obj["q"]
        index_sig = obj["i"]
        if not isinstance(offset, int) or offset < 0:
            raise ValueError
        if not isinstance(page_size, int) or page_size < 1:
            raise ValueError
        if not isinstance(signature, str) or not isinstance(index_sig, str):
            raise ValueError
    except Exception as exc:  # noqa: BLE001 - surface as a query-level error
        raise CursorError("invalid pagination cursor") from exc
    return Cursor(
        offset=offset,
        page_size=page_size,
        query_sig=signature,
        index_sig=index_sig,
    )
