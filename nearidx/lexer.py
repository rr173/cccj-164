"""Fixed tokenisation shared by the builder, the query parser and snippets.

A token is a maximal run of Unicode alphanumerics (``str.isalnum()``) plus
``_``.  Every other character is a separator.  Only the token text as a whole
is case-folded (``str.casefold``); the original characters are never rewritten
and their character / UTF-8 byte offsets are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List


@dataclass(frozen=True, slots=True)
class Token:
    """One token occurrence in a document."""

    pos: int          # 0-based token position
    text: str         # original slice, never case-folded
    norm: str         # text.casefold(), lookup key
    char_start: int   # original char offset (inclusive)
    char_end: int     # original char offset (exclusive)
    byte_start: int   # UTF-8 byte offset (inclusive)
    byte_end: int     # UTF-8 byte offset (exclusive)


def is_token_char(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def iter_tokens(text: str) -> Iterator[Token]:
    """Yield tokens left to right, assigning positions from 0.

    One pass over the string groups maximal runs of token characters.  Byte
    offsets are accumulated as we go, so multibyte input is linear time.
    """
    n = len(text)
    i = 0
    pos = 0
    byte_at = 0  # UTF-8 byte offset of text[i]

    while i < n:
        ch = text[i]
        if not is_token_char(ch):
            byte_at += len(ch.encode("utf-8"))
            i += 1
            continue
        start = i
        byte_start = byte_at
        while i < n and is_token_char(text[i]):
            byte_at += len(text[i].encode("utf-8"))
            i += 1
        raw = text[start:i]
        yield Token(
            pos=pos,
            text=raw,
            norm=raw.casefold(),
            char_start=start,
            char_end=i,
            byte_start=byte_start,
            byte_end=byte_at,
        )
        pos += 1


def token_spans(text: str) -> List[Token]:
    return list(iter_tokens(text))


def single_query_token(s: str) -> str:
    """Lex an exact query string; it must produce exactly one token.

    Returns the folded lookup key.  Raises ``ValueError`` otherwise: the query
    language never splits, stems or expands its tokens.
    """
    if not isinstance(s, str):
        raise ValueError("query token must be a string")
    if s == "":
        raise ValueError("empty query token")
    spans = token_spans(s)
    if len(spans) != 1:
        raise ValueError(
            f"query token {s!r} does not lex to exactly one token under the "
            "fixed lexical rule (letters/digits/underscore only, no separators)"
        )
    only = spans[0]
    if only.char_start != 0 or only.char_end != len(s):
        raise ValueError(f"query token {s!r} is a token plus separators")
    folded = s.casefold()
    folded_spans = token_spans(folded)
    if len(folded_spans) != 1 or folded_spans[0].char_end != len(folded):
        raise ValueError(f"query token {s!r} folds to a non-token")
    return folded
