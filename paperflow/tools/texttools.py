"""Local text tools (no network): searching the paper body for evidence.

Quote verification is a *code* guarantee: no LLM decides whether a quote is
present. Getting that guarantee right means modelling how PDF extraction
mangles text, because a verifier that rejects quotes that *are* in the paper
is worse than no verifier at all - it launders "my code said no" into
"the claim is unsupported".

Matching tiers (weakest relaxation last; every tier is reported in
``match_mode`` so a repaired match is never silently equated with a verbatim
one):

``verbatim``
    the quote occurs literally after collapsing whitespace and lowercasing
    (the original behaviour).
``hyphen_repaired``
    the quote occurs literally after repairing PDF artifacts: soft hyphens
    and zero-width characters removed, a hyphen at a line break joined, and
    hyphens *inside* a word dropped on both sides (``English-\\nto-German``
    -> ``englishtogerman`` == ``English-to-German``). Still a plain
    substring test.
``token_aligned``
    the quote's tokens occur **in order**, where (a) token comparison ignores
    attached punctuation, and (b) the paper may contain extra *noise* tokens
    between them - but only bare small integers (page numbers extracted into
    the sentence, e.g. ``the sequence\\n6\\nlength``) and only within a small
    budget. Word order, wording and every number in the quote must still be
    present: an edited number is never skipped, it simply fails.

A quote containing an ellipsis (``...``) is verified segment by segment, in
order, and the result reports how many segments matched.
"""

from __future__ import annotations

import re
from pathlib import Path

from paperflow.core.jsonutil import json_dumps

_WHITESPACE = re.compile(r"\s+")

#: characters PDF extraction sprinkles into words
_SOFT_HYPHEN = "\u00ad"
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
_INVISIBLE = re.compile(f"[{_SOFT_HYPHEN}{_ZERO_WIDTH}]")

#: a hyphen sitting at a line break: "English-\nto-German" -> "Englishto-German"
_LINE_BREAK_HYPHEN = re.compile(r"[-\u2010\u2011]\s*\n\s*")
#: a hyphen between two word characters: "self-attention" -> "selfattention"
_INNER_HYPHEN = re.compile(r"(?<=\w)[-\u2010\u2011](?=\w)")
#: everything that is not a letter/digit, dropped when comparing tokens
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
#: the shape of a page/line number in extracted text
_NOISE_TOKEN = re.compile(r"\d{1,3}")
_TOKEN = re.compile(r"\S+")
#: ellipsis markers used when a quote elides a middle part
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\. \. \.|…|\[\s*\.\.\.\s*\])\s*")

#: floors that keep the verifier from accepting non-evidence
MIN_QUOTE_CHARS = 8
MIN_QUOTE_TOKENS = 2
#: the relaxed tiers demand a longer, more distinctive quote
MIN_ALIGNED_CHARS = 20
MIN_ALIGNED_TOKENS = 4
#: how many noise tokens may be skipped inside one token alignment
MAX_NOISE_TOKENS = 4

#: match tiers, least relaxed first
MODES = ("verbatim", "hyphen_repaired", "token_aligned")

#: explicit verification status of a claim's quote. "No answer" is never
#: allowed to look like a pass: a check that could not run is
#: ``unverifiable``, one that ran and failed is ``unverified``.
STATUS_VERIFIED = "verified"
STATUS_UNVERIFIED = "unverified"
STATUS_UNVERIFIABLE = "unverifiable"


def normalize_ws(text: str) -> str:
    """Collapse every whitespace run to a single space, lowercased.

    Used by quote verification so that a quote extracted across wrapped
    lines ("win rate\\n44.2%") still matches the source text.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


def canonical_text(text: str) -> str:
    """Whitespace-normalized text with PDF hyphenation artifacts repaired.

    In order: invisible characters stripped, hyphens at line breaks joined,
    hyphens inside words removed, whitespace collapsed, lowercased. Removing
    inner hyphens on *both* sides is what lets a word the PDF split across
    two lines compare equal to the quote's single hyphenated word.
    """
    text = _INVISIBLE.sub("", text)
    text = _LINE_BREAK_HYPHEN.sub("", text)
    text = _INNER_HYPHEN.sub("", text)
    return normalize_ws(text)


def canonical_tokens(text: str) -> list[str]:
    """Tokens of :func:`canonical_text` with attached punctuation removed.

    ``"ensembles,"`` and ``"ensembles"`` compare equal; digits are never
    dropped, so a quote whose number was edited still fails to match.
    """
    return [t for t in (_NON_ALNUM.sub("", tok) for tok in canonical_text(text).split(" ")) if t]


def _token_spans(canonical: str) -> list[tuple[str, int, int]]:
    """(token, start, end) triples, so a token match maps back to a location."""
    spans = []
    for match in _TOKEN.finditer(canonical):
        token = _NON_ALNUM.sub("", match.group())
        if token:
            spans.append((token, match.start(), match.end()))
    return spans


def _context(source: str, loc: int, length: int, context_window: int) -> str:
    start = max(0, loc - context_window)
    end = min(len(source), loc + length + context_window)
    return source[start:end]


def _not_found(reason: str, segments: int = 1, status: str = STATUS_UNVERIFIED) -> dict:
    return {
        "found": False,
        "reason": reason,
        "loc": None,
        "context": None,
        "match_mode": "none",
        "segments": segments,
        "noise_tokens": 0,
        "status": status,
    }


def _hit(reason: str, loc: int, context: str, mode: str, segments: int = 1, noise: int = 0) -> dict:
    return {
        "found": True,
        "reason": reason,
        "loc": loc,
        "context": context,
        "match_mode": mode,
        "segments": segments,
        "noise_tokens": noise,
        "status": STATUS_VERIFIED,
    }


def _align_tokens(haystack: str, needle_tokens: list[str]) -> tuple[int, int, int] | None:
    """Match `needle_tokens` in order, skipping only noise tokens.

    Returns ``(loc, end, skipped_noise)`` or None. Every skipped haystack
    token must be a bare small integer (an extracted page/line number) and
    the skip budget is capped, so tokens lifted from far-apart sentences
    cannot be glued together into a match.
    """
    spans = _token_spans(haystack)
    hay_tokens = [s[0] for s in spans]
    if len(hay_tokens) < len(needle_tokens):
        return None
    budget = min(MAX_NOISE_TOKENS, max(1, len(needle_tokens) // 4))

    for first in range(len(hay_tokens) - len(needle_tokens) + 1):
        if hay_tokens[first] != needle_tokens[0]:
            continue
        cursor = first + 1
        skipped = 0
        ok = True
        for token in needle_tokens[1:]:
            while cursor < len(hay_tokens) and hay_tokens[cursor] != token:
                if not _NOISE_TOKEN.fullmatch(hay_tokens[cursor]):
                    ok = False
                    break
                skipped += 1
                cursor += 1
                if skipped > budget:
                    ok = False
                    break
            if not ok or cursor >= len(hay_tokens):
                ok = False
                break
            cursor += 1  # consume the matched token
        if ok:
            return spans[first][1], spans[cursor - 1][2], skipped
    return None


def _ordered_match(source: str, segments: list[str], normalize, context_window: int) -> tuple[int, int, str] | None:
    """Find every segment in `source`, in order, as substrings.

    Returns ``(first_loc, noise_tokens, context)`` or None.
    """
    cursor = 0
    first_loc: int | None = None
    for segment in segments:
        needle = normalize(segment)
        loc = source.find(needle, cursor)
        if loc == -1:
            return None
        if first_loc is None:
            first_loc = loc
        cursor = loc + len(needle)
    assert first_loc is not None
    return first_loc, 0, _context(source, first_loc, cursor - first_loc, context_window)


def _ordered_token_match(source: str, segments: list[str], context_window: int) -> tuple[int, int, str] | None:
    """Token-align every segment of an elided quote, in order."""
    cursor = 0
    first_loc: int | None = None
    noise_total = 0
    for segment in segments:
        tokens = canonical_tokens(segment)
        if len(tokens) < MIN_ALIGNED_TOKENS:
            return None
        aligned = _align_tokens(source[cursor:], tokens)
        if aligned is None:
            return None
        loc, end, skipped = aligned
        if first_loc is None:
            first_loc = cursor + loc
        noise_total += skipped
        cursor += end
    assert first_loc is not None
    return first_loc, noise_total, _context(source, first_loc, cursor - first_loc, context_window)


def _match_single(full_text: str, quote: str, context_window: int, min_chars: int) -> dict:
    """Match one quote that has no elided middle."""
    classic = normalize_ws(full_text)
    needle_classic = normalize_ws(quote)
    loc = classic.find(needle_classic)
    if loc != -1:
        return _hit(
            "quote found verbatim (whitespace-normalized)",
            loc,
            _context(classic, loc, len(needle_classic), context_window),
            "verbatim",
        )

    canonical = canonical_text(full_text)
    needle_canonical = canonical_text(quote)
    loc = canonical.find(needle_canonical)
    if loc != -1:
        return _hit(
            "quote found after repairing PDF hyphenation (line-break and in-word hyphens)",
            loc,
            _context(canonical, loc, len(needle_canonical), context_window),
            "hyphen_repaired",
        )

    needle_tokens = canonical_tokens(quote)
    if len(needle_canonical) >= MIN_ALIGNED_CHARS and len(needle_tokens) >= MIN_ALIGNED_TOKENS:
        aligned = _align_tokens(canonical, needle_tokens)
        if aligned is not None:
            start, end, skipped = aligned
            note = "quote found token-by-token (punctuation-insensitive"
            note += f", {skipped} page-number token(s) skipped in the source)" if skipped else ")"
            return _hit(note, start, _context(canonical, start, end - start, context_window), "token_aligned", noise=skipped)
    return _not_found("quote not found in the paper text")


def _match_segments(full_text: str, segments: list[str], context_window: int, min_chars: int) -> dict:
    """Match an ellipsis-elided quote segment by segment, in document order.

    Tried from the strictest interpretation downwards, so the reported mode
    says how much repair was actually needed: literal segments, then
    hyphen-repaired segments, then token alignment.
    """
    canonical = canonical_text(full_text)
    modes = (
        ("verbatim", normalize_ws(full_text), lambda src: _ordered_match(src, segments, normalize_ws, context_window), "as literal segments"),
        ("hyphen_repaired", canonical, lambda src: _ordered_match(src, segments, canonical_text, context_window), "as hyphen-repaired segments"),
        ("token_aligned", canonical, lambda src: _ordered_token_match(src, segments, context_window), "as token-aligned segments"),
    )
    for tier, source, matcher, label in modes:
        found = matcher(source)
        if found:
            loc, noise, context = found
            return _hit(
                f"quote found {label} ({len(segments)} segments, elided middle skipped)",
                loc,
                context,
                f"{tier}_segments",  # never reported as a plain whole-quote match
                segments=len(segments),
                noise=noise,
            )
    return _not_found("quote not found in the paper text (elided quote)", segments=len(segments))


def _split_ellipsis(quote: str) -> list[str]:
    """Split an elided quote; returns [] when it is not a usable elision."""
    segments = [s.strip() for s in _ELLIPSIS.split(quote)]
    if len(segments) < 2 or any(not s for s in segments):
        return []
    if not all(len(normalize_ws(s)) >= MIN_QUOTE_CHARS and len(canonical_tokens(s)) >= MIN_QUOTE_TOKENS for s in segments):
        return []
    return segments


def verify_quote(full_text: str, quote: str, min_chars: int = MIN_QUOTE_CHARS, context_window: int = 180) -> dict:
    """Deterministic quote-presence check against the paper text.

    Case-insensitive, tolerant of PDF extraction artifacts and of an elided
    middle (see the module docstring). ``found`` is True only when the
    quote's own words - in order, numbers included - occur in the paper. No
    LLM and no third-party fuzzy library is involved.

    Returns ``{"found", "reason", "loc", "context", "match_mode",
    "segments", "noise_tokens", "status"}``. ``loc`` indexes the normalized
    text the ``context`` was taken from. ``match_mode`` is ``verbatim`` /
    ``hyphen_repaired`` / ``token_aligned`` for a whole-quote match, the same
    three names with a ``_segments`` suffix when the quote was elided with an
    ellipsis and matched segment by segment, or ``none`` when it was not
    found. ``status`` is the explicit verdict: ``verified``, ``unverified``
    (the check ran and did not find the quote) or ``unverifiable`` (there was
    nothing to check against, or no usable quote to check).
    """
    if not quote or not quote.strip():
        return _not_found("empty quote", status=STATUS_UNVERIFIABLE)
    if len(normalize_ws(quote)) < min_chars or len(canonical_tokens(quote)) < MIN_QUOTE_TOKENS:
        return _not_found("quote too short to verify", status=STATUS_UNVERIFIABLE)
    if not normalize_ws(full_text):
        return _not_found("no full text available", status=STATUS_UNVERIFIABLE)

    segments = _split_ellipsis(quote)
    if segments:
        return _match_segments(full_text, segments, context_window, min_chars)
    return _match_single(full_text, quote, context_window, min_chars)


def verify_claims(full_text: str, claims: list[dict]) -> list[dict]:
    """Annotate every claim with its deterministic verification result.

    Each claim dict gains ``quote_verified`` (bool), ``quote_status``
    (explicit ``verified`` / ``unverified`` / ``unverifiable``),
    ``quote_verification`` (reason string), ``quote_match_mode`` (how the
    quote matched), ``quote_loc`` (match index, may be None) and
    ``quote_context`` (surrounding passage, may be None).

    Passing an empty ``full_text`` is a supported case (abstract-only runs):
    every claim is then annotated ``unverifiable`` **explicitly**, so that no
    downstream stage can mistake "not checked" for "fine".
    """
    for claim in claims:
        result = verify_quote(full_text, claim.get("quote", ""))
        claim["quote_verified"] = result["found"]
        claim["quote_status"] = result["status"]
        claim["quote_verification"] = result["reason"]
        claim["quote_match_mode"] = result["match_mode"]
        claim["quote_loc"] = result["loc"]
        claim["quote_context"] = result["context"]
    return claims


def search_text(path: str, query: str, window: int = 300, max_hits: int = 3) -> str:
    """Find passages of `query` in the paper text file at `path`.

    Returns a JSON string of the form::

        {"query": ..., "total": N, "matches": [{"index": i, "passage": "..."}]}

    The Critic uses this to verify that a claimed quote actually appears in
    the paper body without re-sending the whole text to the LLM.
    """
    doc = Path(path)
    if not doc.is_file():
        return json_dumps({"error": f"text file not found: {path}", "total": 0, "matches": []})

    text = doc.read_text(encoding="utf-8", errors="replace")
    needle = query.strip().lower()
    if not needle:
        return json_dumps({"query": query, "total": 0, "matches": []})

    matches: list[dict] = []
    start = 0
    while True:
        idx = text.lower().find(needle, start)
        if idx == -1 or len(matches) >= max_hits:
            break
        passage = text[max(0, idx - window) : idx + len(needle) + window]
        passage = _WHITESPACE.sub(" ", passage).strip()
        matches.append({"index": idx, "passage": passage})
        start = idx + len(needle)

    return json_dumps({"query": query, "total": len(matches), "matches": matches})
