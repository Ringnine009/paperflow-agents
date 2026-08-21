"""Pure parsers for external API responses (no network, fully unit-testable).

* :func:`parse_arxiv_atom`   - arXiv Atom XML feed (export.arxiv.org)
* :func:`parse_crossref_work` - Crossref JSON work record
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

_JATS_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    """Collapse newlines/indentation and strip tags-ish noise."""
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text).strip()


def _arxiv_id_from_url(url: str) -> str:
    """'http://arxiv.org/abs/1706.03762v7' -> '1706.03762'."""
    return url.rsplit("/", 1)[-1].split("v", 1)[0]


def parse_arxiv_atom(xml_text: str) -> list[dict]:
    """Parse an arXiv Atom feed into a list of metadata dicts."""
    root = ET.fromstring(xml_text)
    results: list[dict] = []
    for entry in root.findall("a:entry", ATOM_NS):
        entry_id = entry.findtext("a:id", default="", namespaces=ATOM_NS)
        pdf_url = None
        for link in entry.findall("a:link", ATOM_NS):
            if (link.get("title") or "").lower() == "pdf":
                pdf_url = link.get("href")
        doi = entry.findtext("arxiv:doi", namespaces=ATOM_NS)
        results.append(
            {
                "arxiv_id": _arxiv_id_from_url(entry_id) if entry_id else "",
                "title": _clean(entry.findtext("a:title", namespaces=ATOM_NS)),
                "authors": [
                    _clean(author.findtext("a:name", namespaces=ATOM_NS))
                    for author in entry.findall("a:author", ATOM_NS)
                ],
                "published": _clean(entry.findtext("a:published", namespaces=ATOM_NS)),
                "abstract": _clean(entry.findtext("a:summary", namespaces=ATOM_NS)),
                "pdf_url": pdf_url,
                "abs_url": _clean(entry.findtext("a:id", namespaces=ATOM_NS)),
                "doi": doi or None,
            }
        )
    return results


def _strip_jats(text: str) -> str:
    return _clean(_JATS_TAG.sub(" ", text))


def parse_crossref_work(json_text: str) -> dict:
    """Parse a Crossref ``/works/<doi>`` response into a metadata dict.

    Raises :class:`ValueError` on malformed input.
    """
    try:
        data = json.loads(json_text)
        message = data["message"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed Crossref response: {exc}") from exc

    title = message.get("title") or [""]
    authors = [
        f"{author.get('given', '')} {author.get('family', '')}".strip()
        for author in message.get("author", [])
        if author.get("family") or author.get("given")
    ]
    published = None
    for field in ("published-print", "published-online", "issued"):
        parts = message.get(field, {}).get("date-parts") or []
        if parts and parts[0]:
            published = "-".join(str(p) for p in parts[0][:3])
            break

    pdf_url = None
    for link in message.get("link", []) or []:
        if "pdf" in (link.get("content-type") or "").lower():
            pdf_url = link.get("URL")
            break

    return {
        "title": title[0],
        "authors": authors,
        "published": published,
        "doi": message.get("DOI"),
        "url": message.get("URL"),
        "pdf_url": pdf_url,
        "abstract": _strip_jats(message.get("abstract") or ""),
    }
