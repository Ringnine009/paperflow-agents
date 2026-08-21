"""Entry-point parsing: turn a raw user input into a structured InputSpec.

Supported entry kinds:
  * ``arxiv``  - https://arxiv.org/abs/<id>, /pdf/<id>, export.arxiv.org URLs
  * ``doi``    - ``10.xxxx/...`` (optionally prefixed with ``doi:``)
  * ``pdf``    - a path to a local .pdf file
  * ``url``    - any other http(s) URL (paper landing page)
  * ``title``  - free text treated as a paper title to search for
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_PDF_RE = re.compile(r"arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_API_RE = re.compile(r"export\.arxiv\.org/api/query\S*id_list=([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)
_DOI_RE = re.compile(r"^(?:doi:\s*)?(10\.\d{4,9}/[^\s]+)$", re.IGNORECASE)
_HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)
_SLUG_KEEP = re.compile(r"[^\w-]+")  # \w is unicode-aware: CJK titles survive
_SLUG_DASH = re.compile(r"-+")


@dataclass
class InputSpec:
    """Normalized description of what the user asked to review."""

    raw: str                 #: the original user input
    kind: str                #: arxiv | doi | pdf | url | title
    value: str               #: normalized id / doi / path / url / title
    pdf_override: str | None = None  #: optional local PDF to use as full text

    def to_dict(self) -> dict:
        return {"raw": self.raw, "kind": self.kind, "value": self.value, "pdf_override": self.pdf_override}

    @classmethod
    def from_dict(cls, data: dict) -> "InputSpec":
        return cls(raw=data["raw"], kind=data["kind"], value=data["value"], pdf_override=data.get("pdf_override"))


def arxiv_id_from_url(url: str) -> str | None:
    """Extract a bare arXiv id (version suffix dropped) from a URL, if any."""
    for pattern in (_ARXIV_ABS_RE, _ARXIV_PDF_RE, _ARXIV_API_RE):
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def is_doi(text: str) -> bool:
    """True when the string looks like a DOI (10.xxxx/...)."""
    return _DOI_RE.match(text.strip()) is not None


def parse_entry(raw: str, pdf_override: str | None = None) -> InputSpec:
    """Classify a raw entry string and return a normalized InputSpec."""
    text = raw.strip()
    if not text:
        raise ValueError("empty entry: nothing to review")

    arxiv_id = arxiv_id_from_url(text)
    if arxiv_id:
        return InputSpec(raw=raw, kind="arxiv", value=arxiv_id, pdf_override=pdf_override)

    if is_doi(text):
        doi = _DOI_RE.match(text).group(1)
        return InputSpec(raw=raw, kind="doi", value=doi, pdf_override=pdf_override)

    if _HTTP_RE.match(text):
        return InputSpec(raw=raw, kind="url", value=text, pdf_override=pdf_override)

    if Path(text).is_file() and text.lower().endswith(".pdf"):
        return InputSpec(raw=raw, kind="pdf", value=str(Path(text).resolve()), pdf_override=pdf_override)

    return InputSpec(raw=raw, kind="title", value=text, pdf_override=pdf_override)


def sanitize_slug(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe lowercase slug from a title (CJK-friendly)."""
    slug = _SLUG_KEEP.sub("-", text.lower())
    slug = _SLUG_DASH.sub("-", slug).strip("-")
    return slug[:max_len].rstrip("-")
