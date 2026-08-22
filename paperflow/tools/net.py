"""Networking helpers shared by the fetch tools.

* :func:`throttle` - per-host minimum interval between requests. arXiv's
  API guidelines ask for >= 3s between calls to export.arxiv.org, and we
  honor that globally so heavy use cannot hammer the service.
* :func:`download_pdf` / :func:`extract_pdf_text` - fetch a PDF once into a
  content-addressed cache and pull its text with pypdf.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests
from pypdf import PdfReader

USER_AGENT = "PaperFlow/0.1 (portfolio research assistant; arXiv/Crossref API user)"


class ToolNetError(Exception):
    """A network-tool failure that should be surfaced to the LLM."""


class PDFError(ToolNetError):
    """A PDF could not be downloaded or parsed."""


#: last request timestamp per host, used by :func:`throttle`
_LAST_REQUEST: dict[str, float] = {}


def throttle(host: str, min_interval: float = 3.0) -> None:
    """Sleep so that consecutive requests to `host` are >= `min_interval` apart."""
    now = time.time()
    wait = _LAST_REQUEST.get(host, 0.0) + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST[host] = time.time()


#: statuses worth retrying - arXiv rate limits surface as 429, and 5xx are
#: transient server errors; everything else (404, 403, ...) is final
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def http_get_bytes(
    url: str,
    timeout: int = 30,
    headers: dict | None = None,
    max_retries: int = 3,
    retry_delay: float = 8.0,
) -> bytes:
    """GET a URL and return raw bytes, retrying rate limits / 5xx with backoff.

    arXiv's public API blocks IPs that burst too many requests (HTTP 429);
    waiting out the window with exponential backoff is the polite recovery.
    """
    response = None
    for attempt in range(max_retries + 1):
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT, **(headers or {})}
        )
        if response.status_code not in RETRYABLE_STATUS or attempt >= max_retries:
            break
        time.sleep(retry_delay * (attempt + 1))
    response.raise_for_status()
    return response.content


def http_get(url: str, timeout: int = 30, headers: dict | None = None) -> str:
    """GET a URL and return decoded text."""
    return http_get_bytes(url, timeout=timeout, headers=headers).decode("utf-8", errors="replace")


def _assert_http_url(url: str) -> None:
    if not url.lower().startswith(("http://", "https://")):
        raise ToolNetError(f"not an http(s) URL: {url!r}")


def download_pdf(url: str, cache_dir: str | Path, timeout: int = 60) -> Path:
    """Download a PDF into a content-addressed cache; reuse on repeat calls."""
    _assert_http_url(url)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    dest = cache / f"{digest}.pdf"
    if dest.exists():
        return dest
    dest.write_bytes(http_get_bytes(url, timeout=timeout))
    return dest


def extract_pdf_text(path: str | Path, max_chars: int = 60000) -> str:
    """Extract text from a local PDF with pypdf, truncated to `max_chars`."""
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises assorted exceptions on bad files
        raise PDFError(f"cannot parse PDF {Path(path).name}: {exc}") from exc
    text = "\n".join(pages).replace("\x00", "")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:max_chars]
