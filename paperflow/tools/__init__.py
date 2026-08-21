"""Default tool suite: everything the agent team can call.

``build_default_registry`` wires each tool to its implementation with the
JSON schemas the LLM needs. All tools are plain functions; the registry
turns them into OpenAI function-calling schemas.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import quote

from paperflow.config import Settings
from paperflow.core.tools import ToolRegistry, make_schema
from paperflow.tools import net, texttools
from paperflow.tools.parsers import parse_arxiv_atom, parse_crossref_work

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"


def _arxiv_search(query: str, max_results: int = 5, timeout: int = 45) -> str:
    """Search arXiv by arXiv query syntax (e.g. ``id:1706.03762``, ``ti:Transformer``)."""
    from paperflow.core.jsonutil import json_dumps

    q = query.strip()
    if q and ":" not in q.split()[0]:
        q = f"all:{q}"  # plain words -> default field
    url = f"{ARXIV_API}?search_query={quote(q)}&start=0&max_results={int(max_results)}&sortBy=relevance"
    net.throttle("export.arxiv.org", min_interval=3.0)
    items = parse_arxiv_atom(net.http_get(url, timeout=timeout))
    return json_dumps(items)


def _resolve_doi(doi: str, timeout: int = 45) -> str:
    """Resolve a DOI to metadata via Crossref, falling back to doi.org handles."""
    from paperflow.core.jsonutil import json_dumps

    doi = doi.strip()
    try:
        net.throttle("api.crossref.org", min_interval=1.0)
        work = parse_crossref_work(net.http_get(f"{CROSSREF_API}/{quote(doi, safe='')}", timeout=timeout))
        return json_dumps(work)
    except Exception as crossref_error:  # noqa: BLE001 - fall back below
        try:
            net.throttle("doi.org", min_interval=1.0)
            handle = net.http_get(f"https://doi.org/api/handles/{quote(doi, safe='')}", timeout=timeout)
            import json as _json

            values = _json.loads(handle).get("values", [])
            urls = [v.get("data", {}).get("value") for v in values if v.get("type") == "URL"]
            return json_dumps({"doi": doi, "url": urls[0] if urls else None, "title": None,
                               "authors": [], "published": None, "pdf_url": None,
                               "abstract": "", "fallback": True,
                               "crossref_error": str(crossref_error)})
        except Exception as exc:  # noqa: BLE001
            raise net.ToolNetError(f"DOI {doi!r} could not be resolved: {exc}") from exc


def _fetch_url(url: str, max_chars: int = 20000, timeout: int = 30) -> str:
    """Fetch a URL's text content (truncated) - useful for landing pages."""
    text = net.http_get(url, timeout=timeout)
    return text[: int(max_chars)]


def _extract_to_text_file(source_path: Path, cache_dir: str, max_chars: int) -> Path:
    """Extract PDF text once into a content-addressed .txt file.

    The tool hands the LLM the path of the extracted *text* (not the PDF),
    so downstream agents can read the paper body without re-parsing it.
    """
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:16]
    out = Path(cache_dir) / f"{digest}.txt"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(net.extract_pdf_text(source_path, max_chars=max_chars), encoding="utf-8")
    return out


def _pdf_extract_message(path: Path, cache_dir: str, max_chars: int) -> str:
    text_path = _extract_to_text_file(path, cache_dir, max_chars)
    text = text_path.read_text(encoding="utf-8")
    preview = text[:1500]
    return (
        f"TEXT_PATH: {text_path}\n"
        f"CHARS: {len(text)}\n"
        f"Preview (first 1500 chars):\n{preview}"
    )


def _fetch_pdf_text(url: str, cache_dir: str, timeout: int = 90, max_chars: int = 60000) -> str:
    """Download a PDF (cached) and report its extracted text + path."""
    net._assert_http_url(url)
    path = net.download_pdf(url, cache_dir, timeout=timeout)
    return _pdf_extract_message(path, cache_dir, max_chars=max_chars)


def _read_pdf(path: str, cache_dir: str, max_chars: int = 60000) -> str:
    """Extract text from a local PDF file."""
    return _pdf_extract_message(Path(path), cache_dir, max_chars=max_chars)


def build_default_registry(settings: Settings, cache_dir: str | Path) -> ToolRegistry:
    """Assemble the full tool suite with the given runtime settings."""
    cache_dir = str(cache_dir)
    max_chars = settings.max_fulltext_chars

    registry = ToolRegistry()
    registry.register_func(
        "arxiv_search",
        "Search arXiv (title/abstract/author/id). query uses arXiv syntax, e.g. "
        '"id:1706.03762", "ti:Transformer", "au:Vaswani", or plain words. Returns '
        "a JSON list of paper metadata incl. pdf_url.",
        make_schema(
            {
                "query": {"type": "string", "description": "arXiv search query"},
                "max_results": {"type": "integer", "description": "max results (1-10)", "default": 5},
            },
            required=["query"],
        ),
        _arxiv_search,
    )
    registry.register_func(
        "resolve_doi",
        "Resolve a DOI (e.g. 10.54254/2753-8818/2026.DL34010) to paper metadata: "
        "title, authors, published, and a PDF link when open access.",
        make_schema({"doi": {"type": "string", "description": "the DOI to resolve"}}, required=["doi"]),
        _resolve_doi,
    )
    registry.register_func(
        "fetch_url",
        "Fetch the text of any http(s) URL (truncated). Use for paper landing pages.",
        make_schema(
            {"url": {"type": "string"}, "max_chars": {"type": "integer", "default": 20000}},
            required=["url"],
        ),
        _fetch_url,
    )
    registry.register_func(
        "fetch_pdf_text",
        "Download a PDF from a URL (cached) and extract its text. Returns the path "
        "of the extracted text plus a preview. Always call this to obtain full text.",
        make_schema({"url": {"type": "string", "description": "direct PDF URL"}}, required=["url"]),
        lambda url: _fetch_pdf_text(url, cache_dir, max_chars=max_chars),
    )
    registry.register_func(
        "read_pdf",
        "Extract text from a LOCAL pdf file path. Returns the path of the extracted "
        "text plus a preview.",
        make_schema({"path": {"type": "string", "description": "absolute path to a local .pdf"}}, required=["path"]),
        lambda path: _read_pdf(path, cache_dir, max_chars=max_chars),
    )
    registry.register_func(
        "search_text",
        "Search the paper full-text file (given by path in your context) for a "
        "phrase and return the surrounding passages. Use to verify quotes.",
        make_schema(
            {
                "path": {"type": "string", "description": "path to the paper full-text file"},
                "query": {"type": "string", "description": "phrase to locate"},
                "window": {"type": "integer", "default": 300},
                "max_hits": {"type": "integer", "default": 3},
            },
            required=["path", "query"],
        ),
        texttools.search_text,
    )
    return registry
