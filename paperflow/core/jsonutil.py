"""Small JSON utilities shared by agents."""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    """Extract the first JSON value embedded in a model response.

    LLMs frequently wrap JSON in ```json fences or add prose around it;
    this walks the string looking for a balanced ``{...}`` / ``[...]``
    block and parses it.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip fenced code blocks
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break  # unbalanced inside; keep scanning
            start = text.find(open_ch, start + 1)
    raise ValueError(f"no JSON object found in model output: {text[:200]!r}")


def json_dumps(obj: Any) -> str:
    """Compact JSON with unicode preserved (titles/authors are often CJK)."""
    return json.dumps(obj, ensure_ascii=False, default=str)
