"""Configuration: .env loading and runtime settings.

Secrets discipline: the API key is only ever read from the environment or
from a git-ignored `.env` file. The loader never overrides variables that
are already set in the process environment, and it never writes anything.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Repository root: .../paperflow (parent of the `paperflow` package dir)
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The user's research direction, used by the Synthesizer to score relevance.
RESEARCH_FOCUS = (
    "multi-agent LLM systems, social reasoning / theory of mind in agents, "
    "game-theoretic interaction, dynamic belief models (e.g. DBN), "
    "and evidence-driven claim verification"
)

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: str | Path | None = None) -> Path | None:
    """Load a KEY=VALUE file into the environment (never overriding).

    Resolution order when `path` is None:
      1. the `PAPERFLOW_ENV_FILE` environment variable
      2. ``<repo root>/.env``
      3. ``<cwd>/.env``

    Returns the path that was loaded, or ``None`` if no file was found.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_file = os.environ.get("PAPERFLOW_ENV_FILE")
        if env_file:
            candidates.append(Path(env_file))
        candidates.append(REPO_ROOT / ".env")
        candidates.append(Path.cwd() / ".env")

    for candidate in candidates:
        if not candidate.is_file():
            continue
        # utf-8-sig strips the BOM some Windows editors prepend to .env
        for line in candidate.read_text(encoding="utf-8-sig").splitlines():
            match = _ENV_LINE.match(line)
            if match is None:  # comment or blank line
                continue
            key, value = match.group(1), match.group(2)
            if key in os.environ:  # existing env wins
                continue
            os.environ[key] = value.strip("\"'")
        return candidate
    return None


@dataclass
class Settings:
    """Runtime settings, read from the environment at construction time."""

    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("PAPERFLOW_API_BASE", "https://api.deepseek.com/v1"))
    model: str = field(default_factory=lambda: os.environ.get("PAPERFLOW_MODEL", "deepseek-chat"))
    http_timeout: int = field(default_factory=lambda: int(os.environ.get("PAPERFLOW_HTTP_TIMEOUT", "90")))
    #: cap on how many characters of paper full text are handed to agents
    max_fulltext_chars: int = field(default_factory=lambda: int(os.environ.get("PAPERFLOW_MAX_FULLTEXT_CHARS", "60000")))
    #: guard against infinite tool-call loops per agent
    max_tool_rounds: int = field(default_factory=lambda: int(os.environ.get("PAPERFLOW_MAX_TOOL_ROUNDS", "8")))
    max_output_tokens: int = field(default_factory=lambda: int(os.environ.get("PAPERFLOW_MAX_OUTPUT_TOKENS", "4000")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("PAPERFLOW_TEMPERATURE", "0.2")))

    def require_api_key(self) -> str:
        """Raise a helpful error if no DeepSeek key is configured."""
        if not self.deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or pass --env-file pointing at a .env that defines it."
            )
        return self.deepseek_api_key


def get_settings(reload: bool = False) -> Settings:
    """Return a Settings instance; cached unless `reload=True`."""
    global _CACHED_SETTINGS
    if reload or _CACHED_SETTINGS is None:
        _CACHED_SETTINGS = Settings()
    return _CACHED_SETTINGS


_CACHED_SETTINGS: Settings | None = None
