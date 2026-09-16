"""Published DeepSeek prices, in CNY per 1M tokens.

Kept in one place so the experiment's cost column is auditable: every number
in `docs/arm-comparison-live.md` is `tokens x these prices`, and the prices
carry the URL they were read from and the date they were read.

DeepSeek bills input twice (cache hit / cache miss) and output once, so the
client ledger keeps the same three buckets. Peak and off-peak differ by 2x
(peak = 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri). The experiment reports
**peak** prices, i.e. the worst case, so the quoted cost is never flattering.
"""

from __future__ import annotations

from dataclasses import dataclass

#: where these numbers come from (read 2026-09-16)
PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing"
PRICE_READ_ON = "2026-09-16"

#: USD -> CNY used to convert the published USD prices; DeepSeek's own top-up
#: page prices the balance in CNY. Kept explicit so the conversion is auditable.
USD_TO_CNY = 7.1


@dataclass(frozen=True)
class Pricing:
    """Price per 1M tokens, in CNY."""

    cache_hit_per_m: float
    cache_miss_per_m: float
    output_per_m: float
    label: str = ""

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "cache_hit_cny_per_m": self.cache_hit_per_m,
            "cache_miss_cny_per_m": self.cache_miss_per_m,
            "output_cny_per_m": self.output_per_m,
        }


def _cny(usd_per_m: float) -> float:
    return round(usd_per_m * USD_TO_CNY, 6)


#: deepseek-flash, peak hours: $0.03 / $0.30 / $1.20 per 1M
PRICING_PEAK_CNY = Pricing(
    cache_hit_per_m=_cny(0.03),
    cache_miss_per_m=_cny(0.30),
    output_per_m=_cny(1.20),
    label="deepseek-flash peak",
)

#: deepseek-flash, off-peak (half of peak)
PRICING_OFFPEAK_CNY = Pricing(
    cache_hit_per_m=_cny(0.015),
    cache_miss_per_m=_cny(0.15),
    output_per_m=_cny(0.60),
    label="deepseek-flash off-peak",
)


def cost_cny(totals: dict, pricing: Pricing = PRICING_PEAK_CNY) -> float:
    """Cost in CNY of a usage-totals dict, rounded to 6 decimals.

    Accepts the ledger's totals (``cache_hit_tokens`` / ``cache_miss_tokens``
    / ``completion_tokens``).
    """
    hit = float(totals.get("cache_hit_tokens", 0) or 0)
    miss = float(totals.get("cache_miss_tokens", 0) or 0)
    out = float(totals.get("completion_tokens", 0) or 0)
    return round(
        (
            hit * pricing.cache_hit_per_m
            + miss * pricing.cache_miss_per_m
            + out * pricing.output_per_m
        )
        / 1_000_000,
        6,
    )
