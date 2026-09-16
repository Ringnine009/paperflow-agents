"""Ground truth for the paper under test, transcribed before any arm was run.

The experiment needs a scoring key that does not come from the model being
scored. This module is that key: the paper's own Table 2, its stated
contribution list and the DOIs/ids it cites, transcribed by hand from
``research/werewolf-multiagent-paper.pdf`` (extracted text, 21,902 chars).

**Discipline:** this file is written *before* the runs. If a scored report
disagrees with it, the report is wrong - the key is never adjusted after the
fact to make an arm look better. Every entry below is traceable to a line of
the paper; the section it came from is named in ``source``.

Used by :mod:`scripts.compare_arms_live` for the deterministic columns only
(contribution coverage, number-to-configuration attribution). Nothing here is
scored by an LLM.
"""

from __future__ import annotations

#: the paper under test
PAPER = {
    "title": "Dynamic Belief Networks and Deep-Thinking Probes for Multi-agent Social Reasoning",
    "author": "Zhenxiao Guo",
    "affiliation": "Tongji University",
    "venue": "Proceedings of CONF-MPCS 2026 Symposium",
    "doi": "10.54254/2753-8818/2026.DL34010",
    "pages": "203-212",
}

#: Table 2 "Main metrics in six configurations", as printed.
#: configuration letter -> (win rate %, vote accuracy %, survival rounds)
TABLE2 = {
    "A": {"name": "Vanilla", "win_rate": "44.2", "vote_accuracy": "35.5", "survival": "2.73", "modules": []},
    "B": {"name": "MaKTO", "win_rate": "54.6", "vote_accuracy": "44.8", "survival": "2.78", "modules": ["MaKTO-Proxy"]},
    "C": {"name": "DBN", "win_rate": "53.8", "vote_accuracy": "45.7", "survival": "2.67", "modules": ["DBN"]},
    "D": {"name": "DBN+DTR", "win_rate": "58.6", "vote_accuracy": "49.1", "survival": "2.61", "modules": ["DBN", "DTR"]},
    "E": {"name": "MaKTO+DBN", "win_rate": "68.8", "vote_accuracy": "59.3", "survival": "2.59", "modules": ["MaKTO-Proxy", "DBN"]},
    "F": {"name": "MaKTO+DBN+DTR", "win_rate": "68.2", "vote_accuracy": "66.6", "survival": "2.33", "modules": ["MaKTO-Proxy", "DBN", "DTR"]},
}

#: Table 1 module matrix - which modules each configuration switches on.
TABLE1_MODULES = {
    "A": {"MaKTO-Proxy": False, "DBN": False, "DTR": False},
    "B": {"MaKTO-Proxy": True, "DBN": False, "DTR": False},
    "C": {"MaKTO-Proxy": False, "DBN": True, "DTR": False},
    "D": {"MaKTO-Proxy": False, "DBN": True, "DTR": True},
    "E": {"MaKTO-Proxy": True, "DBN": True, "DTR": False},
    "F": {"MaKTO-Proxy": True, "DBN": True, "DTR": True},
}

#: every number the paper states as a headline magnitude (win rate, vote
#: accuracy, survival, BCR, α, pp deltas, N). A percentage in a report that is
#: in none of these sets is a candidate fabrication.
HEADLINE_NUMBERS = {
    "win_rates": {"44.2", "54.6", "53.8", "58.6", "68.8", "68.2"},
    "vote_accuracy": {"35.5", "44.8", "45.7", "49.1", "59.3", "66.6"},
    "survival": {"2.73", "2.78", "2.67", "2.61", "2.59", "2.33"},
    "bcr": {"1.77", "1.60", "1.63", "1.43", "1.38", "1.71", "1.68"},
    "pp_deltas": {"24.6", "23.8", "19.5", "9.3", "10.2", "3.4", "7.3", "4.8", "0.6", "0.15", "0.08", "0.05"},
    "scales": {"3000", "500", "9", "36", "512", "500", "0.3", "0.1", "0.5", "20"},
    # bands/approximations the prose uses ("over 53%", "close to 69%", "44%-55%")
    "approximate": {"44", "53", "55", "69", "0.20", "0.23", "0.05", "0.10", "2.5"},
}

#: The contribution checklist: what a complete review of this paper must
#: contain. ``statement`` is one honest sentence; ``groups`` are the token
#: groups a report has to contain (in any order, all of them) for the
#: contribution to count as covered. Written before scoring; not tuned to any
#: arm's phrasing.
CONTRIBUTIONS = [
    {
        "id": "problem",
        "statement": "Frames the two problems: recursive agreement between homogeneous LLM agents, and stable detection of deception.",
        "groups": [
            ["recursive agreement", "recursive consensus", "echo chamber", "groupthink", "positive-feedback", "positive feedback"],
            ["deception", "deceptive"],
        ],
        "source": "Abstract; Introduction (para 4)",
    },
    {
        "id": "dbn_mechanism",
        "statement": "DBN keeps a per-player suspicion estimate updated by exponential moving average across rounds.",
        "groups": [
            ["suspicion"],
            ["exponential moving average", "ema"],
        ],
        "source": "Abstract; 2.3",
    },
    {
        "id": "dbn_alpha",
        "statement": "The EMA weight is alpha = 0.3 for groups C-F, chosen by a 0.1/0.3/0.5 sensitivity check.",
        "groups": [
            ["0.3", "α = 0.3", "alpha = 0.3"],
            ["sensitivity", "0.1", "0.5"],
        ],
        "source": "2.3; 3.2, Figure 8",
    },
    {
        "id": "dtr_probe",
        "statement": "A cross-model DTR probe reads layerwise logit-lens distributions of a separately deployed Qwen2.5-3B-Instruct as a cognitive-load proxy.",
        "groups": [
            ["dtr", "deep-thinking token ratio"],
            ["logit", "logit-lens"],
        ],
        # entity names go through plain substring matching on the canonical
        # text: tokenization splits "Qwen2.5-3B-Instruct" on its hyphens, so a
        # report may legitimately write it slightly differently
        "substrings": ["qwen"],
        "source": "Abstract; 2.4",
    },
    {
        "id": "setup",
        "statement": "Nine-player Werewolf on AgentScope with DeepSeek-V3.2 agents; six configurations of 500 games each, 3000 in total.",
        "groups": [
            ["nine-player", "nine player", "9-player", "9 player", "nine players"],
            ["agentscope"],
            ["3000", "3,000"],
        ],
        "source": "Abstract; 2.1",
    },
    {
        "id": "win_rate",
        "statement": "Villager win rate rises from a 44.2% baseline (below random) to 68.8% for MaKTO-Proxy + DBN.",
        "groups": [
            ["44.2"],
            ["68.8"],
        ],
        "source": "Abstract; 3.1, Table 2",
    },
    {
        "id": "vote_accuracy",
        "statement": "Vote accuracy rises monotonically from 35.5% (A) to 66.6% (F).",
        "groups": [
            ["35.5"],
            ["66.6"],
        ],
        "source": "3.1, Table 2",
    },
    {
        "id": "additive",
        "statement": "MaKTO-Proxy and DBN gains are additive/orthogonal: +24.6 pp together against ~20 pp of separate gains.",
        "groups": [
            # the combination word AND a magnitude, so a passing mention of
            # "additive" in an unrelated sentence cannot score this point
            ["orthogonal", "additive", "additivity", "independent", "19.5", "24.6", "23.8"],
            ["24.6", "23.8", "19.5", "9.3", "10.2", "orthogonal stacking", "orthogonal contribution"],
        ],
        "source": "3.1; 4",
    },
    {
        "id": "dtr_marginal",
        "statement": "DTR adds vote accuracy in both settings: +3.4 pp without MaKTO (C to D) and +7.3 pp with it (E to F).",
        "groups": [
            ["3.4", "7.3"],
            ["vote accuracy", "vote-accuracy", "accuracy"],
        ],
        "source": "3.1, Figure 6; 4; 5",
    },
    {
        "id": "brier",
        "statement": "Brier score converges faster with DBN, below 0.05 by round 3 for F.",
        "groups": [
            ["brier"],
            ["0.05", "converge", "convergence"],
        ],
        "source": "Abstract; 3.2, Figure 7",
    },
    {
        "id": "handicap",
        "statement": "The capability-survival conflict: F has the best vote accuracy but the lowest survival, read through the handicap principle.",
        "groups": [
            ["handicap", "signaling", "signalling"],
            ["survival"],
        ],
        "source": "Abstract; 3.1; 4, Discussion",
    },
    {
        "id": "bc_vs_e",
        "statement": "F's win rate (68.2%) is slightly below E's (68.8%) because F is eliminated earlier (2.33 vs 2.59 rounds).",
        "groups": [
            ["68.2"],
            ["2.33"],
        ],
        "source": "3.1; 4",
    },
    {
        "id": "limitations",
        "statement": "Limitations: a single game condition and homogeneous model backends, so cross-game/cross-model generalization is untested.",
        "groups": [
            ["single game", "one game", "homogeneous", "generalization", "generalisation"],
            ["limitation", "not tested", "untested", "future"],
        ],
        "source": "5, Conclusions (limitations)",
    },
    {
        "id": "future_work",
        "statement": "Future work: dynamic alpha scheduling, multi-round DTR aggregation, heterogeneous agent populations, other games (Avalon/Diplomacy).",
        "groups": [
            ["dynamic alpha", "alpha scheduling", "avalon", "diplomacy", "heterogeneous"],
        ],
        "source": "5, Conclusions (future directions)",
    },
]

#: DOIs / arXiv ids the paper cites. A "Related Work" section that invents an
#: id is checkable without any LLM.
CITED_IDS = {
    "arxiv": {
        "2502.06060", "2309.04658", "2512.09187", "2501.14225",
        "2404.01602", "2601.04832", "2602.13517",
    },
    "doi": {"10.54254/2753-8818/2026.DL34010"},
    "loose": {"1706.03762"},  # not cited here; used only by unrelated fixtures
}

#: one dict so scripts can hand the whole key around
PAPER_FACTS = {
    "paper": PAPER,
    "doi": PAPER["doi"],
    "table1_modules": TABLE1_MODULES,
    "table2": TABLE2,
    "headline_numbers": HEADLINE_NUMBERS,
    "contributions": CONTRIBUTIONS,
    "cited_ids": CITED_IDS,
}
