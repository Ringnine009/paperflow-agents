# Dynamic Belief Networks and Deep-Thinking Probes for Multi-agent Social Reasoning

## Overview

This paper proposes a framework combining a **Dynamic Belief Network (DBN)** and a **Deep-Thinking Token Ratio (DTR) probe** to address two longstanding problems in LLM-powered multi-agent social deduction: recursive agreement between homogeneous agents and stable detection of deception. The framework is evaluated in nine-player Werewolf across 3000 simulated games (six configurations × 500 games) using DeepSeek-V3.2 agents. The DBN maintains per-player suspicion estimates via exponential moving average (EMA) smoothing (α=0.3), while the DTR probe uses logit-lens to measure Jensen-Shannon divergence across transformer layers of a separate probe model (Qwen2.5-3B-Instruct) as a proxy for cognitive load. Results show that combining DBN with MaKTO-Proxy reasoning raises villager win rate from 44.2% to 68.8%, and adding DTR further improves vote accuracy (to 66.6%) but reduces survival, revealing a non-monotonic relationship between individual capability and collective payoff, interpreted through the lens of the handicap principle.

## Method

The framework consists of three modules combined across six configurations (A–F):

1. **MaKTO-Proxy**: few-shot chain-of-thought prompting to elicit reasoning.
2. **Dynamic Belief Network (DBN)**: per-player suspicion estimates updated each round via EMA smoothing with α=0.3, designed to dampen the positive-feedback loop between homogeneous models.
3. **Deep-Thinking Token Ratio (DTR) probe**: computes the average Jensen-Shannon divergence between layerwise logit-lens distributions of a separate probe model (Qwen2.5-3B-Instruct) to estimate cognitive load of utterances, relying on cross-architecture inference.

Experiments run 500 games per configuration in nine-player Werewolf with DeepSeek-V3.2 agents. Metrics include villager win rate, vote accuracy, mean survival rounds, and Brier Score convergence. Sensitivity analysis on α (0.1/0.3/0.5) is conducted under Configuration C.

## Key Claims & Evidence

- **Unassisted baseline achieves 44.2% villager win rate, lower than random.** **[supported]** — Table 2 confirms Group A win rate 44.2%; quote found verbatim.
- **Combining MaKTO-Proxy and DBN increases win rate to 68.8%.** **[supported]** — Table 2 confirms Group E win rate 68.8%, a ~24.6 pp increase over baseline.
- **Vote accuracy rises monotonically from 35.5% to 66.6%.** **[supported]** — Table 2 vote accuracy values confirm the monotonic rise across groups A→F.
- **Group F has the highest vote accuracy but the shortest survival.** **[supported]** — Table 2 confirms Group F vote accuracy 66.6% (highest) and mean survival 2.33 rounds (shortest).
- **DBN and MaKTO have orthogonal and additive effects.** **[supported]** — Paper reports combined gain ~24.6 pp, approximately equal to the sum of independent gains (~20 pp); acknowledged as approximate.
- **DTR contributes more vote accuracy in the presence of MaKTO.** **[supported]** — Paper explicitly reports +3.4 pp (C→D) without MaKTO vs +7.3 pp (E→F) with MaKTO, indicating positive interaction.
- **DBN improves Brier Score convergence.** **[supported]** — DBN groups (C–F) decline more rapidly from round 2; Group F falls below 0.05 by round 3.
- **α=0.3 is optimal for DBN.** **[supported]** — Sensitivity analysis under Configuration C shows lowest Brier Score and ideal BCR (1.63 rounds) at α=0.3.
- **Werewolf utterances have higher DTR density peaks.** **[supported]** — Figure 9 shows higher density peak for werewolves, consistent with deception being complex cognitive processing.
- **The full-module configuration leads to early elimination of the target agent.** **[supported]** — Survival rounds 2.33 vs 2.73 support the mechanism; quote found verbatim.
- **The capability-survival conflict is analogous to the handicap principle.** **[supported]** — Paper discusses Spence signaling theory and Zahavi handicap principle; quote found verbatim.

## Strengths

- **Clear, well-motivated problem framing**: The paper identifies two concrete failure modes (recursive agreement and deception detection) and designs modules targeting each.
- **Systematic ablation design**: Six configurations (A–F) allow clean attribution of each module's contribution, including interaction effects between DBN and DTR.
- **Multiple complementary metrics**: Win rate, vote accuracy, survival, and Brier Score convergence provide a richer picture than a single outcome measure.
- **Honest reporting of non-monotonic effects**: The paper does not hide that the best-identifying configuration (F) has the shortest survival, and interprets this through the handicap principle rather than glossing over it.
- **All 11 claims verified**: The Critic confirmed every claim is supported by verbatim quotes and surrounding context.

## Limitations

- **No statistical significance testing**: Win-rate and vote-accuracy differences (e.g., 68.8% vs 68.2% between E and F) are reported without confidence intervals or significance tests, so some differences may not be robust.
- **Limited game counts for sensitivity analyses**: The α sensitivity analysis uses only 500 games per condition (Configuration C); statistical power is unclear.
- **DTR distribution overlap limits discriminability**: The paper itself notes "a large amount of overlap, which restricts complete discriminability" between werewolf and villager DTR distributions, tempering the strength of the DTR signal.
- **Additivity claim is approximate**: The orthogonal/additive claim rests on ~24.6 pp vs ~20 pp sum of independent gains, which is approximate rather than exact.
- **Cross-architecture inference assumption**: DTR relies on logit-lens from a separate probe model (Qwen2.5-3B-Instruct) applied to utterances from DeepSeek-V3.2 agents; the validity of this cross-architecture cognitive-load proxy is asserted rather than validated.
- **No comparison to alternative belief-update mechanisms**: The DBN's EMA approach is not benchmarked against other belief models (e.g., Bayesian updates, particle filters), so its relative merit is unclear.
- **Critic output was available and fully incorporated**; no missing-verdict caveats apply.

## Related Work

- **Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction** (arXiv:2407.13943) — Bailis et al. introduce a Werewolf-based framework for evaluating LLM strategic reasoning, deception, and persuasion, directly relevant to the game setting and evaluation methodology used here.
- **A Survey of Multi-Agent Deep Reinforcement Learning with Communication** (arXiv:2203.08975) — Zhu et al. survey communication mechanisms in multi-agent systems, relevant to how agents coordinate and share (or conceal) information in social deduction.
- **Theory of Mind for Explainable Human-Robot Interaction** (arXiv:2512.23482) — Bauer et al. discuss ToM as a mechanism for inferring others' mental states, conceptually related to the DBN's suspicion-tracking of other agents' beliefs.
- **Honeypot Allocation for Cyber Deception in Dynamic Tactical Networks: A Game Theoretic Approach** (arXiv:2308.11817) — Sayed et al. model deception and belief dynamics in adversarial settings, relevant to the game-theoretic framing of deception detection.

## Relevance to My Research Direction

**Score: 8/10**

This paper is highly relevant to my focus on multi-agent LLM systems, social reasoning/theory of mind, and game-theoretic interaction. The DBN's per-agent suspicion tracking is a concrete instantiation of dynamic belief modeling (albeit a simple EMA rather than a full DBN), directly addressing the "dynamic belief models" thread of my research. The DTR probe's attempt to measure cognitive load as a deception signal is an innovative approach to evidence-driven claim verification in adversarial settings. The handicap-principle interpretation of the capability-survival tradeoff offers a compelling game-theoretic lens on how individual competence affects collective outcomes. However, the paper's reliance on a simplified EMA belief model (rather than a true probabilistic DBN) and the absence of significance testing temper its direct transferability to my work on rigorous belief-model evaluation.

## Suggested Next Steps

1. **Add statistical rigor**: Report confidence intervals and significance tests (e.g., bootstrap or permutation tests) for win-rate and vote-accuracy differences, especially between Groups E and F.
2. **Benchmark DBN against alternative belief models**: Compare EMA smoothing against Bayesian belief updates, particle filters, or learned belief estimators to establish the DBN's relative merit.
3. **Validate the DTR cross-architecture assumption**: Test whether logit-lens cognitive-load estimates from the probe model correlate with ground-truth deception labels and with the agent's own internal representations.
4. **Investigate the survival-vs-accuracy tradeoff**: Design interventions (e.g., calibrated self-concealment strategies) that preserve identification accuracy while improving survival, testing the handicap-principle hypothesis more directly.
5. **Scale up game counts**: Increase games per configuration and run the α-sensitivity analysis with more replicates to confirm the optimality of α=0.3.
6. **Explore heterogeneous agent populations**: Test whether mixing models of different capabilities alters the recursive-agreement and deception-detection dynamics.