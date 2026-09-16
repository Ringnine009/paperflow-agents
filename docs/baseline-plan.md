# Baseline experiment plan — three-arm comparison (A / B / C)

**状态：设计已定，质量结论未跑。** 离线能测的列已测（见下文表格），
涉及「模型写得好不好」的列**没有**测——原因在下面写清楚了，不伪造结果。

工具已就绪：`scripts/compare_arms.py`（离线可跑，`tests/test_compare_arms.py` 覆盖三臂产物）。
换掉 LLM 后端即可执行真正实验。

---

## 1. 假设

| 臂 | 架构 | 待验证的假设 |
| --- | --- | --- |
| **A** | 单次长 prompt 直接产出报告（无任务板、无 artifact、无工具循环） | 省调用，但**没有任何 claim 级可核验产物**：报告里每个数字都不可复查 |
| **B** | 现行四阶段流水线（Researcher→Reader→Critic→Synthesizer）+ 确定性核验 + 机器 ledger | 成本更高，但每条 claim 都带机器判定，且判定进入交付物 |
| **C** | 与 B 相同，但**关闭**确定性核验与报告注入（复刻修复前架构） | 成本与 B 接近，但「未核验」在产物里不可见 → 与 A 同样不可审计 |

## 2. 指标

| 指标 | 定义 | 是否架构决定 |
| --- | --- | --- |
| 结构完整度 | 报告含 8 个规定 `##` 标题的数量 | 否（取决于模型） |
| 核验命中率 | 引文在抽取全文中被代码找到的比例 | 是（B 才有；口径见 `docs/upgrade-notes.md`） |
| 可核验 claim 比例 | 带确定性状态的 claim / claim 总数 | 是 |
| 判定入交付物 | 报告是否含 `[unverified]`/`[unverifiable]` 与 ledger | 是 |
| 成本 | LLM 调用次数、发送字符数、真实 token/费用 | 前两项是；token/费用需真实 API |
| 人工复查成本 | 审阅者需要逐条回到原文核对的 claim 数 | 是（B 之后剩 0） |

## 3. 离线已测（架构决定列，真实数字）

```console
$ python scripts/compare_arms.py
arm                          llm_calls  prompt_chars  claims  checked  headings  [supported]  [unverified]
A_single_prompt                      1          1070       0        0         8            2             0
B_pipeline                           6         25426       2        2         8            1             2
C_pipeline_no_verification           6         23329       2        0         8            2             0
```

可下的结论（只涉及架构）：

- **A 的 claim 完全不可核验**：`claims_total = 0`——单次 prompt 不产出任何 claim 级产物，
  所以「核验命中率」对它无定义（不是 0%，是无定义）。
- **C 复刻了审计到的故障**：2/2 claim 在报告里写 `[supported]`，其中 1 条引文是编造的，
  而 `[unverified]` 出现 0 次、无 ledger——不可审计。
- **B 的代价可量化**：比 C 多 **2,097 字符**（约 +9%，≈520 估计 token）的 prompt 内容，
  换来 2/2 claim 的确定性状态与报告里的机器判定。这是本实验**唯一**可以放心引用的成本结论。
- 结构完整度三臂都是 8/8，因为报告文本来自脚本文案——**这一列在这里没有信息量**，
  它只证明测量装置好用。

## 4. 为什么质量结论现在跑不了

1. **没有付费 LLM 预算**：项目约束是不得调用付费 API（DeepSeek 计费）。
   离线 `FakeLLM` 的回答由测试脚本写死，所以「结构完整度」「叙述质量」测的是**我的脚本**，
   不是架构对模型的影响——把它当结论就是伪造。
2. **token/费用不可测**：`FakeLLM` 返回桩 `usage`（1/1/2）。真实记账需要 `LLMClient.chat()`
   保留 `usage`（当前丢弃，见 upgrade-notes 中等问题 b）+ 真实 API 调用。
3. **单次运行的方差**：LLM 输出的方差很大，任何质量结论都需要每臂 N≥5 次、固定论文集合与评分表；
   现在只能跑一次脚本，样本量本身也不够。
4. **评分需要人**：结构完整度可自动测，但「Limitations 是否真的切中要害」这类质量维度需要人工评分表，
   否则又是一次自我评判。

## 5. 真实验需要什么条件（可执行清单）

- [ ] 一个可用的付费/本地模型端点（DeepSeek 或本地 vLLM 均可），预算约 30 次运行 × 每臂，
      按 B 臂 ~6 次调用/run 估算 ≈ 540 次调用。
- [ ] `LLMClient.chat()` 记录 `usage`（prompt/completion tokens）并汇总到 board——
      这是实验的前置实现，尚未做。
- [ ] 固定论文集合：5 篇（含 1 篇只有摘要的付费墙论文，专门测问题 4 的抽象模式），全部走
      `fetch_pdf_text` 缓存，避免 arXiv 限流。
- [ ] 每臂 N=5 次重复（同一入口、同一 temperature），记录均值与极差。
- [ ] 评分表：结构完整度（自动）+ 逐条 claim 的可核验性（自动）+ 人工 3 人盲评
      「Limitations/Strengths 是否切题」（1–5 分）。
- [ ] 预注册判据：若 B 在「可核验 claim 比例」上显著高于 A/C，且 A/C 的成本优势 < 25%，
      则判定流水线 + 核验在可审计性上值得其成本；否则如实报告流水线不划算。

## 6. 换成真模型要改哪里

```python
# scripts/compare_arms.py
# 1) 把 FakeLLM(...) 换成 LLMClient.from_settings(settings)（需要 DEEPSEEK_API_KEY）
# 2) 三臂共用同一篇论文入口与同一 temperature
# 3) A 臂保持单次 chat()；B/C 臂沿用 Pipeline.run()（C 臂由 verification_disabled() 上下文管理器关闭核验）
# 4) 输出仍是 docs/arm-comparison.json 的同构结构，指标列不变
```

`verification_disabled()` 是一个仅在实验内生效的上下文管理器，退出时恢复被替换的
`ReaderAgent.after_run` 与 `SynthesizerAgent` 的报告注入函数——实验不会把线上行为留在被改写状态
（`tests/test_compare_arms.py` 覆盖了三臂的产物差异）。
