# Upgrade notes — 审计修复记录（P0×4 + 中等问题）

本文记录一次针对「宣传 > 实现」风险的全量审计修复：每一项都按 **问题 → 失败测试（红灯）→
修复 → 真实数字 → 面试可讲点** 写清楚，所有数字都来自仓库内可复现的命令输出。

复现全部数字（离线，不调付费 API）：

```bash
python scripts/recompute_verification.py --json docs/verification-before-after.json   # 全量归档重算
python scripts/compare_arms.py --json docs/arm-comparison.json                        # 三臂对照
python scripts/demo_ssrf_guard.py                                                     # SSRF 实测
python scripts/demo_report_verification.py                                            # 报告注入实测
python -m pytest                                                                      # 离线全量测试
```

---

## 问题 1（高）：引文核验器 92% 假阴性 —— 77.9% → 99.2%

### 问题

核验器本身的设计是对的：`str.find` 逐字比对、不经 LLM。但审计发现 26 次「未命中」里 24 次是
**核验器自己的错**。在归档的 131 条 claim 上重算，旧实现只命中 **102/131 = 77.9%**。

三个真实根因（都来自归档 PDF 抽取文本，不是假想场景）：

| 根因 | 归档原文 | 旧实现看到的 |
| --- | --- | --- |
| PDF 换行连字符 | `WMT 2014 English-\nto-German` | `english- to-german`，与引文 `English-to-German` 不等 |
| 断词连字符 | `sur-\nprisingly well` | `sur- prisingly well` |
| 抽取噪声插入数字 | `when the sequence\n6\nlength n` | 句子中间多了一个页码 `6` |
| 引文省略逗号 | 原文 `ensembles, by over 2 BLEU` | 引文写 `ensembles by over 2 BLEU` |
| 引文中段省略 | `the big model ... outperforms ...` | 整串子串查找必然失败 |

最刺眼的一例：`examples/attention-is-all-you-need/critic_output.json` 里 LLM 自己写着
"Although the quote_verified flag was false, the text is present verbatim in the paper"，
而 after_run 用错误的机器判定**否决了正确的语义证据**。

### 失败测试（红灯）

新增 `tests/test_quote_matching.py`（fixture 直接读归档的 `examples/*/artifacts/full_text.txt`）：

```console
$ python -m pytest tests/test_quote_matching.py -q
FAILED tests/test_quote_matching.py::test_archived_false_negatives_are_now_found[0..5]
FAILED ... test_line_break_hyphen_is_repaired
FAILED ... test_soft_hyphen_and_zero_width_are_stripped
FAILED ... test_page_number_noise_between_words_is_skipped
FAILED ... test_inline_punctuation_difference_is_tolerated
FAILED ... test_elided_quote_matches_segment_wise
FAILED ... test_genuinely_absent_quote_still_fails_on_real_archive
14 failed, 4 passed
```

其中**反向测试是重点**：真正不存在的引文必须继续失败（防止把核验器放宽到永远通过）。

### 修复

`paperflow/tools/texttools.py` 改成三级匹配，每级都在结果里显式标注 `match_mode`，
**修复过的命中绝不会被伪装成逐字命中**：

1. `verbatim` —— 原行为（空白归一 + 小写后子串查找）。
2. `hyphen_repaired` —— 去掉软连字符/零宽字符、连接换行处连字符、并**对双方**去掉词内连字符；
   仍然是纯子串查找（`English-\nto-German` 与 `English-to-German` 都归一为 `englishtogerman`）。
3. `token_aligned` —— 按 token 顺序匹配，允许跳过的**只能是**裸小整数（页码噪声），且有预算上限
   （`min(4, len//4)`）；标点不参与 token 比较（`ensembles,` ≡ `ensembles`）。
   引文里的数字永远必须命中，篡改数字必然失败。
4. 含省略号的引文按段顺序匹配，结果里 `segments > 1`，模式带 `_segments` 后缀。

同时收紧了下限：`MIN_QUOTE_CHARS` 4 → 8、新增 `MIN_QUOTE_TOKENS = 2`（归档里最短的真实引文是
43 字符 / 7 token，因此这一步对真实数据零损失，却堵住了「引文 = `BLEU` 也算通过」的洞）。

### 真实数字（全量 19 个归档 run 重算）

`scripts/recompute_verification.py` 用**冻结的旧实现**（`verify_quote_legacy`，逐字复制修复前代码）
与当前实现分别重算同一批 claim：

| | claims | 命中 | 命中率 |
| --- | ---: | ---: | ---: |
| 修复前（冻结旧实现） | 131 | 102 | **77.9%** |
| 修复后（当前实现） | 131 | 130 | **99.2%** |

- 新命中 28 条：`hyphen_repaired` 18、`token_aligned` 9、`verbatim_segments` 1（其余 102 条仍是逐字命中）。
- 全量结果：`verbatim` 102、`hyphen_repaired` 18、`token_aligned` 9、`verbatim_segments` 1。
- 仍然未命中 **1** 条：`44.2% without beliefs and 68.8% with dynamic beliefs` —— 该 run 的
  `full_text.txt` 里确实没有这句话（LLM 编的），**应该继续失败**。
- 可信度校验：冻结旧实现重算的结果与归档里存下的 `quote_verified` 标记 **117/117 完全一致**，
  说明对比基准是忠实的（`docs/verification-before-after.json` 的 `legacy_reproduces_stored`）。

### 面试可讲点

- 「我的核验器 92% 的失败是自己的 bug」——把**核验器的准确率当产品指标**来测，而不是只看功能有没有跑通；
  主动做了反向测试（真不存在的引文必须失败、数字被改必须失败、同段落词沙拉必须失败），证明放宽的是
  PDF 噪声容忍度，不是判定标准。
- 三级 tier + `match_mode` 上报：修复过的命中永远不会和逐字命中混为一谈，可审计。
- 用**冻结的旧实现**做 before/after，且用归档里已存的标记交叉验证基准的忠实性——数字不是回忆出来的。

---

## 问题 2（高）：确定性判定不进交付物 —— 38 条 claim 在报告里仍写 [supported]

### 问题

降级只写进 `critic_output.json`，人读的 `report.md` 不受影响：7 个含降级的 run 里 6 个报告
仍然写 `[supported]`，且 `unverified` 在报告中出现 0 次。README 的 "regardless of what any LLM says"
只对 JSON 成立。

用新增的一致性检查**回溯审计归档报告**（`scripts/recompute_verification.py` 末尾输出）：

```console
=== report/verification consistency of the archived reports (new check) ===
attention-is-all-you-need    claims=7 non-verified=3 violations=7
    - report.md has no '## Verification Ledger' section, so its 3 non-verified claim(s) are invisible to the reader
    - claim #0 (unverified) is missing from the verification ledger
...
reports with at least one violation: 11/17
violation kinds: {"bullet-still-supported": 38, "ledger-entry-missing": 40, "no-ledger": 11}
```

即：**11 个报告不合格，其中 38 条已被机器降级的 claim 在报告正文里仍被写成 supported。**

### 失败测试（红灯）

新增 `tests/test_report_verification.py`：

```console
$ python -m pytest tests/test_report_verification.py -q
E   ModuleNotFoundError: No module named 'paperflow.agents.verification'
1 error in 0.01s
```

（新模块的典型红灯：被测行为尚不存在。修复前的**行为**证据即上面归档重算的 38 条。）

### 修复

新增 `paperflow/agents/verification.py`，两层都由**代码**而不是 LLM 负责：

1. `inject_machine_verdicts(board, markdown)`：报告写完后重写它——能对齐到「机器判定为问题」的
   claim bullet 会被替换成 `**[unverified]**`/`**[unverifiable]**`，并追加一个完全由代码生成的
   `## Verification Ledger` 小节（含每条未核验 claim 的引文、判定原因、Critic 结论）。幂等。
2. `check_report_consistency(board, markdown)`：报告生成后校验文本与机器判定是否一致，返回违规清单；
   `SynthesizerAgent.save_output` 把违规写进 board log（`level="warning"`），不静默吞掉。

bullet↔claim 对齐用多键打分（完全措辞 → 引文片段 → token 覆盖率 ≥ 0.5），因为 Synthesizer 习惯改写；
实测在归档报告上 **64/66 = 97%** 的 bullet 能对齐（旧的「按原文包含」匹配在公开样例上是 0/7）。

### 真实数字

```
$ python scripts/demo_report_verification.py
- **The model attains a 99.9% win rate with zero variance.** **[unverified]** — reported in Table 2.
...
## Verification Ledger
- Deterministic quote check: **1/2** claims carry a quote that was located in the paper text.
- **1 claim(s) could not be verified** and must not be read as supported evidence:
  - **[unverified]** claim #1: The model reaches a 99.9% win rate with zero variance
    - quote: "the win rate reaches 99.9% with zero variance"
    - deterministic check: quote not found in the paper text
--- board log ---
... synthesizer report carries 1/2 machine-unverified claim(s); 1 report bullet(s) corrected
```

- LLM 写的 `[supported]` 被改写为 `[unverified]`：1/1 条可对齐的反例。
- 一致性检查回溯归档：发现 11/17 报告不合格、38 条 bullet 与机器判定矛盾（修复前无任何机制能发现）。
- **未做**：归档里的历史报告**不会**被追改（改写历史比暴露问题更糟）。新 run 才带 ledger。

### 语料偏差（必须一起读）

19 个归档 run 主要来自两篇论文（Transformer 与作者的狼人杀论文），28 条新命中里 18 条是同一篇
论文里重复出现的引文。因此 **99.2% 应读作「在这批归档上」**，不是普适准确率；要得到普适数字需要
跨会议/跨排版的多来源语料，见 [`baseline-plan.md`](baseline-plan.md) 第 5 节的实验条件。

### 面试可讲点

- 「确定性判定必须进入交付物，否则等于没有」——把 `report.md` 当成被测对象，而不是只看 JSON artifact。
- 拒绝「让 LLM 照抄状态」：状态由代码注入 + 代码校验，LLM 只负责叙述。
- 用同一条检查回溯审计历史产物，把「有多少条 claim 被谎报为 supported」量化成 38，而不是印象。

---

## 问题 3（高）：SSRF 防护未覆盖 `fetch_url`

### 问题

`_assert_http_url` 只在 `fetch_pdf_text` / `download_pdf` 生效；`_fetch_url` 直接走 `net.http_get`，
且 `requests` 默认跟随重定向。

### 失败测试（红灯）

新增 `tests/test_fetch_tools_ssrf.py`（含**真实起在 127.0.0.1 的本地服务**做靶机）：

```console
$ python -m pytest tests/test_fetch_tools_ssrf.py -q
FAILED ... test_fetch_url_refuses_a_loopback_service          # 本地服务内容被读回
FAILED ... test_fetch_url_refuses_private_and_link_local[...] # 5 个私网/元数据地址
FAILED ... test_fetch_url_refuses_non_http_schemes[...]       # file:// ftp:// gopher://
FAILED ... test_fetch_url_validates_every_redirect_hop        # 302 -> 169.254.169.254 被跟随
FAILED ... test_fetch_url_validates_redirect_hops_to_loopback
FAILED ... test_download_pdf_also_validates_redirect_hops
15 failed
```

### 修复

`paperflow/tools/net.py`：

- 新增 `http_get_bytes_checked` / `http_get_checked`：**每一跳**都过 `_assert_http_url`，
  用 `allow_redirects=False` 手动跟随（`urljoin` 解析相对 Location），最多 5 跳，超限报错。
- `download_pdf` 改走 checked 通道；`fetch_url` 改走 `http_get_checked`（重试次数降到 1，避免恶意主机用 429
  让 agent 睡在退避里）。
- `http_get` / `http_get_bytes` 保留给固定的 arXiv/Crossref API 用，文档里明确写「不要传用户 URL」。
- 文档写明**未修复**的残余风险：DNS rebinding（校验时解析公网、请求时解析私网）需要连接级 IP pinning。

### 真实数字（实测）

```
$ python scripts/demo_ssrf_guard.py
target: http://127.0.0.1:51811/admin
1. control - the service is live: True
2. pre-fix call path (net.http_get, what fetch_url used): 'INTERNAL-SECRET-CANARY (internal admin p'
3. fetch_url: refused -> blocked host ('127.0.0.1' resolves to non-public address 127.0.0.1) - SSRF guard rejects private/loopback/link-local addresses
4. what the agent sees: TOOL ERROR: ToolNetError: blocked host ('127.0.0.1' resolves to non-public address 127.0.0.1)
```

第 2 行即修复前的泄漏路径（可复现），第 3/4 行是修复后的实际拒绝；`call_safe` 让 agent 看到
`TOOL ERROR` 而不是崩溃。

### 面试可讲点

- 安全边界要问「**哪些入口没走这道检查**」——同一个模块里三个函数，只有一个漏了，README 却按最全的口径宣传。
- 重定向是 SSRF 的经典绕过：`allow_redirects=False` + 逐跳校验，而不是只看第一个 URL。
- 主动在文档里写下未修复的 TOCTOU/DNS rebinding 残余风险，而不是宣称「已防住 SSRF」。

---

## 问题 4（高）：抽象摘要模式静默跳过核验

### 问题

拿不到全文时 `ReaderAgent.after_run` 直接 `return`，于是 `quote_verified` 这个 key **根本不存在**；
Critic 的 `claim.get("quote_verified") is False` 永不命中 → 无降级、无日志、无标记。
归档中有 3 个 run 的 14 条 claim 处于这种「没有注解」状态，它们的 Critic 结论全是 supported。
README 却写着「the Critic marks claims *unverifiable*」——宣传先于实现。

### 失败测试（红灯）

新增 `tests/test_abstract_mode.py`：

```console
$ python -m pytest tests/test_abstract_mode.py -q
FAILED test_reader_marks_claims_unverifiable_without_full_text   # 注解完全缺失
FAILED test_reader_logs_the_unverifiable_count
FAILED test_reader_still_reports_verified_when_the_text_is_present
FAILED test_critic_calls_uncheckable_claims_unverifiable
FAILED test_critic_never_assumes_supported_without_an_annotation # 缺注解 ≠ 通过
FAILED test_abstract_mode_status_reaches_the_report
6 failed
```

### 修复

- `verify_quote` 返回显式 `status`：`verified` / `unverified`（查了、没找到）/ `unverifiable`（没得查）。
- `verify_claims` 把 `quote_status` 写进 claim；**空全文是受支持的输入**，此时每条 claim 都被显式标为
  `unverifiable`。
- `ReaderAgent.after_run` 不再提前返回，日志同时给出 `n/N quotes found` 与 `m/N unverifiable`。
- `CriticAgent.after_run`：`unverifiable` → 结论 `unverifiable`；`unverified` → `unverified`；
  **完全没有注解** → 也判 `unverifiable`（「不知道」就报「不知道」，绝不默认 supported）。
- 报告侧：这类 claim 进 ledger 时带 `[unverifiable]` 标记与「no full text」原因。

**这是一次有意的契约收紧**：`tests/test_agents_offline.py::test_critic_verifies_claims_with_search_text`
里手工写的 reader 产物原先没有注解，现按新契约补上真实注解（该引文确实在 fixture 全文里），断言未改动。

### 真实数字

- 无全文 run：`quote_status = "unverifiable"`（2/2 claim），board log 出现
  `quote verification: 0/2 quotes found verbatim in full text; 2/2 claim(s) unverifiable`。
- 有全文且引文命中：`quote_status = "verified"`，两种状态在报告里可区分（`[verified]` vs `[unverifiable]`）。
- README 的 Limitations 语义（Critic 标 unverifiable）从「宣传」变成「实现」。

### 面试可讲点

- 「key 不存在」比「值为 false」更危险：它是**沉默**的通过。用三态显式状态取代布尔缺省。
- 把 README 里已经写出、代码却没实现的行为当成测试用例来源（文档即规格）。
- 契约收紧时同步更新既有测试的**前置数据**（补注解），而不是放宽断言。

---

## 中等问题

### (a) 报告 ↔ 核验联动失效（0/7）—— 已修复，42% → 95%

旧联动（前端 `paperflow/web/static/timeline.js::linkClaims`）靠「报告块包含 claim 原文」精确匹配，
Synthesizer 习惯改写措辞 → 审计的公开样例上 **0/7** 命中。

修复：抽出同一套打分（精确包含 = 1；否则 claim 的 token 覆盖率，阈值 0.6、≥4 个 token），
贪心分配保证「一条 bullet 只连一条 claim」，精确命中优先。量化（归档 17 个 run、118 条 claim bullet）：

| | 命中 bullet | 占比 |
| --- | ---: | ---: |
| 修复前（精确包含） | 50 | 42% |
| 修复后（+ token 覆盖率回退） | 112 | **95%** |

其中 `examples/attention-is-all-you-need`（审计点名的样例）从 **0/7 → 7/7**。
Python 侧的机器 ledger 对齐单独测过：64/66 = 97%（阈值 0.5，因为它还有 ledger 兜底）。
js 侧阈值更严（0.6），且只在回退路径生效，避免误连。

### (b) 零 token/成本记账 —— **已修复**（见问题 5）

`LLMClient.chat()` 曾经丢弃 `usage`，`FakeLLM` 返回桩数据（1/1/2），离线无法测真实成本。
现已实现逐次真实记账（含 DeepSeek 的 cache hit/miss 分桶）与请求前的硬预算熔断，
线上实验据此完成，实测总花费 ¥2.33（首轮 N=3 用 ¥1.34，扩样到 N=5 再用 ¥0.99）。详见下文问题 5。

---

## 问题 5（高）：唯一缺失的关键证据 —— 真实模型下的三臂对照

### 问题

项目此前**只有架构维度的数据，质量维度零证据**。`docs/arm-comparison.json` 是
`FakeLLM` 脚本化产物，文件自己声明 `tokens_measurable: false` 且
`quality_note: SCRIPTED model behaviour ... not evidence about model quality`。
面试必问：「你这四个 Agent 和写四个 prompt 有什么差别？和单次长 prompt 比有增益吗？」
——此前无法用数据回答。

### 设计

预注册协议见 [`arm-comparison-live-design.md`](arm-comparison-live-design.md)
（在跑之前写死：臂定义、指标口径、成本口径、成功判据、停止规则）。要点：

* 同一篇论文（作者本人 Werewolf 论文，21,902 字符，**无截断**）、同一任务文本、同一模型
  （`deepseek-chat` → 实际服务 `deepseek-flash`）、同一 temperature 0.2、thinking 显式关闭；
* 三臂 A（单次长 prompt）/ B（现行流水线 + 确定性核验）/ C（同流水线但关闭核验，复刻修复前架构）；
* 每臂独立重复，A→B→C 交错执行，预算熔断时网格仍保持平衡。**先跑 3 次（当时的自设下限），
  后按「局限第 5 条」自费扩到 5 次**（见下方「扩样到 N=5」，只补买缺失的 2 次重复）；
* **主结论列全部确定性、不经 LLM**：章节完整度、claim 引文可定位率、数字归属错误、
  覆盖度（14 条预先冻结的贡献清单，每条要求「概念 + 量级」同时命中）；
* LLM 只用于次要的「引文是否真支持论断」抽样列，并**测量其一致性**：每批换序判两次。

### 为跑通实验先做的实现（红→绿）

| 任务 | 红灯 | 绿灯 |
| --- | --- | --- |
| F token/成本记账 + 预算熔断 | `test_llm_usage.py`：`ImportError: cannot import name 'BudgetExceeded'`（8 个用例全红） | 8 passed（全量 211 passed） |
| G 实验测量核心 + 网格 | `test_compare_arms_live.py`：`No module named 'scripts.compare_arms_live'` | 25 passed |
| H 语义判官解析与一致性 | `test_judge_semantic_support.py` | 18 passed |

`LLMClient` 现在：累加 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` /
`completion_tokens`（缺 cache 明细时整段按**贵的** miss 计，未知绝不按折扣算）；
按公开价折算人民币；**在发请求之前**用「本次输入 + max_tokens 输出」的最坏情况校验预算，
超限抛 `BudgetExceeded` 且请求不发出——不可能跑完才发现超支。

### 真实数字

```console
$ python scripts/compare_arms_live.py --runs 3 --budget 30 \
      --out docs/arm-comparison-live.json --runs-dir outputs/arm-comparison-live
arm spend: 0.8063 CNY   judge spend: 0.5342 CNY   total: 1.3405 CNY (cap 30.0)   ok=9 failed=0 skipped=0
arm                          ok     calls          tokens in/out        CNY   coverage  errors  support
A_single_prompt              3/3      1.0      5,991 /  2,773     0.0253     14.0/14     0.0      —
B_pipeline                   3/3      9.0     43,623 /  8,652     0.1157     13.0/14     0.0      —
C_pipeline_no_verification   3/3     10.7     50,707 / 10,245     0.1277     13.7/14     0.0      —

# 扩样（只买缺失的第 4、5 次重复，已存的 9 次直接复用）
$ python scripts/compare_arms_live.py --extend-from docs/arm-comparison-live-n3.json \
      --runs 5 --budget 10 --out docs/arm-comparison-live.json --runs-dir outputs/arm-comparison-live
bought by this extension: 0.9897 CNY (0.5305 arms + 0.4591 judge); cumulative for the experiment: 2.3302 CNY
arm                          ok     calls          tokens in/out        CNY   coverage  errors  support
A_single_prompt              5/5      1.0      5,991 /  2,899     0.0264     14.0/14     0.0     1.00
B_pipeline                   5/5      9.0     41,116 /  8,509     0.1138     13.4/14     0.0     0.75
C_pipeline_no_verification   5/5     10.8     50,703 / 10,001     0.1271     13.0/14     0.0     0.75
```

| 维度 | A 单次长 prompt | B 流水线 | C 关闭核验 |
| --- | ---: | ---: | ---: |
| 成本/次（N=5） | **¥0.0264** | ¥0.1138（**4.31×**） | ¥0.1271（4.81×） |
| 覆盖度（14 条中） | **14.0**（14–14） | 13.4（13–14） | 13.0（12–14） |
| 确定性事实错误 | 0.0 | 0.0 | 0.0 |
| 8 个章节齐全 | 8/8 | 8/8 | 8/8 |
| claim 带程序可定位引文 | 不存在 claim | **9.8/9.8 = 100%** | 0（核验被关闭） |
| 报告含机器 ledger | **0/5** | **5/5** | 0/5 |

**结论：流水线没有买到质量，买到的是可核验性与可审计性，代价 4.31 倍成本。**
事实错误三臂全为 0，八个章节每轮齐全。流水线独有的东西是：每条 claim 都带一条**程序**能在
原文定位到的引文，以及一份**由代码而不是由模型**写进报告的判定 ledger。

### 扩样到 N=5：它改掉了一个结论，也查出一个产物完整性问题

3 次重复是当时的自设下限，报告里把它写成了局限第 5 条。花 **¥0.99**（预算 ¥10，用掉 9.9%）
再买 2 次重复/臂，得到的是三件事：

1. **覆盖度差异被证伪，而不是被确认。** N=3 时写的是「A 每轮都不低于 B（14/14 vs 13/14）」，
   两臂区间不相交。到 N=5：A 保持 14/14（5/5 轮满分），B 为 14、13、14、13、13 —— 区间
   **[13,14] 与 A 的 [14,14] 重叠**，覆盖度不再区分两臂；B 的漏点在 `brier` / `future_work` /
   `dbn_alpha` 之间移动，**不是同一个盲点**。也就是说，上一轮那条「A 的覆盖度优势」是 3 个样本
   撑起来的，扩样后必须撤回。这是本轮扩样唯一改掉的结论。
2. **成本倍数稳定。** 4.57× → **4.31×**，仍落在 N=3 的逐次比值带 [4.05, 5.55] 内，
   而比值带本身从 1.50 变宽到 2.34（更多样本可以带出新的极值，方向要如实报告）。
3. **出现一个此前没见过的臂内事件**：B 在第 5 次重复里「程序可定位引文占比」掉到 **0.0**
   （此前最低 0.1），即那一轮报告的 claim 条目没有一条能被程序在原文定位。这不改变架构结论
   （该列三臂区间仍然重叠），但说明流水线的引文可定位率本身波动很大，N=3 的 0.177 是高估。

**同时查出一个产物完整性问题（比结论更要紧）**：扩样要对已存报告重新评分，于是把「已存报告」
与「上一轮发布的字数」逐条比对 —— 9 个 N=3 run 目录里有 **4 个**已经不再是当初被测量的文本
（`A_single_prompt-r1/-r2`、`B_pipeline-r1`、`C_pipeline_no_verification-r1`，文件时间戳
06:02–06:04Z，晚于冻结网格自己的 `started_at` 05:57:59Z）。其中 B-r1 的新文本实际覆盖 14/14，
C-r1 的新文本只覆盖 12/14 —— 也就是说**上一轮发布的 N=3 覆盖度数字与今天磁盘上的报告已经不是
同一份证据**。处理方式：冻结件 `docs/arm-comparison-live-n3.json` / `-n3.md` 原样保留（用
`git show HEAD:` 取出，blob 哈希与提交一致），N=5 报告里的 N=3 列一律用「同一套评分器重评
磁盘报告」的口径，并在报告 §5.1 并列两种读法与各文件时间戳；同时把这项检查做成代码
（`stored_report_integrity()`，有独立红灯用例），而不是写在 README 里的一句话。

扩样的工程实现同样是红→绿：先写失败测试（`--runs 5` 断言每臂 5 次、报告生成要求每臂 ≥5 次、
只买缺失重复、区间重叠判定、完整性检查），红灯证据是
`ImportError: cannot import name 'InsufficientRuns'` 与
`plan_for(5) -> 9 runs: {A:3, B:3, C:3}` —— 旧实现把 3 次的调度表切片，`--runs 5` 会**静默只买
3 次**，这是扩样前必须拆掉的雷。全量测试 224 passed。

### 过程中发现并修掉的两个测量缺陷（重要，别只报好听的数）

第一次网格的评分器给出 A/B/C 各 1.7 / 0.7 / 2.3 个「事实错误」。逐条核对发现**全是测量 bug**：

1. **比较句被误判**：`Group E: A=44.2%, B=54.6%, C=53.8%, E=68.8%` 这种对照表，
   旧逻辑把句中出现的每个组名与每个百分数两两比对，一句话产生 5 个假「张冠李戴」。
   改为只判定**与标签绑定**的数字（`group E = 68.8%` / `group E reached 68.8%`），
   未绑定的数字只查「是否存在于原文」。
2. **可由原文推出的算术被当成编造**：`E is +10.2 pp over D`、`F trails E by 0.6 pp`
   都是对 Table 2 做减法，不是编造。现在用 key 里的数值预计算差集/和集，可推出的值放行；
   真正推不出的（如 `41.0 pp`）仍然报警。

修完 scorer 后用 `--score-only` **对同一批已存报告重新评分**（不重买样本、不发一次请求，
并验证了幂等），九个报告全部零事实错误。这件事本身是个面试点：**评分器也要被审计**。

同样发现并修复的接线 bug：流水线的 claim 列表没有传进判官采样器，导致 B/C 的判官样本
引文全为空、全判 `unclear`——报告里 B 的 support rate 一度是 0.0。修好后 B/C 均为
两次判定完全一致（κ = 1.00）。

### 已知局限（必须一起读）

* **样本小**：1 篇论文 × 1 个模型 × **5 次重复**。除成本列外，各臂在所有质量列上的
  min–max 区间都互相重叠；区分不开的列只能写「本样本内未测得差异」，不是「A 比 B 好」。
  n=5 下任何显著性检验都不成立，报告里也不做。
* **判官问题不对称**：A 没有引文可判，只能问「原文是否包含该论断」（较松），
  B/C 问的是「这条引文是否支持该论断」（较严）。因此 support rate **不可跨臂比较**，
  本文只做臂内使用与一致性报告。
* **错误检测只认 `%`/`pp` 且需绑定配置**：写错方法、作者、机制它看不见。
  「0 错误」应读作「无此类算术/归属错误」。
* **`arxiv_search` 被打桩**：流水线的 related-work 工具循环未被考察。

### 面试可讲点

1. **敢报负面结论**：花了钱做真实对照，结论是「多阶段流水线不提升质量，只提升可核验性，
   成本 4.3–4.6 倍」。这比再贴一张架构图有力得多——它证明我会为了答案而不是为了好看去做实验。
2. **扩样后主动撤回自己上一轮的结论**：N=3 时得出的「A 的覆盖度每轮都不低于 B」在 N=5 被
   证伪（区间重叠）。我在报告里写明「那条优势是 3 个样本撑起来的，本轮撤回」——
   **花自己的钱去否掉自己的结论**，比任何「结果很漂亮」都更能说明评测是诚实的。
3. **连自己的存档都审计**：扩样时发现 4/9 个已存报告与上一轮发布的数字已不是同一份文本，
   于是把「已存报告 vs 已发布字数」做成代码里的完整性检查（带时间戳），报告里并列两种口径
   而不是挑一个好看的。
4. **预注册 + 事先冻结清单**：覆盖度清单、错误判定规则、成功判据都在跑之前写死在
   `paper_ground_truth.py` / `arm-comparison-live-design.md`，事后没改过一个字。
5. **主结论不依赖 LLM**：三个主指标全是确定性检查；LLM 只用于次要列，且**测量了它的一致性**
   （换序双判，首轮 κ = 1.00，扩样重复上 κ = 0.90/0.88），并明确声明该列不可跨臂比较。
6. **评分器被审计过**：第一次的「事实错误」全是我的测量 bug，我逐条核对、修掉、
   重算并保留了这一段——「一个会说自己数据错了的评测框架」比「一个从不出错的评测框架」可信。
7. **成本是可核验的**：token 来自 API 的 `usage`（含 cache 分桶），价格来自官方文档并注明读取日期，
   按**峰值**价（更贵的一档）折算成人民币；扩样预算 ¥10 只花掉 ¥0.99（9.9%）。

---

## 问题 6（中）：测试隔离缺陷 —— `load_dotenv` 污染进程环境

### 现象

`paperflow.config.load_dotenv()` 的**设计行为**就是写 `os.environ`（.env 就是这样生效的），
而 `tests/test_config.py::test_load_dotenv_parses_key_value` 直接调用真实的
`load_dotenv(临时文件)` 且不做清理。于是它把 `DEEPSEEK_API_KEY=sk-test-123`、
`PAPERFLOW_MODEL=deepseek-chat` 留在进程环境里给后续测试用。**结果依赖测试执行顺序**：
之后任何读这两个变量的测试（`test_web_app.py` 会读 key）看到的都是上一个测试的假值，
而 `load_dotenv` 又会因为「已有环境变量优先」而**静默拒绝**覆盖——失败模式很隐蔽。

复现证据（红灯）：一次全量运行里该用例失败为
`AssertionError: assert 'sk-test-123' == 'sk-leaky-key'`——临时文件里的值根本没写进去，
因为前面那个用例已经把同名变量占住了。

### 根因

环境变量是**进程级全局状态**，而 pytest 默认不做快照；本项目有多个测试直接
（或经由被调代码）写 `os.environ`。`monkeypatch` 只能撤销它自己设置过的键，
管不了被调库代码直接写的键。

### 修法

`tests/conftest.py` 增加 `autouse=True` 的 `restore_environ` fixture：
每个用例前快照整个 `os.environ`，用例结束（含异常）后删除新增键、还原被改键。
**没有放宽任何断言、没有加 xfail**，泄漏本身仍然被测试断言着：

* `test_dotenv_load_does_not_leak_into_the_next_test`——断言泄漏在用例内**真实发生**；
* `test_the_previous_test_s_dotenv_leak_is_gone`——断言它**没有活过**用例边界。

### 防复发

* fixture 是 autouse 的，新增测试自动受保护，不需要记得写 `monkeypatch`；
* 两个方向的断言都在，任何人删掉 fixture 都会立刻变红；
* 全量套件连续跑 3 次：`211 passed, 3 deselected`（稳定，无随机失败）。

### 面试可讲点

「全跑失败、单跑通过」是测试隔离问题的典型信号，但这次要诚实区分两种情况：
我确实发现并修掉了一个**真实的**环境变量泄漏（有红灯证据），
而最初报来的那个 `AttributeError` 是**编辑窗口期的陈旧观察**——`calls` 属性在
测试写完之后才补上，所以那次失败是「测试先落地、实现在后」的正常 TDD 中间态，
不是顺序依赖。能区分这两者，比笼统说一句「修好了」有价值。

---

## 交付验证

```console
$ python -m pytest
211 passed, 3 deselected in 5.83s        # 原有 149 + 新增 62

$ python -m pytest -m smoke -q
...                                      # 3 个 smoke 全过（本次 arXiv/Crossref 未限流，无 429）

$ node tests/test_markdown.mjs && node tests/test_timeline.mjs
markdown renderer tests OK
timeline/claim-linking tests OK
```

红灯 → 绿灯一览：

| 任务 | 红灯证据 | 绿灯 |
| --- | --- | --- |
| A 引文核验 | `test_quote_matching.py`：14 failed | 18 passed |
| B 判定入交付物 | `test_report_verification.py`：collection error（模块不存在；行为证据：归档 38 条 bullet 与机器判定矛盾） | 9 passed |
| C SSRF | `test_fetch_tools_ssrf.py`：15 failed（本地服务内容真的被读回） | 17 passed |
| D 摘要模式 | `test_abstract_mode.py`：6 failed | 6 passed |
| E 三臂对照（离线） | `test_compare_arms.py`：`No module named 'scripts.compare_arms'` | 3 passed |
| F token 记账 + 预算熔断 | `test_llm_usage.py`：`ImportError: BudgetExceeded`（8 红） | 8 passed |
| G 线上实验测量核心 | `test_compare_arms_live.py`：`No module named 'scripts.compare_arms_live'` | 25 passed |
| H 语义判官解析/一致性 | `test_judge_semantic_support.py`：`cannot import name 'parse_reasons'` | 18 passed |
| I 环境隔离 | 全量运行中 `test_dotenv_load_does_not_leak_into_the_next_test` 红 | 全量 211 passed ×3 |
| J 扩样到 N=5 | `test_compare_arms_live.py`：`ImportError: cannot import name 'InsufficientRuns'`；`plan_for(5) -> 9 runs: {A:3, B:3, C:3}`（旧实现切片，`--runs 5` 会静默只买 3 次） | 13 个新用例绿，全量 **224 passed** |

---

## 未完成 / 存疑

1. ~~**三臂对照的质量结论跑不了**（无付费 LLM）~~ → **已完成**，见问题 5。真实模型、3 臂 × **5 次重复**、
   真实 token 与人民币记账，结论与局限在 [`arm-comparison-live.md`](arm-comparison-live.md)。
   仍然存疑的部分：样本只有 1 篇论文 × 1 个模型 × 5 次重复，除成本列外各臂质量区间互相重叠；
   判官问题跨臂不对称，support rate 不可跨臂比较；错误检测只覆盖 `%`/`pp` 类数字；
   `arxiv_search` 被打桩，related-work 质量未测；**9 个 N=3 run 目录里有 4 个的报告已被覆写**
   （发布数字仍冻结在 `arm-comparison-live-n3.json`，两种口径并列在报告 §5.1）。
2. ~~**token/成本记账未实现**~~ → **已完成**，见问题 5：`usage` 三桶累加、公开价折算、
   请求前硬预算熔断（`LLMClient(budget_cny=...)`）。首轮 ¥1.34 + 扩样 ¥0.99 = **实测总花费 ¥2.33**。
3. **DNS rebinding（TOCTOU）未防**：guard 只校验解析结果，未做连接级 IP pinning（已在 `net.py` 文档注明）。
4. **历史归档未被追改**：`examples/*/report.md`、`outputs/pf-*/report.md` 保留原始错误形态作为审计证据
   （因此新的一致性检查会在它们上面报 11/17 不合格——这是预期行为，不是回归）。
5. **引文核验的边界**：改写过的引文、只有数字被替换而措辞相同的引文，本核验器抓不到
   （已写入 README Limitations）。
6. **线上实验只覆盖本地 PDF 入口**：三臂都用 `read_pdf` 起手，`arxiv` / `doi` / `url` 入口与
   SSRF 防护、arXiv 限流重试等路径未被这次实验覆盖（它们由离线测试覆盖）。
