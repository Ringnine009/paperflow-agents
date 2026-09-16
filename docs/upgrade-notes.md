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

### (b) 零 token/成本记账

`LLMClient.chat()` 仍然丢弃 `usage`，`FakeLLM` 返回的是桩数据（1/1/2），所以**离线无法测真实成本**。
`scripts/compare_arms.py` 只报可测的替代指标：LLM 调用次数与发送字符数（成本代理），
并显式输出 `tokens_measurable: false`。**本项未修复**，不谎报。

---

## 交付验证

```console
$ python -m pytest
149 passed, 3 deselected in 4.69s        # 原有 96 个离线测试全绿 + 新增 53 个

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
| E 三臂对照 | `test_compare_arms.py`：`No module named 'scripts.compare_arms'` | 3 passed |

---

## 未完成 / 存疑

1. **三臂对照的质量结论跑不了**（无付费 LLM）：offline 只能用脚本化 FakeLLM，质量列测的是脚本不是架构。
   已交付可复用对照工具 + 实测的「架构决定」列，实验设计见 [`baseline-plan.md`](baseline-plan.md)。
2. **token/成本记账未实现**：`chat()` 仍丢弃 `usage`（见中等问题 b）。
3. **DNS rebinding（TOCTOU）未防**：guard 只校验解析结果，未做连接级 IP pinning（已在 `net.py` 文档注明）。
4. **历史归档未被追改**：`examples/*/report.md`、`outputs/pf-*/report.md` 保留原始错误形态作为审计证据
   （因此新的一致性检查会在它们上面报 11/17 不合格——这是预期行为，不是回归）。
5. **引文核验的边界**：改写过的引文、只有数字被替换而措辞相同的引文，本核验器抓不到
   （已写入 README Limitations）。
