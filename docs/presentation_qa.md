# Presentation Q&A prep (中英双语) — Who Wants to Be a PoliMillionaire?

Likely professor questions + tight answers. 每条「Q/A」英文在前,「问/答」中文在后(只为对照,临场可中可英)。
Numbers are from our own runs (cite the cell on screen).

**Headline facts to anchor every answer:** Qwen2.5-7B-Instruct, 4-bit, local on a Colab T4, greedy,
**no external LLM APIs**, **30 s/question**. Baseline ≈ 87% (dev). Live sweep **26/32 = 81%** (News 92%,
Philosophy 90%, Science 83%). Leaderboard: 15/15/15 for Entertainment/History/Science & Philosophy, News 12,
Maths 11.

**每个回答都先抛这些硬数据:** Qwen2.5-7B-Instruct、4-bit、Colab T4 本地、贪心解码、**不调外部大模型 API**、**每题 30 秒**。
离线基线≈87%。实时扫场 **26/32 = 81%**(News 92%、Philosophy 90%、Science 83%)。排行榜:Entertainment/History/Science/
Philosophy 满级 15,News 12,Maths 11。

---

## A · Design & model choice / 设计与模型选择

**Q: Why Qwen2.5-7B, and why 4-bit?**
A 7B instruct model is the largest that loads and answers within **30 s on a free T4**. 4-bit nf4
(bitsandbytes, bf16 compute) cuts it to ~6 GB VRAM with negligible quality loss at our decode lengths.
Bigger models either don't fit or breach the time wall at ~11 tok/s.
**问:为什么选 Qwen2.5-7B,为什么用 4-bit?**
答:7B 指令模型是免费 T4 上能在 **30 秒内加载并作答**的最大模型。4-bit nf4(bitsandbytes、bf16 计算)把显存压到约 6 GB,在
我们的输出长度下质量几乎无损。更大的模型要么放不下,要么在约 11 tok/s 的速度下会超时。

**Q: Why greedy decoding (temperature 0)?**
Determinism — reproducible runs and stable logs — and speed. We *did* test sampling for self-consistency
voting (§5); it timed out against the wall and the sampled chains repeated the **same** mistake, so greedy
won on both axes.
**问:为什么用贪心解码(temperature 0)?**
答:为了确定性(可复现、日志稳定)和速度。我们确实试过采样来做自洽投票(§5),但它会超时,而且采样出的多条链犯**同一个错**,
所以贪心在这两点上都更优。

**Q: What is temperature 0 used?**

**What it is:** temperature scales the logits before softmax. High → flatter, more random; low → sharper;
**→ 0 collapses the distribution to its peak, so the model always picks the single most-likely token —
deterministic (same prompt → same answer)**. Trade-off: a single greedy pass reports confidence ≈ 1.0 on
everything → that's our 

**它是什么:** temperature 在 softmax 之前缩放 logits。高 → 更平、更随机;低 → 更尖;**→ 0 把分布塌缩到峰值,模型永远选概率
最大的那个 token——确定性(同样 prompt → 同样答案)**。代价:单遍贪心几乎对所有题都报告置信度 ≈ 1.0 → 这就是我们的**过度自信**
问题;投票能校准它,但在时限处会超时。

---

## B · Constraints compliance / 规则合规

**Q: Is your RAG content "raw and non-generated" as required?**
Yes. We fetch **headlines, article bodies, and encyclopedia text verbatim** and put them in the prompt. No
summariser, no second model.
**问:你们的 RAG 内容是否符合"原始、未经生成"的要求?**
答:是。我们抓取**标题、文章正文、百科文本的原文**直接放进 prompt。没有摘要器,没有第二个模型。

**Q: 30-second limit — how do you guarantee it?**
Every pipeline stage is timed against a budget; the runner is **crash-safe and always submits something**
even if a stage fails or runs long. We warm up the model **before** any timed question so the cold start
(~1–2 min) never counts. Live latencies were ~2–7 s/turn — comfortable margin.
**问:30 秒限制怎么保证?**
答:每个流水线阶段都有计时预算;运行器**崩溃安全,即使某阶段失败或超时也一定会提交答案**。我们在**任何计时题之前**预热模型,
所以约 1–2 分钟的冷启动不计入。实时每回合约 2–7 秒,余量充足。

---

## C · Pipeline architecture / 流水线架构

**Q: Walk me through one question.**
`QAPipeline.answer()` runs 7 stages: **classify** the topic → optional **deterministic solver** → optional
**retrieval** → **build prompt** (strategy chosen per topic) → **generate** (local Qwen) → optional **tool**
(calculator) → **parse** the letter. Each stage is optional and timed; dependency-injected so offline and
live share the *exact* same pipeline.
**问:讲一道题是怎么走的。**
答:`QAPipeline.answer()` 跑 7 个阶段:**分类**主题 → 可选**确定性求解器** → 可选**检索** → **构建 prompt**(按主题选策略)→
**生成**(本地 Qwen)→ 可选**工具**(计算器)→ **解析**字母。每阶段都可选且计时;用依赖注入,所以离线和实时**共用完全相同**的流水线。

**Q: Offline vs live — same code?**
Identical pipeline and logging (one `EvalRecord` per turn). The only switch is `config.mode='live'` plus a
logged-in game client (`run_session`). That's the whole offline⇄live diff — one line.
**问:离线和实时是同一套代码吗?**
答:流水线和日志完全一致(每回合一条 `EvalRecord`)。唯一开关是 `config.mode='live'` 加一个已登录的游戏客户端(`run_session`)。
离线⇄实时的全部差异就这一行。

---

## D · Prompt engineering / Prompt 工程(notebook 2)

**Q: What did prompt strategy actually change?**
On the dev set: zero-shot and few-shot both **87%**; chain-of-thought **91.3%**. The entire gain is **Maths:
0.50 → 1.00**. Knowledge topics were already saturated, so prompting only moves the needle where reasoning
matters.
**问:prompt 策略到底改变了什么?**
答:开发集上:zero-shot 和 few-shot 都 **87%**;思维链 **91.3%**。提升**全部来自 Maths:0.50 → 1.00**。知识类题目本就饱和,
所以 prompt 只在需要推理的地方起作用。

**Q: What is `cot_v2` and why not just CoT?**
`cot_v2` is our live chain-of-thought, forced out by **two real logged failures**: (1) the model reasoned
correctly but wrote `Answer: B` when the matching numbers were in option C — so we force it to **match every
number, not just the conclusion**; (2) it wrote paragraphs of LaTeX and **ran out of tokens before the
`Answer:` line** — so we **ban LaTeX, cap the steps, and force a final `Answer:` line**.
**问:`cot_v2` 是什么,为什么不用普通 CoT?**
答:`cot_v2` 是我们的实时思维链,由**两个真实日志失败**逼出来的:(1) 模型推理正确却写了 `Answer: B`,而匹配的数字其实在选项
C——所以我们强制它**核对每一个数字,而不只是结论**;(2) 它写了一大段 LaTeX,**在写出 `Answer:` 之前就用光了 token**——所以我们
**禁用 LaTeX、限制步数、强制最后一行写 `Answer:`**。

**Q: Did CoT ever hurt?**
Yes — honest finding. On **News**, CoT *regressed* (zero-shot recalled dates correctly; CoT over-reasoned
Zelensky/Brexit into wrong answers). That's why the live router does **not** force CoT on recall topics.
**问:CoT 有没有起反效果?**
答:有——这是个诚实的发现。在 **News** 上,CoT **变差**了(zero-shot 能正确回忆日期;CoT 把 Zelensky/Brexit 过度推理成了错的)。
所以实时路由器**不会**在回忆类题目上强制 CoT。

---

## E · Maths, the solver & routing / 数学、求解器与路由(notebook 4)

**Q: Is the "calculator" just `eval()`?**
No — it's a **safe AST evaluator**: we parse the expression and walk only arithmetic nodes (`+ - * / ** %`).
`eval()` is never called, so arbitrary code can't run. (`src/tools/calculator.py`, `math_solvers.py`.)
**问:那个"计算器"是不是就是 `eval()`?**
答:不是——它是**安全的 AST 求值器**:我们解析表达式,只遍历算术节点(`+ - * / ** %`)。从不调用 `eval()`,所以无法执行任意代码。
(`src/tools/calculator.py`、`math_solvers.py`。)

**Q: How does the deterministic solver avoid being wrong?**
Each solver is **type-specific and abstains by default**. It fires only when it (a) recognises the question
type (finite-field roots, gcd, percentages, weekdays, …) and (b) its computed value matches **exactly one**
option. Otherwise it returns `None` and the LLM handles it. So it can help but never silently overrides.
**问:确定性求解器怎么保证不出错?**
答:每个求解器都**针对特定类型、默认弃权**。它只在 (a) 识别出题型(有限域求根、gcd、百分比、星期…)且 (b) 计算结果恰好匹配
**唯一一个**选项时才触发。否则返回 `None`,交给 LLM。所以它能帮忙,但绝不会悄悄覆盖。

**Q: What did the routing experiment find?**
Four prompt conditions over **40 labelled Maths questions**, with a category×condition heatmap. The finding
is deliberately **conservative**: only **interval-counting and temporal** questions benefit from the
structured prompt; everything else stays on `cot_v2`, which never truncates at the wall.
**问:路由实验得出了什么?**
答:在 **40 道带标注的 Maths 题**上跑 4 种 prompt 条件,做了类别×条件热力图。结论刻意**保守**:只有**区间计数和时间类**题目从
结构化 prompt 受益;其余都留在 `cot_v2` 上,它不会在时限处被截断。我们没有用小样本过拟合出一个花哨的路由器。

---

## F · RAG / News(notebooks 3 & 5)— **name the APIs / 点名 API**

**Q: Name your retrieval sources.**
**Google News RSS** (headlines), the **Guardian Content API** (raw article bodies), and a **headless Chromium
browser via Playwright** to open other articles past the consent wall; **Wikipedia** and a local **FAISS
index over Simple-Wikipedia** (plus a **BM25 index over English Wikipedia**) as fallbacks.
**问:说出你们的检索源。**
答:**Google News RSS**(标题)、**Guardian Content API**(文章正文原文)、**Playwright 驱动的无头 Chromium 浏览器**(越过同意弹窗
打开其他文章);兜底是 **Wikipedia** 和本地的 **Simple-Wikipedia FAISS 索引**(外加一个**英文维基的 BM25 索引**)。

**Q: Why a browser at all?**
The post-cutoff answer often lives in the article **body**, but the Google-News link is a JS consent-wall
redirect `requests` can't pass, and DuckDuckGo is blocked on the Colab IP. A real browser runs the JS and
reads the rendered text. The Guardian API is tried **first** (one ~0.2 s call) and the browser is the fallback.
**问:为什么需要浏览器?**
答:截止日之后的答案常在文章**正文**里,但 Google News 的链接是 JS 同意弹窗跳转,`requests` 过不去,而 DuckDuckGo 在 Colab IP
被封。真实浏览器会执行 JS 并读取渲染后的文本。Guardian API **优先**尝试(一次约 0.2 秒调用),浏览器是兜底。

**Q: Does RAG actually help, or is it decoration?**
The RAG-vs-no-RAG ablation shows it helps where the model's prior is post-cutoff — News scored **92%** live
and retrieval fired on **all 13** News questions. On topics inside the model's knowledge (Science), retrieval
stays off.
**问:RAG 真的有用,还是装饰?**
答:RAG 对比无 RAG 的消融显示:在模型先验是截止日之后的地方它确实有用——News 实时 **92%**,检索在**全部 13 题**上触发。在模型
本就掌握的主题(Science),检索保持关闭。

**Q: How do you keep retrieval within 30 s?**
Tight timeouts per fetch, a bounded number of bodies, and a cascade that stops early when a date-windowed
headline query already answers. A failed fetch degrades gracefully to headline-only — never blocks the turn.
**问:检索怎么控制在 30 秒内?**
答:每次抓取都有紧超时,正文数量有上限,且采用级联——当带日期窗口的标题查询已能作答时就提前停止。抓取失败会优雅降级为只用标题,
绝不阻塞这一回合。

---

## G · Agentic AI(notebook 3 §5)— the honest finding / 诚实的发现

**Q: You mention voting and tools but say you use neither live — why include them?**
Because the assignment asks us to *investigate* agentic techniques, and a **negative result is still a
result**. **Self-consistency** (N sampled CoT chains voting) **timed out** at the wall and its chains shared
the same slip; the **calculator** only helps arithmetic, where a single good `cot_v2` pass already won. We
kept the code and the analysis but shipped the simpler config — a documented engineering judgement.
**问:你们提到了投票和工具,却说实时都没用——那为什么还放进来?**
答:因为作业要求我们**研究** agentic 技术,而**负面结果也是结果**。**自洽**(N 条采样思维链投票)在时限处**超时**,且多条链犯同一个
错;**计算器**只对算术有用,而那里一次好的 `cot_v2` 已经赢了。我们保留了代码和分析,但上线了更简单的配置——这是有据可查的工程判断。

**Q: What is the confidence number then?**
For voting it's the **vote share** (e.g. 2/3) — a real calibration signal. In the shipped single-pass config
confidence is the model's own, which exposes the **overconfidence** problem below.
**问:那个置信度数字是什么?**
答:投票时它是**得票比例**(如 2/3)——一个真实的校准信号。在上线的单遍配置里,置信度是模型自己的,这恰好暴露了下面的**过度自信**问题。

---

## H · Evaluation & limitations / 评估与局限

**Q: Biggest weakness?**
**Overconfidence** — the model is confidently wrong on what it doesn't know (e.g. the "most moons" question:
it says Jupiter, but Saturn overtook it post-cutoff). Confidence ≠ correctness, so we can't safely use it to
abstain. RAG addresses the *knowledge* half; calibration is open work.
**问:最大的弱点是什么?**
答:**过度自信**——模型对自己不知道的东西自信地答错(比如"最多卫星"那题:它答木星,但土星在截止日后反超了)。置信度 ≠ 正确率,
所以不能安全地用它来弃权。RAG 解决了**知识**那一半;校准仍是未解决的工作。

**Q: Your live sample sizes are tiny (1–13 per topic). Isn't 81% noisy?**
Yes — we're explicit about it. The dev set (23 Qs) and the routing study (40 Qs) carry the statistical
claims; the live sweep demonstrates the **end-to-end contract** (server feeds, we submit, server grades) and
the leaderboard levels, not a precise accuracy estimate.
**问:你们实时样本很小(每主题 1–13 题),81% 不会很噪声吗?**
答:会——我们明确承认。统计性结论由开发集(23 题)和路由研究(40 题)承担;实时扫场展示的是**端到端契约**(服务器下发、我们提交、
服务器判分)和排行榜关卡,而不是精确的准确率估计。我们不从小的实时样本过度下结论。

**Q: Offline 87% vs live 81% — why the gap?**
Different, harder question distribution and small live n. The per-topic pattern is consistent: knowledge
topics strong, Maths the hardest. No contradiction with the dev-set prediction.
**问:离线 87% 对实时 81%,为什么有差距?**
答:题目分布不同且更难,加上实时 n 很小。各主题的规律是一致的:知识类强、Maths 最难。与开发集的预测不矛盾。

**Q: Reproducibility?**
Greedy decoding, one typed `RunConfig` logged per run, one JSONL `EvalRecord` per turn, fixed seeds where
sampling is used. Any run can be replayed and every wrong answer is dumped with the option we picked.
**问:可复现性?**
答:贪心解码,每次运行记录一个带类型的 `RunConfig`,每回合一条 JSONL `EvalRecord`,用到采样的地方固定随机种子。任何运行都可重放,
每道错题都连同我们选的选项一并导出。

---

## I · Quick-fire / 快问快答(one-liners)

- **Why not fine-tune? / 为什么不微调?** No training budget/time on a T4, and the rules reward
  prompting/agentic work; we invest there. / T4 上没有训练预算和时间,且规则鼓励 prompt/agentic 工作,我们把精力投在那。
- **Why 4-bit not 8-bit? / 为什么 4-bit 而非 8-bit?** 8-bit doesn't leave enough T4 headroom for activations +
  KV cache at our lengths. / 8-bit 在我们的长度下留给激活和 KV cache 的 T4 余量不够。
- **What's `Option.id` about? / `Option.id` 是什么?** We submit by integer option id, not letter — matches the
  game API contract (notebook 3 §7 verifies it). / 我们按整数选项 id 提交,而非字母,符合游戏 API 契约(notebook 3 §7 已验证)。
- **Game server rate-limits? / 服务器限流怎么办?** We pace requests politely (a pause between competitions) and
  save real leaderboard pushes for after the ~1-week-pre-deadline reset. / 我们礼貌地控制请求节奏(赛道间暂停),把真正的
  排行榜冲刺留到截止前约一周的重置之后。
- **One model for all 6 topics? / 6 个主题一个模型?** No — **per-race routing**: Maths uses `cot_v2`
  single-pass; others use the shared few-shot pipeline; News adds RAG. / 不——**按赛道路由**:Maths 用 `cot_v2` 单遍;
  其余用共享的 few-shot 流水线;News 加 RAG。
