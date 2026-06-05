# Per-Competition Strategy — 2-minute script / 2 分钟讲稿

> Spoken script, ~300 words ≈ 2 min. One core idea: **we route by competition because each one fails
> differently.** 中文对照见每段末尾。

---

**(0:00 – 0:20) The frame**

Every question runs through one pipeline — a local Qwen2.5-7B, 4-bit, greedy, no external APIs, 30 seconds
a turn. The only thing that changes per competition is three knobs: **retrieve or not, which prompt, and
whether a solver helps.** We tune them per competition because the *failures* are different.

中:所有题走同一条流水线——本地 Qwen2.5-7B、4-bit、贪心、不调外部 API、每题 30 秒。赛道之间只改三个旋钮:**是否检索、哪种 prompt、是否用求解器**——因为每个赛道的失败模式不同。

**(0:20 – 0:50) Reasoning failures → Maths**

**Maths** fails on *reasoning*, not knowledge. Chain-of-thought is the only place it helps — it took Maths
from 50 to 100 percent offline. But no single prompt wins: counting questions need structured enumeration,
arithmetic prefers plain step-by-step. So Maths uses **adaptive routing** plus a safe abstaining solver.
That fixed a clock-counting death and took us to level 11.

中:**Maths** 败在*推理*。思维链只在这里有用(离线 50%→100%),但单一 prompt 不够:计数题要结构化枚举、算术要逐步——所以 Maths 用**自适应路由**加安全弃权求解器,修好了计数题、冲到第 11 关。

**(0:50 – 1:25) Knowledge failures → News, Science, History**

**News, Science, and History** fail on *missing knowledge*, so we **force retrieval** — but routed
differently. News is post-cutoff, so it goes to the **live web**: Google News, the Guardian API, a real
browser for raw article text. It scored 92 percent. Science and History go to **local Wikipedia indexes**,
which is safe and doesn't hit rate limits. We do **not** force CoT here — on News it actually made the model
over-reason correct dates into wrong answers.

中:**News、Science、History** 败在*知识缺失*,所以**强制检索**,但路由不同:News 是截止日之后,走**实时网络**(Google News、Guardian、真实浏览器读原文),拿到 92%;Science/History 走**本地 Wikipedia 索引**,安全且不触发限流。这里**不**强制 CoT——News 上 CoT 反而把正确日期推理成错答。

**(1:25 – 2:00) Prior wins → Entertainment, Philosophy**

**Entertainment and Philosophy** the model already knows, so we leave the baseline alone and retrieve only
on demand. We even *tried* forcing retrieval on Entertainment — it hammered Wikipedia, 40 percent of turns
died to rate-limiting, so we reverted it. Both already max out at level 15. Bottom line: every choice
answers a specific logged failure, and the conservative ones were validated by offline A/Bs that prevented
bad deploys.

中:**Entertainment、Philosophy** 模型本就会,所以保持基线、只按需检索。Entertainment 我们*试过*强制检索——把 Wikipedia 打爆、40% 的题死于限流,于是回退;两者都已打满第 15 关。一句话:每个选择都对应一个具体日志失败,保守的那些都经离线 A/B 验证,避免了糟糕的上线。

---

## Appendix — Exact prompt per competition / 附录:每个赛道实际用的 prompt

> Wired in `notebooks/03_live_play.ipynb` (routed by `competition_id`, the reliable live signal).
> Prompt builders live in `src/prompting/builder.py`. 路由由 `competition_id` 决定,prompt 定义在 `src/prompting/builder.py`。

| # | Competition 赛道 | Prompt strategy 用的 prompt | Notes 说明 |
|---|---|---|---|
| 0 | **Entertainment 娱乐** | **`few_shot_entertainment`** | few-shot with **film/music/TV exemplars** (Jaws / Thriller / The Office) + a domain-aware instruction, **NO CoT** — single committed letter. 用影视音乐电视范例 + 领域指令,不推理,只给一个字母。 |
| 1 | **History 历史** (Ancient History and Politics) | **`few_shot_v1`** | shared few-shot (3 generic exemplars: capital-of / photosynthesis / 6×7) + forced RAG. 共享 few-shot + 强制检索。 |
| 2 | **Science 科学** (Science and Nature) | **`few_shot_v1`** | same shared few-shot + forced RAG. 同上 + 强制检索。 |
| 3 | **Maths 数学** | **adaptive — `RoutingPromptBuilder(MATHS_LIVE_POLICY)`** | per question: counting / temporal / enumeration → **`structured_enumeration_cot`**; everything else (arithmetic, logic, stats/concept) → **`cot_v2`** (fallback). No retrieval, no calculator, ~300-token cap. 按题:计数/时间/枚举→`structured_enumeration_cot`,其余→`cot_v2`;无检索、无计算器。 |
| 4 | **Philosophy 哲学** (Philosophy and Psychology) | **`few_shot_v1`** | shared few-shot, on-demand RAG. 共享 few-shot,按需检索。 |
| 5 | **News 新闻** | **`few_shot_v1`** | shared few-shot + forced RAG to live web. **Deliberately NOT CoT** — CoT regressed News. 共享 few-shot + 强制实时网络检索;**刻意不用 CoT**(CoT 会让 News 变差)。 |

**The two prompts that matter most / 两个最关键的 prompt:**

- **`cot_v2`** (Maths default fallback): "Solve in AT MOST 3 very short steps. **No LaTeX.** When two
  options share a conclusion, pick the one whose **numbers match exactly**. You MUST end with `Answer: X`."
  — born from two real logged failures (a t-test that wrote the right reasoning but the wrong letter, and
  one that ran out of tokens writing LaTeX before the answer line).
  **`cot_v2`**(Maths 默认兜底):"最多 3 步、**禁用 LaTeX**、两选项结论相同时选**数字完全匹配**的、必须以 `Answer: X` 结尾"
  ——由两个真实日志失败逼出来的(推理对但写错字母;写 LaTeX 写到超 token 没写出答案)。

- **`structured_enumeration_cot`** (Maths counting): "Solve by **EXPLICIT ENUMERATION** — list EVERY
  case one per line in order, **boundary-check** the endpoints, count **only after** listing." — the
  clock-chime fix that took Maths from level 9 → 11.
  **`structured_enumeration_cot`**(Maths 计数题):"**显式枚举**——每个情形一行列出、**检查端点**、**列完再数**"
  ——修好钟声题、把 Maths 从第 9 关推到第 11 关。

**Summary / 小结:** 4 of 6 competitions share **`few_shot_v1`** (knowledge races — the baseline already
wins, so we don't touch it); Entertainment swaps in domain exemplars (**`few_shot_entertainment`**); only
Maths uses adaptive CoT routing. We add prompt machinery **only where a logged failure proved it's needed**.

**小结:** 6 个赛道里有 4 个共用 **`few_shot_v1`**(知识赛道,基线已赢就不动);Entertainment 换成领域范例
(**`few_shot_entertainment`**);只有 Maths 用自适应 CoT 路由。**只在日志证明确有需要的地方**才加 prompt 复杂度。
