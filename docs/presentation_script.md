# Live presentation script — PoliMillionaire (for the professor defense)

**Note:** this is a *spoken* script for the live talk (not the recorded ≤5-min video — that's
`video_script_nb03.md`). ~5–6 min of talking, leaves room for interruptions. **J = Jiaxin, R = Runjie.**
Screen: drive `notebooks/03_live_play.ipynb` (outputs already saved). 「»」= say in English; 「中」= 中文对照.

---

### 1 · Hook + the problem (J) — ~40 s
*Screen: notebook 03 title cell*

» Good morning. Our system plays the quiz *Who Wants to Be a PoliMillionaire?* under three hard rules: the
model runs **locally** — **no external LLM APIs**, **30 seconds per question**, and any retrieved content must
be **raw, not generated**. So the interesting problem isn't "use a big model" — it's **how far can we push a
small 7-billion model with good engineering**. That framing drives every notebook.

**中** 早上好。我们的系统在三条硬性规则下玩这个答题游戏:模型**本地运行**、**不调外部大模型 API**、**每题 30 秒**,且检索内容必须是
**原始、未经生成**的。所以真正有意思的问题不是"换个大模型",而是**好的工程能把一个 70 亿的小模型推到多远**。

### 2 · Architecture in one breath (R) — ~45 s
*Screen: scroll to §4 pipeline-wiring cell*

» Everything goes through **one orchestrator**, `QAPipeline`, with seven timed stages: classify the topic, an
optional **deterministic solver**, optional **retrieval**, build the **prompt**, **generate** locally,
optional **calculator tool**, then **parse** the answer. It's crash-safe — we always submit something before
the wall. Offline and live use the *same* pipeline; the only switch is `mode='live'` plus a game client.

**中** 一切都走**同一个编排器** `QAPipeline`,七个计时阶段:分类、可选的确定性求解器、可选检索、构建 prompt、本地生成、可选计算器、
解析答案。全程崩溃安全,到点前一定提交。离线和实时**共用同一条流水线**,唯一开关是 `mode='live'` 加一个游戏客户端。

### 3 · The three findings that shaped it (J→R) — ~80 s
*Screen: §2/§3 result charts (notebook 2 outputs) if shown, else describe*

» **(J)** Three findings from the offline studies. **One:** chain-of-thought only helps **Maths** — it lifted
Maths from **50% to 100%** while knowledge topics were already saturated. So we route by topic, not one prompt
for all.
» **(R)** **Two:** a *negative* result we're proud of — **self-consistency voting timed out** at the 30-second
wall and its sampled chains made the **same** mistake, so we ship a single good chain-of-thought instead.
**Three:** the model is **overconfident** — confidently wrong on post-cutoff facts — which motivates
retrieval.

**中** **(J)** 离线研究的三个发现。**一:**思维链只对 **Maths** 有用——把 Maths 从 **50% 提到 100%**,知识类题目本就饱和,所以我们
**按赛道分流**,而不是一种 prompt 通吃。**(R)** **二:**一个我们很看重的**负面结果**——**自洽投票在 30 秒墙前超时**,而且采样出的
多条链犯**同一个错**,所以我们改用单条高质量思维链。**三:**模型**过度自信**——对训练截止后的事实自信地答错——这正是引入检索的动机。

### 4 · Live RAG — name the APIs (R) — ~50 s
*Screen: §1 playwright install cell + §8 RAG-usage table*

» For **News** — questions about events after the model's cutoff — we retrieve raw content from, by name:
**Google News RSS** for headlines, the **Guardian Content API** for article bodies, and a **headless Chromium
browser through Playwright** to open other articles past the consent wall, with **Wikipedia** and a local
**FAISS index** as fallbacks. Nothing is summarised by a second model — it's verbatim text in the prompt.
Live, News scored **92%** and retrieval fired on **all 13** News questions.

**中** 对 **News**(模型截止日之后的事件),我们检索原始内容,点名:用 **Google News RSS** 取标题、**Guardian Content API**
取正文、**Playwright 驱动的无头 Chromium** 越过同意弹窗打开其他文章,并以 **Wikipedia** 和本地 **FAISS 索引**兜底。没有第二个
模型做摘要——都是原文进 prompt。实时跑 News **92%**,检索在**全部 13 题**上都触发了。

### 5 · ▶ Live demo — notebook 3 (J) — ~60 s
*Screen: §5 run_session output (one game), then §8 scoreboard*

» This is the real test — the game **server** feeds and grades; we only learn "correct" **after** we submit.
Here's one Entertainment game: each turn prints the question, our pick, and the server's verdict — **3 of 4
correct, level 3**, every turn under 7 seconds. Then we sweep all six competitions: **overall 26 of 32, 81
percent**; on the leaderboard, four topics max out at **level 15**, News reaches **12** and Maths **11** —
the two hardest, exactly where our analysis predicted.

**中** 这是真正的测试——题目由**服务器**下发并判分,我们**提交后**才知道对错。这是一局 Entertainment:每回合打印题目、我们的选择和
服务器结果——**4 题对 3 题、到第 3 关**,每回合 7 秒以内。然后连扫 6 个赛道:**总体 32 题对 26 题,81%**;排行榜上四个赛道打满到
**第 15 关**,News 到 **12**、Maths 到 **11**——最难的两个,正好和我们预测一致。

### 6 · Close + integrity (R+J) — ~30 s
*Screen: scroll back to title*

» **(R)** To sum up: a small local model, made strong by **per-topic routing, a safe deterministic solver,
live RAG, and honest agentic experiments** — including the ones we *rejected*. We used a coding assistant for
code; the **design, experiments, and analysis are ours**.
» **(J)** We're happy to dig into any part — the solver gate, the routing study, or the retrieval cascade.
Thank you.

**中** **(R)** 总结:一个本地小模型,靠**按赛道分流、安全确定性求解器、实时 RAG,以及诚实的 agentic 实验**(包括被我们**否决**的方案)
变强。代码用了编程助手,但**设计、实验、分析都是我们自己的**。**(J)** 任何部分都欢迎深入——求解器门控、路由研究或检索级联。谢谢。

---

### Delivery tips
- Lead with the **constraints** — professors grade the *thinking under constraints*, not raw accuracy.
- Volunteer the **two negative results** (self-consistency timeout, CoT hurting News). Owning limitations
  reads as rigour and pre-empts the obvious questions.
- Keep `presentation_qa.md` open in another tab; the demo naturally sets up sections D–H of it.
- If interrupted, stop and answer — don't race the script. The 6 sections are modular; you can resume anywhere.
