# 5-min video script — notebook tour, focus on **Notebook 03 (Live Play)**

**Constraints (PDF):** ≤5 minutes · screen-capture of the *notebook* (no slides) · both members speak ·
**must name the RAG APIs out loud.** Presenters: **J = Jiaxin Yang**, **R = Runjie (Simone) Dai**.

> 用法:每段「*屏幕*」是录屏动作提示;「»」是照念的**英文旁白**,下面「中」是对应的**中文翻译**(只为对照,录的时候念英文)。
> 主录 **`notebooks/03_live_play.ipynb`**(输出已存好,**不要现场重跑**);开头花 ~40 秒在 `notebooks/` 文件列表上做"套件总览"。
> §5 的 live 排行榜数字若与实际最新一跑不一致,**以屏幕为准**,口播随之改。

---

## 0:00 – 0:30 · 封面:这是什么 — **J**
*屏幕:`notebooks/` 文件夹列表(01→05)+ 03 顶部标题*

» Hi, we're **Jiaxin** and **Runjie**. Our project is an agent that plays the quiz *Who Wants to Be a
PoliMillionaire?* — **Qwen2.5-7B-Instruct in 4-bit**, running **locally** on a Colab T4, **no external LLM
APIs**, **30 seconds per question**. We built it as a series of notebooks; today we walk the suite, then go
deep on **notebook 3 — the live game**.

**中** 大家好,我们是 **Jiaxin** 和 **Runjie**。我们的项目是一个会玩《谁想成为 PoliMillionaire?》答题游戏的智能体——用 **4-bit 的
Qwen2.5-7B-Instruct**,在 Colab T4 上**本地运行**,**不调用任何外部大模型 API**,**每题限时 30 秒**。我们把它做成了一系列
notebook;今天先快速过一遍整套,再重点深入 **notebook 3——实战游戏**。

## 0:30 – 1:10 · 套件总览 — **J → R**
*屏幕:依次划过 `notebooks/` 里的 01,02,04,05*

» **(J)** The suite reads top to bottom. **Notebook 1** is the **offline baseline** on our own dev set — about
**87 percent**. **Notebook 2** benchmarks three prompt strategies, where chain-of-thought lifts **Maths from
50 to 100 percent**.
» **(R)** **Notebook 4** is our reasoning study — a deterministic solver plus adaptive routing for Maths.
**Notebook 5** stress-tests live **News** with retrieval. And **notebook 3** — our focus — is the **real
test**: the live game server.

**中** **(J)** 整套从上往下读。**Notebook 1** 是在我们自己开发集上的**离线基线**——约 **87%**。**Notebook 2** 对比三种 prompt
策略,其中思维链把 **Maths 从 50% 提到 100%**。
**(R)** **Notebook 4** 是我们的推理研究——确定性求解器加上 Maths 的自适应路由。**Notebook 5** 用检索压力测试实时 **News**。
而我们重点讲的 **notebook 3**,就是**真正的测试**:实时游戏服务器。

## 1:10 – 1:45 · 为什么 03 是「真正的测试」 — **R**
*屏幕:03 cell [1] 的 markdown(OFFLINE vs LIVE 对比)*

» **(R)** Notebook 1 grades **itself** on a dev set where we already know the answers. Notebook 3 is
different: **the game server feeds the questions and grades them**, and we learn "correct" **only after we
submit**. Same pipeline, same JSONL logging — the **only switch** is `config.mode = 'live'` plus a logged-in
game client. **One line** flips offline to live.

**中** **(R)** Notebook 1 是在一个我们**已知答案**的开发集上**给自己打分**。Notebook 3 不一样:**题目由游戏服务器下发并判分**,我们
**只有提交之后**才知道对错。同样的流水线、同样的 JSONL 日志——**唯一的开关**就是 `config.mode = 'live'` 加上一个已登录的游戏客户端。
**一行**就从离线切到实时。

## 1:45 – 2:20 · Setup + 安全凭据 + 预热 — **J**
*屏幕:03 §1 的 clone / install / playwright cell,再到 §4 登录 cell、§3 模型预热 cell*

» **(J)** Setup **clones the repo's `main` branch**, installs the inference stack, and — important for News —
installs a **headless Chromium** browser. We log into the game with a **PoliMi email**; the password comes
from a **Colab secret**, **never hard-coded**. The model is **loaded and warmed up before any timed
question**, so the 30-second clock never pays the cold-start cost.

**中** **(J)** Setup 会**克隆仓库的 `main` 分支**,安装推理栈,并且——对 News 很关键——装一个**无头 Chromium** 浏览器。我们用
**PoliMi 邮箱**登录游戏;密码来自 **Colab secret**,**绝不写死在代码里**。模型在**任何计时题之前就加载并预热**,所以那 30 秒
计时永远不用承担冷启动开销。

## 2:20 – 2:35 · ▶ 扫场开始 — 两条流水线 — **R**
*屏幕:03 §5 的 scoreboard 表(逐行往下指)*

» **(R)** Now we **sweep all six competitions**, one live game each — and a game **ends the moment we miss**,
so what the scoreboard shows is **how far each race climbed** before the first slip. **Overall, 26 of 32 right.**
**Two pipelines** drive it: a **shared few-shot** for the knowledge topics, and a **dedicated chain-of-thought**
for Maths. Let me take the competitions **one at a time** — strategy, result, and **why**.

**中** **(R)** 现在我们**连扫全部 6 个赛道**,每个赛道打一局实时游戏——而**一旦答错一题游戏就结束**,所以成绩单显示的是**每个赛道在第一次
失手前爬了多远**。**总体 32 题对 26 题。**驱动它的是**两条流水线**:知识类赛道共用一条 **few-shot**,Maths 用**专属的思维链**。下面
我**一个赛道一个赛道**讲——策略、结果、**为什么**。

## 2:35 – 2:48 · ① Entertainment — **J**
*屏幕:scoreboard 第 0 行 + 该题错题卡*

» **(J)** **Entertainment** runs the **shared `few_shot_v1`** — three solved examples, answer with a letter —
plus **Wikipedia retrieval**. On the **leaderboard it's maxed at level 15**; in this sweep the very first
question — a Mariah Carey album — slipped. **Why few-shot:** it's pure recall, **no reasoning needed**, so the
prompt just **locks the answer format and stays fast**. The miss was a **retrieval recall slip** — it pulled
the wrong album page — **not** a strategy error.

**中** **(J)** **Entertainment** 跑**共享的 `few_shot_v1`**——三个示例、只回字母——再加 **Wikipedia 检索**。**排行榜已打满到第
15 关**;这局第一题(一道 Mariah Carey 专辑题)就失手了。**为什么用 few-shot:**这是纯记忆题,**不需要推理**,prompt 只负责**锁定答案
格式、保持快**。失手是**检索召回偏了**(取错了专辑页),**不是**策略问题。

## 2:48 – 3:01 · ② Ancient History & Politics — **R**
*屏幕:scoreboard 第 1 行 + 该题错题卡*

» **(R)** **History** uses the **same `few_shot_v1` + Wikipedia**. Also **maxed at level 15**; this game missed a
fine point of Roman law — a freed slave's legal status. **Why the same pipeline:** it's another **knowledge
topic**, so it shares the recall path. The slip came because **Wikipedia returned broad pages** — "Roman
Empire" — **too coarse** for that fine legal distinction.

**中** **(R)** **History** 用**同一条 `few_shot_v1` + Wikipedia**。同样**打满到第 15 关**;这局错在一个罗马法细节——被释奴的法律
地位。**为什么同一条流水线:**它同样是**知识类**,共用记忆路径。失手是因为 **Wikipedia 返回的是宽泛页面**(「罗马帝国」),对那种**细粒度
法理区分太粗**。

## 3:01 – 3:13 · ③ Science & Nature — **J**
*屏幕:scoreboard 第 2 行(retrieval_fired=0)+ 该题错题卡*

» **(J)** **Science** runs `few_shot_v1`, but here **retrieval is gated OFF** — the classifier judged the model
already knew enough. Result: **5 of 6 this game**, level 15. **Why skip retrieval:** science concepts live in
the **model's own training**, so retrieving would only **add noise and latency**. Five-of-six **confirms that
call**; the one miss was a physics detail — near-field sound intensity.

**中** **(J)** **Science** 跑 `few_shot_v1`,但这里**检索被闸门关掉**——分类器判断模型自身知识已经够。结果:这局 **6 题对 5 题**,第 15
关。**为什么不检索:**科学概念大多在**模型的预训练里**,贸然检索只会**引噪声、加延迟**。6 对 5 **印证了这个判断**;唯一的错是个物理细节——
近场声强。

## 3:13 – 3:35 · ④ Maths — **R**(唯一独立流水线)
*屏幕:scoreboard 第 3 行 + per-question 表上 `strat=cot_v2`*

» **(R)** **Maths is the exception** — its **own chain-of-thought `cot_v2`**: reason step by step, but **capped
at three steps, no LaTeX**, with an **option-matching check**, **single-pass**, **no retrieval, no calculator**.
On the leaderboard, **level 11** — one of our **two hardest**; this game missed a 2-by-2 / 3-by-3 determinant.
**Why a whole separate pipeline:** Maths is the **only topic that needs real reasoning** — that's the **50-to-100
jump** from notebook 2. But reasoning **blows the 30-second wall**, so we **cap the steps and ban LaTeX** so it
**always reaches the answer**; self-consistency was **dropped after a 41-second timeout**.

**中** **(R)** **Maths 是例外**——它有**自己的思维链 `cot_v2`**:分步推理,但**最多 3 步、禁 LaTeX**,带**选项匹配校验**、**单遍**、
**不检索、不用计算器**。排行榜上**第 11 关**——是我们**两个最难赛道**之一;这局错在一道 2×2 / 3×3 行列式题。**为什么单开一条流水线:**Maths
是**唯一需要真推理**的赛道——这正是 notebook 2 里**从 50% 到 100%**的来源。但推理会**撞 30 秒墙**,所以我们**砍步数、禁 LaTeX**,让它
**总能跑到答案**;self-consistency 在**一次 41 秒超时后被砍掉**。

## 3:35 – 3:48 · ⑤ Philosophy & Psychology — **J**
*屏幕:scoreboard 第 4 行(retrieval_fired=4/10)+ 该题错题卡*

» **(J)** **Philosophy** is back on `few_shot_v1` + RAG, which **fired on 4 of 10** questions. Result: **9 of 10
— our best knowledge run**, level 15. **Why selective retrieval:** it's knowledge **plus concepts**, so the gate
**fires only when a question needs it**. The single miss was a **"NOT" question** on brain regions — a hard
edge case.

**中** **(J)** **Philosophy** 回到 `few_shot_v1` + RAG,检索**在 10 题里触发了 4 题**。结果:**10 题对 9 题——知识类里我们最好的一
局**,第 15 关。**为什么按需检索:**它是知识**加概念**,所以闸门**只在题目需要时才触发**。唯一的错是一道**「NOT」反向题**(脑区),属于很难
的边角。

## 3:48 – 4:25 · ⑥ News — live RAG + 点名 API — **R**
*屏幕:scoreboard 第 5 行(retrieval_fired=13/13)+ RAG/Tool usage 表*

» **(R)** **News is where live RAG earns its place** — `few_shot_v1` plus **routed live retrieval**, which fired
on **all 13 questions**. Let me **name the sources out loud**, as the rules require: **Google News RSS** for
headlines, the **Guardian Content API** for raw article bodies, and a **headless Chromium browser through
Playwright** to open other articles **past the consent wall** — with **Wikipedia** and a local **FAISS index**
as fallbacks. Result: **12 of 13**, level 12 — the **other hardest topic**. **Why go live:** News is **past the
model's knowledge cutoff**, and the **numbers and quotes live in the article body, not the headline** — so we
read the **raw, non-generated** body, exactly as allowed. The one miss was an **exact percentage buried in the
text**.

**中** **(R)** **News 正是实时 RAG 体现价值的地方**——`few_shot_v1` 加上**路由式实时检索**,在**全部 13 题**上都触发了。按规则我要
**把数据源念出来**:用 **Google News RSS** 取标题,用 **Guardian Content API** 取文章正文原文,再用 **Playwright 驱动的无头 Chromium
浏览器**打开其他文章、**越过同意弹窗**——并以 **Wikipedia** 和本地 **FAISS 索引**作为兜底。结果:**13 题对 12 题**,第 12 关——**另一个
最难赛道**。**为什么要联网:**News **超出模型知识截止**,而**具体数字和引用藏在正文里、不在标题**——所以我们读**原始、未经生成**的正文,完全
符合规则。唯一的错是**正文里一个精确百分比**读偏了。

## 4:25 – 4:45 · 排行榜 + 可审计的错题 — **J**
*屏幕:03 scoreboard 的 `lb_level` 列 + 错题清单*

» **(J)** Step back to the **leaderboard**: **Entertainment, History, Science and Philosophy max out at level
15**; **News reaches 12** and **Maths 11** — **exactly the two topics our analysis called hardest**. And the
notebook dumps **every wrong question** with the option we picked, so **each miss is auditable** — you saw, none
of the six was a routing error.

**中** **(J)** 退回到**排行榜**:**Entertainment、History、Science、Philosophy 都打满到第 15 关**;**News 到第 12 关**、**Maths
到第 11 关**——**正好是我们分析判定的两个最难赛道**。而 notebook 会把**每一道错题**连同我们选的选项一并导出,所以**每个失误都可追溯审计**——
你也看到了,六道错题**没有一道是分流选错**。

## 4:50 – 5:00 · 收尾 + AI 声明 — **R + J**
*屏幕:滚回 03 顶部封面*

» **(R)** To be transparent: we used a **coding assistant** to help write and refactor code, but the
**design, the experiments, and the analysis are ours**.
» **(J)** That's our PoliMillionaire agent — local models, per-race strategies, agentic tools, and live RAG —
and **notebook 3 is where it meets the real game**. Thanks for watching!

**中** **(R)** 坦诚说明:我们用了**编程助手**帮忙写和重构代码,但**设计、实验和分析都是我们自己的**。
**(J)** 这就是我们的 PoliMillionaire 智能体——本地模型、按赛道策略、agentic 工具,以及实时 RAG——而 **notebook 3 正是它与真实
游戏正面交锋的地方**。谢谢观看!

---

### 录制小贴士
- 录的时候**念英文**(「»」那几行),下面的「中」只是给你对照、确认意思用的。
- 两人对半开口(J:封面、套件总览、Setup、排行榜、收尾;R:为什么真测试、扫全赛道点名 API、分流错题、AI 声明)。
- **硬性规则**:2:20–4:00 那段**必须把 RAG 的 API 念出来**(Google News RSS · Guardian Content API · Playwright/headless Chromium · Wikipedia · FAISS)——超时也不能删这段。
- 输出**已存好**:逐段往下滚 `03_live_play.ipynb`,不要现场等模型;§5 用最近一次跑好的结果。
- 卡 5:00;超时先砍 4:00–4:30 的"错题可审计"细节,保 API 那段。
- 录前核对屏幕上的实际数字(overall 26/32 · News 12/13 · Philosophy 9/10 · 排行榜 level 15/12/11),口播与屏幕一致。
