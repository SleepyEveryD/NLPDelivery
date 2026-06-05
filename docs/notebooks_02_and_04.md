# How notebooks 02 and 04 work · 笔记本 02 与 04 的工作原理

*Who Wants to Be a PoliMillionaire? — NLP 2025/26 group assignment.*
*《谁想成为 PoliMillionaire？》— NLP 2025/26 小组作业。*

> Bilingual document. **Part A — English** follows first; **第二部分 · 中文** is below it.
> 双语文档。先是 **Part A — English**，下面是 **第二部分 · 中文**。

Both notebooks isolate **one variable — the prompt strategy** — and run it head-to-head on a
fixed model (4-bit Qwen2.5-7B-Instruct) over a fixed question set. Nothing else changes between
arms, so any accuracy difference is attributable to the prompt alone. Notebook 02 asks the
*Phase-2* question ("does chain-of-thought rescue arithmetic?"); notebook 04 generalises it into a
*reasoning-shape routing* study ("does a per-question prompt beat one universal prompt, and when
does reasoning help vs hurt?").

两个笔记本都只隔离**一个变量——提示词策略（prompt strategy）**，在固定模型（4-bit
Qwen2.5-7B-Instruct）和固定题集上做正面对比。各组（arm）之间除提示词外其他一切不变，因此准确率的
差异完全归因于提示词本身。笔记本 02 回答的是 *Phase-2* 的问题（“思维链能否拯救算术？”）；笔记本 04
则把它推广为一项*按推理形态路由（reasoning-shape routing）*的研究（“按题目挑选提示词能否胜过统一
提示词？显式推理何时有益、何时有害？”）。

---

# Part A — English

## Notebook 02 · Prompt engineering (Phase 2)

**File:** `notebooks/02_prompt_engineering.ipynb`

### The question it answers
The notebook-01 baseline scored **87%**, with misses concentrated in **Maths** (arithmetic slips)
and one **Science** knowledge-cutoff trap. The central question:

> Does **chain-of-thought** (think step by step) fix the arithmetic on its own — or is a
> deterministic **calculator tool** (Phase 3) actually required?

It compares three registered strategies on the same dev set: `zero_shot_v1` · `few_shot_v1` · `cot_v1`.

### How it runs (cell by cell)
1. **Setup (cells 3–4).** Clones/pulls the repo into Colab, puts `src/` on the path, and points
   `HF_HOME` at a Google-Drive cache so the ~15 GB Qwen weights download **once** and are reused
   across sessions. Installs the inference stack and checks a GPU is present (T4 required).
2. **Load the model once (cell 6).** Reads `configs/base.yaml` into a `RunConfig`, then builds a
   single 4-bit `TransformersEngine`. It explicitly frees any model left in VRAM from a previous run
   (`del` + `gc.collect()` + `torch.cuda.empty_cache()`) **before** allocating the new one — re-running
   the cell must not stack two models on the T4 (that is the CUDA OOM). The same engine is reused for
   all three strategies. It also loads the dev questions and a `QuestionClassifier`.
3. **Run all three strategies (cell 9).** For each strategy `s`, a fresh config
   (`replace(config, run_id='phase2_'+s, prompt_strategy=s)`), a `QAPipeline` with a
   `PromptBuilder(strategy=s)`, and a `BenchmarkRunner` that writes one logged run per strategy under
   `experiments/runs/`. **Only the `PromptBuilder` changes** — engine, classifier, dev set identical.
4. **Compare (cells 11–13).** Loads the three runs into one DataFrame and prints overall accuracy,
   **Maths-only** accuracy (the key column), a topic × strategy pivot, and mean latency / tokens-out.
   Cell 12 plots three bars (overall, Maths, latency vs the 30 s budget); cell 13 dumps the remaining
   misses with their raw output.

### Core implementation
All strategies live in the `_REGISTRY` of **`src/prompting/builder.py`**; `PromptBuilder(strategy)`
selects one by name. The most important strategy is **`cot_v2`**, forced out by two *real* live
failures:
- **Option-matching mismatch** — the model reasoned correctly (df=17, ±2.110 = option C) but wrote
  `Answer: B`, because B and C shared the same conclusion and it matched only the conclusion.
- **Truncation loss** — correct set-up, but ~5 paragraphs of LaTeX meant it never reached the
  `Answer:` line before the 256-token cap, so the parser blindly guessed.

`cot_v2`'s fix: *solve in ≤3 very short steps, plain numbers only (no LaTeX), when two options share a
conclusion pick the one whose numbers match exactly, and end on a new `Answer: X` line.*

### What it found (23 dev questions, `base.yaml`, 4-bit Qwen2.5-7B)
- **Best overall:** `cot_v1` at **91.3%** (21/23), **+4.3 pts** over zero-shot and few-shot (both
  87.0%). The entire edge comes from one topic: Maths.
- **CoT fixes Maths:** zero-shot 0.50 → CoT **1.00** (4/4); few-shot 0.75. Prompting alone handles the
  dev-set arithmetic, so the Phase-3 calculator is *reliability insurance* for harder/timed sums.
- **Latency cost:** CoT ≈ 4.22 s / 35.9 tokens vs zero-shot 1.13 s / 2 tokens — ~3.7× slower but far
  under the 30 s budget.
- **Prompt sensitivity:** Maths is the big mover (0.50 → 0.75 → 1.00); **News regressed** under CoT
  (1.00 → 0.75 — recall wants a direct answer); Science/Nature flat at 0.75 (a knowledge-cutoff miss
  no prompt fixes); Ancient History / Entertainment / Philosophy saturated at 1.00.

**Takeaway:** CoT is the right default for computation but the wrong default for recall — the tension
notebook 04 turns into a routing problem.

## Notebook 04 · Adaptive prompt routing — a small-LLM reasoning study

**File:** `notebooks/04_adaptive_routing.ipynb`

### The question it answers
For a small model, does routing each question to a prompt chosen for its **reasoning shape** beat
forcing one universal prompt — and **when does explicit reasoning help vs hurt?** The motivating
failure is real: the live Maths run died at level 6 on a clock-chime **interval-counting** question,
because `cot_v2`'s hard "≤3 steps" cap made the model **guess before it had counted**. That same cap
stops verbose stats questions from timing out — so one prompt cannot serve both regimes.

### The four conditions
Over a labelled **8-category** reasoning set, only the prompt changes (no retrieval, no calculator):

| condition | prompt |
|---|---|
| **A_universal** | one universal prompt (production `few_shot_v1`) |
| **B_generic_cot** | always plain "think step by step" |
| **C_structured** | always structured enumeration |
| **D_adaptive** | `ReasoningRouter` picks per question |

### How it runs (cell by cell)
1. **Setup (cells 5–6).** Robust clone/pull (recovers if a prior `rm -rf` deleted the CWD). Pins
   **`bitsandbytes>=0.46.1`** — Colab ships a stale version, and without the upgrade the 4-bit loader
   silently falls back to the *simulated* fixture instead of the real model (a session restart is
   required after install).
2. **Load the labelled set (cell 8).** `load_reasoning_eval('data/reasoning_eval.jsonl')` → 40 MCQs
   across 8 categories, each with a gold answer **and** a gold reasoning-category label (the truth the
   router is scored against). The clock-chime question is `ic-001`.
3. **Pick the engine (cell 10).** `USE_REAL_MODEL=True` loads real Qwen 4-bit (`configs/live.yaml`);
   on failure it falls back to **`SimulatedReasoningEngine`** — a deterministic fixture whose
   correctness comes from a hand-set skill table that *encodes the hypothesis*. The fixture validates
   the **harness** (routing, logging, metrics, figures) but yields **no real findings**.
4. **Run the experiment (cell 12).** `AdaptiveRoutingExperiment(engine, max_new_tokens=512)` runs
   4 conditions × 40 questions and writes one `experiments/adaptive_routing/records.jsonl`. The cap is
   raised 256 → 512 because the game allows 130 s/question, fixing truncations where verbose chains
   never reached `Answer:`.
5. **Analysis (cells 15–26, via `src/experiments/analysis.py`).** §5 comparison table + accuracy bars;
   **§6 category × condition heatmap** (the whole hypothesis in one figure — a prompt green on one row
   is red on another); §7 latency vs accuracy; **§8 the oracle** (best fixed strategy per category — if
   D agrees, the router works); §9 routing accuracy + confusion table; §10 failure taxonomy
   (overthinking, boundary_error, skipped_case, arithmetic_drift, no_answer_parsed, hallucinated).
6. **Focused probes (§12–13).** Two deeper studies on the **logic** questions:
   - **§12 — the level-11 induction death** (`log-ind-001`, qid 6737): pits `cot_v2`, `generic_cot`,
     `checklist_cot`, `checklist_sc5` (checklist + 5-vote self-consistency). Rule: adopt the arm that
     fixes induction (`log-ind-*`) **without** regressing the stats look-alikes (`log-stat-*`).
   - **§13 — directional-logic probe:** adds **`implication_cot`**, which scaffolds direction (write
     `P → Q`, the contrapositive `¬Q → ¬P`, forbid the converse `Q → P` and inverse). Go/no-go: if it
     flips a row others miss, build the full directional dataset; else 6737 is a 7B ceiling and stop.

### Core implementation
**`src/classify/reasoning_router.py`** — `route()` is three lines: classify the shape, look it up in a
policy table, return the prompt.

```python
def route(self, question):
    signal   = self.classifier.classify(question)               # arithmetic / temporal / interval_counting /
    strategy = self.policy.get(signal.category, self.fallback)  # discrete_enum / factual_qa / commonsense /
    return signal, strategy                                     # logical / multi_hop
```

The policy encodes the help-vs-hurt finding:

```python
DEFAULT_ROUTING_POLICY = {
    FACTUAL_QA: "direct_answer",  COMMONSENSE: "direct_answer",   # chain hurts recall → answer directly
    ARITHMETIC: "generic_cot",                                    # compute, don't over-enumerate
    TEMPORAL_REASONING:   "structured_enumeration_cot",           # list events/cases before counting
    INTERVAL_COUNTING:    "structured_enumeration_cot",           #   (the clock-chime fix)
    DISCRETE_ENUMERATION: "structured_enumeration_cot",
    LOGICAL_REASONING: "checklist_cot",  MULTI_HOP: "checklist_cot",  # per-item verification checklist
}
```

`RoutingPromptBuilder` renders the chosen strategy through the same machinery as notebook 02. The
harness and probe condition-sets (`LOGIC_FOCUS_CONDITIONS`, `IMPLICATION_PROBE_CONDITIONS`) live in
`src/experiments/adaptive_routing.py`.

### What it expects to find (full write-up: `experiments/adaptive_routing_findings.md`)
- **Adaptive routing (D) beats every fixed prompt (A/B/C)** — no single prompt is best across all
  categories, and D picks the per-category winner.
- **Structured enumeration cures interval-counting / temporal** but **hurts factual recall** — never
  the universal default. **Direct answering wins** on factual_qa / commonsense; **checklists win** on
  logic / multi_hop. The router's weakest spot is **multi_hop** (overlaps with factual recall).
- **Live-game recommendation (notebook 03):** give the Maths pipeline a per-question router that sends
  interval-counting / temporal / enumeration to `structured_enumeration_cot` and leaves concept/stats
  on the current chain — exactly the split `cot_v2`'s single cap conflated.

## How the two notebooks relate

| | Notebook 02 | Notebook 04 |
|---|---|---|
| **Variable isolated** | prompt strategy | prompt strategy |
| **Question set** | 23-question dev set (topic-labelled) | 40-question set (8 reasoning-shape labels) |
| **Arms** | 3 fixed strategies | 3 fixed + 1 adaptive router |
| **Verdict** | CoT fixes Maths but regresses recall | route by reasoning shape; one prompt can't serve both |
| **Feeds into** | the calculator-vs-prompt decision (Phase 3) | the live Maths routing policy in notebook 03 |

02 discovers the help-vs-hurt tension on real topics; 04 formalises it into a router and validates
that per-question routing recovers the per-category best — the policy that goes live in notebook 03.

## Appendix · What zero-shot / few-shot / CoT are (with the actual prompts)

All three are *registered strategies* in `src/prompting/builder.py`; only the user-turn text differs.
Take this target question as the running example:

```
Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
```

### Zero-shot (`zero_shot_v1`)
**What it is:** ask the model to answer **directly, with no examples and no reasoning** — just the
question, the options, and "give me the letter." It's the cheapest baseline; it leans entirely on
what the model already knows. Builder: `_zero_shot_v1`.

**The prompt it produces:**
```
Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
Reply with ONLY the letter of the correct option (A, B, C, or D). No explanation, no punctuation -- the letter alone.
```

### Few-shot (`few_shot_v1`)
**What it is:** prepend a few **solved examples** ("exemplars") before the real question, so the
model copies the *format* (always end with `Answer: X`) and is primed by the pattern. Still no
explicit reasoning is requested — the examples teach by demonstration. The exemplars are drawn from
**outside** the dev set to avoid leakage. Builder: `_few_shot_v1`; the three exemplars are the
`_FEW_SHOT_EXAMPLES` constant.

**The prompt it produces:**
```
Question: What is the capital of France?
A) Madrid
B) Paris
C) Rome
D) Berlin
Answer: B

Question: Which gas do plants absorb from the air for photosynthesis?
A) Oxygen
B) Hydrogen
C) Carbon dioxide
D) Nitrogen
Answer: C

Question: What is 6 multiplied by 7?
A) 42
B) 36
C) 48
D) 49
Answer: A

Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
Answer with ONLY the letter (A, B, C, or D).
```

### Chain-of-thought (`cot_v1`)
**What it is:** explicitly ask the model to **reason step by step first, then commit** to a final
`Answer: X`. The intermediate steps let it *compute before answering* — which is exactly what
rescues arithmetic. The parser keys on the `Answer:` marker. Builder: `_cot_v1`.

**The prompt it produces:**
```
Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
Think step by step briefly (one or two short sentences). Then on a new line, write your final choice as 'Answer: X', where X is one of A, B, C, or D.
```

**One-line contrast:** zero-shot = *just answer*; few-shot = *here are examples, now answer in the
same format*; CoT = *show your reasoning, then answer*. Notebook 02 finds CoT wins overall (it fixes
Maths) but can hurt pure recall — the exact motivation for the `cot_v2` brevity cap and for the
per-question routing in notebook 04.

### Advanced strategies (everything else in `_REGISTRY`)
Each one below reuses the same `Question:` + `A)…D)` block; only the **instruction** under it changes
(and, where noted, the exemplars prepended above it). For each I give what it is, why it exists, and
the literal directive from `src/prompting/builder.py`.

#### `cot_v2` — CoT + option-matching + a hard brevity cap
**Why:** two real live failures — the model reasoned correctly but picked the option whose *conclusion*
matched (not its numbers), and a 5-paragraph LaTeX chain that never reached `Answer:` before the
256-token cap. So: cap the steps, ban LaTeX, force the numbers to match, always reach `Answer:`.
**Directive (below the options):**
```
Solve in AT MOST 3 very short steps. Plain numbers ONLY -- NO LaTeX, no \frac, no \mu/\sigma, no $...$; write 'mu'/'sigma' as words and keep each step under ~12 words. When two options share the same conclusion, pick the one whose numbers (values, signs, degrees of freedom) match your result EXACTLY -- not just the conclusion. You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- always reach that line.
```

#### `cot_maths_v1` — `cot_v2` + two worked Maths exemplars
**Why:** on a ratio question with **no numbers** the model picked a plausible figure without ever
setting up variables. Two exemplars (one variable-introduction ratio, one plain area) teach the
"let s = …, p = …" move first. The directive is `cot_v2`'s plus a no-numbers clause.
**Exemplars prepended above the question:**
```
Question: Yesterday, a worker took three times as long to finish a task and produced half as many items. The worker's items-per-hour rate today is what percent of yesterday's rate?
A) 150
B) 200
C) 300
D) 600
Let today hours=h, items=i; yesterday hours=3h, items=i/2.
Today rate = i/h. Yesterday rate = (i/2)/(3h) = i/(6h).
Today / Yesterday = (i/h) / (i/(6h)) = 6, so 600%.
Answer: D

Question: If a rectangle has length 15 cm and width 8 cm, what is its area in square centimetres?
A) 46
B) 90
C) 120
D) 160
Area = length x width = 15 x 8 = 120.
Answer: C
```
**Directive (below the real question):**
```
Solve in AT MOST 3 very short steps. If the question has no numbers, introduce variables (e.g. let s = speed, p = price) and write the relationships first. Plain numbers ONLY -- NO LaTeX, no \frac, no \mu/\sigma, no $...$; write 'mu'/'sigma' as words and keep each step under ~12 words. When two options share the same conclusion, pick the one whose numbers (values, signs, degrees of freedom) match your result EXACTLY -- not just the conclusion. You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- always reach that line.
```

#### `few_shot_entertainment` — few-shot tuned for pop-culture recall
**Why:** Entertainment is single-fact recall (film/music/TV/books/games/sport) — the very shape where
explicit reasoning *hurts*. So the three generic exemplars are swapped for domain ones (Jaws,
Thriller, The Office) that prime the register, and **no** reasoning is requested.
**Exemplars prepended:** the `_ENTERTAINMENT_EXAMPLES` (Jaws → B, Thriller → C, The Office → A), each
ending in `Answer: X`. **Directive (below the question):**
```
This is an Entertainment trivia question (film, music, TV, books, video games, or sport). Identify the specific work, person, year, or fact asked for. If reference sources are given above, base the answer on them; otherwise use well-known facts. Answer with ONLY the letter (A, B, C, or D) -- no explanation, the letter alone.
```

### Adaptive-routing research conditions (the four shapes notebook 04 compares)
These are deliberately distinct points on the *"how much explicit reasoning?"* axis. The router in
notebook 04 / 03 picks one per question by reasoning shape.

#### `direct_answer` — minimal reasoning (recall / commonsense)
**Why:** for factual recall and everyday judgement, a chain invites hallucinated justification and
drift. Give one committed answer, no scratch-work.
```
Answer immediately with ONLY the letter (A, B, C, or D). No reasoning, no explanation, no punctuation -- the single letter alone.
```

#### `generic_cot` — the vanilla "think step by step" baseline
**Why:** the universal CoT every paper reaches for first — no brevity cap, no option-matching, no
exemplars. It is the *control* the specialised chains are measured against.
```
Let's think step by step. Work through the reasoning, then on a new line write your final choice as 'Answer: X', where X is one of A, B, C, or D.
```

#### `structured_enumeration_cot` — enumerate-first counting (the clock-chime fix)
**Why:** under `cot_v2`'s ≤3-step cap the model wrote "count the chimes" and then *guessed*. The cure
is the opposite of brevity: list every case in order, boundary-check the endpoints, count only after.
```
Plain text ONLY -- NO LaTeX, no \frac, no \(...\), no \[...\], no $...$; write 'mu'/'sigma' as words and keep EVERY line under ~12 words (LaTeX overruns the token budget before the answer -- a guaranteed loss).
Solve by EXPLICIT ENUMERATION -- do NOT guess a total.
1. List EVERY relevant case/event/item ONE PER LINE, in order (chronological for times, ascending for numbers). Write the value beside each.
2. Boundary check: state the first and last item that qualify, and confirm each endpoint is inside the asked range (watch the off-by-one).
3. ONLY NOW add them up -- show the running total.
You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- ALWAYS reach that line.
```

#### `checklist_cot` — a per-option verification checklist (logic / multi-hop)
**Why:** on "which of the following is true?" the small model commits early to a plausible option and
never tests the others or cross-checks its pick's buried details. A checklist forces it to.
```
Work through this checklist:
1. Restate what is being asked in one line.
2. List any assumptions or hidden constraints.
3. Evaluate EACH option (or EACH reasoning hop) in turn -- mark it true or false and why.
4. Validate: does the surviving option match EVERY detail (numbers, signs, scope), not just the broad conclusion? Re-check any you skipped.
Then on a new line write 'Answer: X' (X = A, B, C, or D).
```

#### `implication_cot` — explicit logical-direction scaffold (induction / contrapositive)
**Why:** on the level-11 induction death (qid 6737) the model thought falsity propagates *forward*; it
failed under `cot_v2`, `generic_cot`, `checklist_cot` and 5-vote self-consistency — a systematic
*directional* misconception. This prompt scaffolds the one thing they all skipped.
```
This is a logical-implication question -- reason about DIRECTION explicitly:
1. Write the rule as an implication 'P -> Q'. For an induction rule write it as 'S(k) true -> S(k+1) true'.
2. Write the VALID contrapositive: 'not Q -> not P'. For induction this means truth propagates FORWARD (k to k+1), so FALSITY propagates BACKWARD: if S(k+1) is false then S(k) is false. A false case forces all SMALLER cases false, NOT larger ones.
3. Do NOT assume the converse 'Q -> P', and do NOT assume the inverse 'not P -> not Q' -- neither is valid.
4. Test EACH option using ONLY the rule and its contrapositive; reject any option that would need the converse or the inverse.
Then on a new line write 'Answer: X' (X = A, B, C, or D). Plain text, no LaTeX.
```

> Note: when retrieval fires, every strategy also prepends a *referenced-knowledge* evidence block
> (`_build_context_block`) above the question, instructing the model to answer from the retrieved
> sources rather than the textbook default. The MCQ block and directives shown above are unchanged.

---

# 第二部分 · 中文

## 笔记本 02 · 提示词工程（Phase 2）

**文件：** `notebooks/02_prompt_engineering.ipynb`

### 它要回答的问题
笔记本 01 的基线得分为 **87%**，失分集中在 **Maths（数学）**（算术失误）以及一道 **Science（科学）**
的知识截止陷阱题。核心问题是：

> **思维链（chain-of-thought，逐步思考）** 自身就能修好算术——还是真的需要一个确定性的
> **计算器工具（Phase 3）**？

它在同一个 dev 集上对比三个已注册策略：`zero_shot_v1` · `few_shot_v1` · `cot_v1`。

### 运行流程（逐单元格）
1. **环境准备（cell 3–4）：** 将仓库 clone/pull 到 Colab，把 `src/` 加入路径，并把 `HF_HOME` 指向
   Google Drive 缓存，使 ~15 GB 的 Qwen 权重**只下载一次**、跨会话复用。安装推理依赖栈并检查 GPU
   （需要 T4）。
2. **只加载一次模型（cell 6）：** 从 `configs/base.yaml` 读入 `RunConfig`，构建单个 4-bit
   `TransformersEngine`。在分配新模型**之前**，显式释放上一次运行残留在显存中的模型
   （`del` + `gc.collect()` + `torch.cuda.empty_cache()`）——重复运行该单元格绝不能在 T4 上叠加两个
   模型（那正是 CUDA 显存溢出 OOM 的来源）。同一个 engine 在三个策略间复用。该单元格还会加载 dev 题目
   和一个 `QuestionClassifier`。
3. **运行三个策略（cell 9）：** 对每个策略 `s`，新建一份配置
   （`replace(config, run_id='phase2_'+s, prompt_strategy=s)`），组装带 `PromptBuilder(strategy=s)`
   的 `QAPipeline`，再由 `BenchmarkRunner` 把每个策略的一次运行记录写到 `experiments/runs/`。
   **唯一变化的是 `PromptBuilder`**——engine、classifier、dev 集完全相同。
4. **对比（cell 11–13）：** 把三次运行读入一个 DataFrame，打印总体准确率、**仅 Maths** 的准确率
   （关键列）、topic × strategy 透视表、以及平均延迟 / 输出 token 数。cell 12 画三组柱状图（总体、
   Maths、延迟 vs 30 秒预算）；cell 13 把剩余的错题连同原始模型输出一并列出。

### 核心实现
所有策略都在 **`src/prompting/builder.py`** 的 `_REGISTRY` 中；`PromptBuilder(strategy)` 按名选取。
最重要的策略是 **`cot_v2`**，它由两次*真实*的实战失败逼出来：
- **选项匹配错配：** 模型推理本身正确（df=17，±2.110 = 选项 C），却写成 `Answer: B`——因为 B 和 C
  结论相同，而它只匹配了结论，没回去核对数字。
- **截断丢失：** 推导正确，但写了约 5 段 LaTeX，在 256-token 上限前从未到达 `Answer:` 行，解析器只能
  瞎猜。

`cot_v2` 的修复：*最多 3 个极短步骤、只用纯数字（禁止 LaTeX）、当两个选项结论相同时选数字完全匹配的
那个、并且必须以新的一行 `Answer: X` 结尾。*

### 结论（23 道 dev 题，`base.yaml`，4-bit Qwen2.5-7B）
- **总体最佳：** `cot_v1` 达 **91.3%**（21/23），比 zero-shot 与 few-shot（均 87.0%）高 **+4.3 个
  百分点**。全部优势都来自一个 topic：Maths。
- **CoT 修好了 Maths：** zero-shot 0.50 → CoT **1.00**（4/4）；few-shot 0.75。仅靠提示词就能搞定
  dev 集的算术，因此 Phase-3 计算器对更难/限时的题目是*可靠性保险*。
- **延迟代价：** CoT 约 4.22 秒 / 35.9 token，对比 zero-shot 1.13 秒 / 2 token——慢约 3.7 倍，但
  远低于 30 秒预算。
- **提示词敏感性：** Maths 是最大变量（0.50 → 0.75 → 1.00）；**News 在 CoT 下反而退化**
  （1.00 → 0.75——记忆类题目要的是直接作答）；Science/Nature 持平在 0.75（一道知识截止错题，任何
  提示词都修不了）；Ancient History / Entertainment / Philosophy 已饱和在 1.00。

**要点：** CoT 是计算类的正确默认值，却是记忆类的错误默认值——这一矛盾正是笔记本 04 转化为路由问题
的对象。

## 笔记本 04 · 自适应提示词路由——小模型推理研究

**文件：** `notebooks/04_adaptive_routing.ipynb`

### 它要回答的问题
对小模型而言，把每道题按其**推理形态**路由到对应提示词，能否胜过统一使用一个提示词——以及**显式推理
何时有益、何时有害？** 触发动机是真实的：实战的 Maths 跑在第 6 关因一道钟声**区间计数
（interval-counting）**题阵亡，因为 `cot_v2` 那条硬性“≤3 步”上限让模型**在数完之前就抢答**。而同一条
上限又能阻止冗长的统计题超时——所以一个提示词无法同时服务两种情形。

### 四个条件（condition）
在一个带标注的 **8 类别**推理题集上，只有提示词变化（无检索、无计算器）：

| 条件 | 提示词 |
|---|---|
| **A_universal** | 统一提示词（生产用 `few_shot_v1`） |
| **B_generic_cot** | 一律“逐步思考” |
| **C_structured** | 一律结构化枚举 |
| **D_adaptive** | 由 `ReasoningRouter` 按题选取 |

### 运行流程（逐单元格）
1. **环境准备（cell 5–6）：** 稳健的 clone/pull（若先前的 `rm -rf` 删掉了当前目录也能恢复）。强制
   钉住 **`bitsandbytes>=0.46.1`**——Colab 自带的版本过旧，不升级的话 4-bit 加载器会**静默回退**到
   *模拟 fixture* 而非真实模型（安装后需重启会话）。
2. **加载标注题集（cell 8）：** `load_reasoning_eval('data/reasoning_eval.jsonl')` → 40 道选择题，
   覆盖 8 个类别，每题都有标准答案**以及**标准推理类别标签（路由器据此评分）。钟声题是 `ic-001`。
3. **选择 engine（cell 10）：** `USE_REAL_MODEL=True` 加载真实 Qwen 4-bit（`configs/live.yaml`）；
   失败则回退到 **`SimulatedReasoningEngine`**——一个确定性 fixture，其正确性来自一张手工设定、
   *编码了假设*的技能表。该 fixture 验证的是**流水线骨架**（路由、日志、指标、图表），并**不产生真实
   结论**。
4. **运行实验（cell 12）：** `AdaptiveRoutingExperiment(engine, max_new_tokens=512)` 跑 4 条件 ×
   40 题，写出一个 `experiments/adaptive_routing/records.jsonl`。上限由 256 提到 512，因为游戏每题允许
   130 秒，正好修掉“冗长推理链从未到达 `Answer:`”的截断。
5. **分析（cell 15–26，经 `src/experiments/analysis.py`）：** §5 对比表 + 各条件准确率柱状图；
   **§6 类别 × 条件热力图**（整个假设浓缩成一张图——某提示词在一行是绿、在另一行就是红）；§7 延迟 vs
   准确率；**§8 oracle**（每类别的最佳固定策略——若 D 与之一致，说明路由器奏效）；§9 路由准确率 +
   混淆表；§10 失败分类学（过度推理、边界错误、漏列情形、算术漂移、未解析出答案、幻觉）。
6. **聚焦探针（§12–13）：** 针对**逻辑题**的两项深入研究：
   - **§12 — 第 11 关归纳法阵亡**（`log-ind-001`，qid 6737）：对比 `cot_v2`、`generic_cot`、
     `checklist_cot`、`checklist_sc5`（清单 + 5 票自洽投票）。规则：采用既能修好归纳题（`log-ind-*`）
     又**不**让统计形近题（`log-stat-*`）退化的那条 arm。
   - **§13 — 方向性逻辑探针：** 加入 **`implication_cot`**，为蕴含方向搭脚手架（写出 `P → Q`、写出
     逆否 `¬Q → ¬P`、禁止逆命题 `Q → P` 与否命题）。Go/no-go：若它翻转了别人答错的某行，就启动完整
     方向性数据集的构建；否则 6737 就是 7B 的能力上限，停手。

### 核心实现
**`src/classify/reasoning_router.py`** —— `route()` 只有三行：先分类推理形态，在策略表里查表，返回
提示词。

```python
def route(self, question):
    signal   = self.classifier.classify(question)               # arithmetic / temporal / interval_counting /
    strategy = self.policy.get(signal.category, self.fallback)  # discrete_enum / factual_qa / commonsense /
    return signal, strategy                                     # logical / multi_hop
```

策略表把“有益/有害”的发现直接编码进去：

```python
DEFAULT_ROUTING_POLICY = {
    FACTUAL_QA: "direct_answer",  COMMONSENSE: "direct_answer",   # 思维链伤害记忆 → 直接作答
    ARITHMETIC: "generic_cot",                                    # 计算，但别过度枚举
    TEMPORAL_REASONING:   "structured_enumeration_cot",           # 计数前先把事件/情形列出来
    INTERVAL_COUNTING:    "structured_enumeration_cot",           #   （钟声题的修复）
    DISCRETE_ENUMERATION: "structured_enumeration_cot",
    LOGICAL_REASONING: "checklist_cot",  MULTI_HOP: "checklist_cot",  # 逐项核验清单
}
```

`RoutingPromptBuilder` 通过与笔记本 02 相同的机制渲染所选策略。实验骨架与探针条件集
（`LOGIC_FOCUS_CONDITIONS`、`IMPLICATION_PROBE_CONDITIONS`）位于
`src/experiments/adaptive_routing.py`。

### 预期结论（完整撰写见 `experiments/adaptive_routing_findings.md`）
- **自适应路由（D）应胜过每一个固定提示词（A/B/C）**——没有任何单一提示词在所有类别都最佳，而 D 会挑
  每个类别的赢家。
- **结构化枚举能治区间计数 / 时序题**，却**伤害事实记忆**——绝不能当统一默认值。**直接作答**在
  factual_qa / commonsense 上获胜；**清单**在 logic / multi_hop 上获胜。路由器最弱处是 **multi_hop**
  （与事实记忆表面重叠）。
- **实战建议（笔记本 03）：** 给 Maths 流水线配一个按题路由器，把区间计数 / 时序 / 枚举送往
  `structured_enumeration_cot`，把概念/统计题留在当前链上——正是 `cot_v2` 那条单一上限混淆掉的分界。

## 两个笔记本的关系

| | 笔记本 02 | 笔记本 04 |
|---|---|---|
| **隔离的变量** | 提示词策略 | 提示词策略 |
| **题集** | 23 题 dev 集（按 topic 标注） | 40 题（按 8 种推理形态标注） |
| **对比组** | 3 个固定策略 | 3 个固定 + 1 个自适应路由 |
| **结论** | CoT 修好 Maths 但让记忆退化 | 按推理形态路由；一个提示词服务不了两种情形 |
| **承接去向** | 计算器 vs 提示词的决策（Phase 3） | 笔记本 03 的实战 Maths 路由策略 |

02 在真实 topic 上发现了“有益/有害”的矛盾；04 把它形式化为一个路由器，并验证按题路由能恢复每类别的
最佳表现——这套策略随后在笔记本 03 上线。

## 附录 · zero-shot / few-shot / CoT 到底是什么（附真实 prompt）

这三者都是 `src/prompting/builder.py` 里*已注册的策略*，区别只在 user 轮的文本。下面统一用这道目标题
作为示例：

```
Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
```

### 零样本 Zero-shot（`zero_shot_v1`）
**是什么：** 让模型**直接作答，不给任何示例、也不要推理**——只给题目、选项，再说“只回字母”。这是
最省的基线，完全依赖模型已有的知识。构建函数：`_zero_shot_v1`。

**它生成的 prompt：**
```
Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
Reply with ONLY the letter of the correct option (A, B, C, or D). No explanation, no punctuation -- the letter alone.
```

### 少样本 Few-shot（`few_shot_v1`）
**是什么：** 在真正的题目前面先放几道**已解答的示例（exemplar）**，让模型照搬其*格式*（一律以
`Answer: X` 结尾）并被这种模式“预热”。同样不要求显式推理——靠示范来教。示例都取自 dev 集**之外**，
以避免数据泄漏。构建函数：`_few_shot_v1`；三个示例就是常量 `_FEW_SHOT_EXAMPLES`。

**它生成的 prompt：**
```
Question: What is the capital of France?
A) Madrid
B) Paris
C) Rome
D) Berlin
Answer: B

Question: Which gas do plants absorb from the air for photosynthesis?
A) Oxygen
B) Hydrogen
C) Carbon dioxide
D) Nitrogen
Answer: C

Question: What is 6 multiplied by 7?
A) 42
B) 36
C) 48
D) 49
Answer: A

Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
Answer with ONLY the letter (A, B, C, or D).
```

### 思维链 Chain-of-thought（`cot_v1`）
**是什么：** 明确要求模型**先逐步推理，再给出**最终的 `Answer: X`。中间步骤让它*先计算再作答*——这
正是拯救算术的关键。解析器以 `Answer:` 标记为锚点。构建函数：`_cot_v1`。

**它生成的 prompt：**
```
Question: What is 8 multiplied by 9?
A) 72
B) 81
C) 64
D) 56
Think step by step briefly (one or two short sentences). Then on a new line, write your final choice as 'Answer: X', where X is one of A, B, C, or D.
```

**一句话对比：** 零样本 = *直接答*；少样本 = *给你示例，按同样格式答*；思维链 = *先把推理写出来，再
答*。笔记本 02 发现 CoT 总体获胜（修好了 Maths），但会伤害纯记忆题——这正是 `cot_v2` 那条简洁上限、
以及笔记本 04 里按题路由的动机所在。

### 高级策略（`_REGISTRY` 里的其余全部）
下面每个都复用同一个 `Question:` + `A)…D)` 题块，只是下方的**指令**变了（标注处还会在题块上方加示例）。
每个我都给出：是什么、为何存在、以及来自 `src/prompting/builder.py` 的原文指令。

#### `cot_v2` —— CoT + 选项匹配 + 硬性简洁上限
**为何：** 两次真实实战失败——模型推理正确，却选了*结论*匹配（而非数字匹配）的选项；以及一条 5 段
LaTeX 的链在 256-token 上限前从未到达 `Answer:`。于是：限制步数、禁用 LaTeX、强制数字匹配、必须到达
`Answer:`。
**指令（在选项下方）：**
```
Solve in AT MOST 3 very short steps. Plain numbers ONLY -- NO LaTeX, no \frac, no \mu/\sigma, no $...$; write 'mu'/'sigma' as words and keep each step under ~12 words. When two options share the same conclusion, pick the one whose numbers (values, signs, degrees of freedom) match your result EXACTLY -- not just the conclusion. You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- always reach that line.
```

#### `cot_maths_v1` —— `cot_v2` + 两道数学解答示例
**为何：** 一道**没有数字**的比例题里，模型没设变量就直接挑了个看似合理的数。两道示例（一道引入变量
的比例题、一道朴素面积题）先教会“let s = …, p = …”这一步。指令是 `cot_v2` 再加一条“无数字”从句。
**题块上方插入的示例：**
```
Question: Yesterday, a worker took three times as long to finish a task and produced half as many items. The worker's items-per-hour rate today is what percent of yesterday's rate?
A) 150
B) 200
C) 300
D) 600
Let today hours=h, items=i; yesterday hours=3h, items=i/2.
Today rate = i/h. Yesterday rate = (i/2)/(3h) = i/(6h).
Today / Yesterday = (i/h) / (i/(6h)) = 6, so 600%.
Answer: D

Question: If a rectangle has length 15 cm and width 8 cm, what is its area in square centimetres?
A) 46
B) 90
C) 120
D) 160
Area = length x width = 15 x 8 = 120.
Answer: C
```
**指令（在真题下方）：**
```
Solve in AT MOST 3 very short steps. If the question has no numbers, introduce variables (e.g. let s = speed, p = price) and write the relationships first. Plain numbers ONLY -- NO LaTeX, no \frac, no \mu/\sigma, no $...$; write 'mu'/'sigma' as words and keep each step under ~12 words. When two options share the same conclusion, pick the one whose numbers (values, signs, degrees of freedom) match your result EXACTLY -- not just the conclusion. You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- always reach that line.
```

#### `few_shot_entertainment` —— 为流行文化记忆调过的少样本
**为何：** 娱乐题是单事实记忆（电影/音乐/电视/书/游戏/体育）——正是显式推理*有害*的形态。于是把三个
通用示例换成领域示例（《大白鲨》《颤栗》《办公室》）来预热语域，并**不**要求推理。
**插入的示例：** 即 `_ENTERTAINMENT_EXAMPLES`（大白鲨 → B、Thriller → C、The Office → A），每个都以
`Answer: X` 结尾。**指令（在题块下方）：**
```
This is an Entertainment trivia question (film, music, TV, books, video games, or sport). Identify the specific work, person, year, or fact asked for. If reference sources are given above, base the answer on them; otherwise use well-known facts. Answer with ONLY the letter (A, B, C, or D) -- no explanation, the letter alone.
```

### 自适应路由的研究条件（笔记本 04 对比的四种形态）
这些是*“需要多少显式推理？”*这条轴上刻意区分开的几个点。笔记本 04 / 03 里的路由器会按推理形态为每题
挑一个。

#### `direct_answer` —— 最少推理（记忆 / 常识）
**为何：** 对事实记忆和日常判断，链式推理反而招来幻觉式论证和漂移。给一个确定答案，不留草稿空间。
```
Answer immediately with ONLY the letter (A, B, C, or D). No reasoning, no explanation, no punctuation -- the single letter alone.
```

#### `generic_cot` —— 朴素的“逐步思考”基线
**为何：** 每篇论文最先用的通用 CoT——没有简洁上限、没有选项匹配、没有示例。它是衡量那些专门链的*对照
组*。
```
Let's think step by step. Work through the reasoning, then on a new line write your final choice as 'Answer: X', where X is one of A, B, C, or D.
```

#### `structured_enumeration_cot` —— 先枚举再计数（钟声题的修复）
**为何：** 在 `cot_v2` 的 ≤3 步上限下，模型写了“数一下钟声”然后*就猜*。解法与简洁相反：按序列出每个
情形、对端点做边界检查、列完之后才计数。
```
Plain text ONLY -- NO LaTeX, no \frac, no \(...\), no \[...\], no $...$; write 'mu'/'sigma' as words and keep EVERY line under ~12 words (LaTeX overruns the token budget before the answer -- a guaranteed loss).
Solve by EXPLICIT ENUMERATION -- do NOT guess a total.
1. List EVERY relevant case/event/item ONE PER LINE, in order (chronological for times, ascending for numbers). Write the value beside each.
2. Boundary check: state the first and last item that qualify, and confirm each endpoint is inside the asked range (watch the off-by-one).
3. ONLY NOW add them up -- show the running total.
You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- ALWAYS reach that line.
```

#### `checklist_cot` —— 逐项核验清单（逻辑 / 多跳）
**为何：** 在“以下哪项为真？”里，小模型会过早锁定一个看似合理的选项，从不检验其他项、也不复核所选项里
埋着的细节。清单强迫它做这些。
```
Work through this checklist:
1. Restate what is being asked in one line.
2. List any assumptions or hidden constraints.
3. Evaluate EACH option (or EACH reasoning hop) in turn -- mark it true or false and why.
4. Validate: does the surviving option match EVERY detail (numbers, signs, scope), not just the broad conclusion? Re-check any you skipped.
Then on a new line write 'Answer: X' (X = A, B, C, or D).
```

#### `implication_cot` —— 显式逻辑方向脚手架（归纳 / 逆否）
**为何：** 在第 11 关归纳法阵亡（qid 6737）里，模型以为“假”是*向前*传播的；它在 `cot_v2`、
`generic_cot`、`checklist_cot` 乃至 5 票自洽下都失败——这是一种系统性的*方向性*误解。这个 prompt 专门
搭起它们都跳过的那一步。
```
This is a logical-implication question -- reason about DIRECTION explicitly:
1. Write the rule as an implication 'P -> Q'. For an induction rule write it as 'S(k) true -> S(k+1) true'.
2. Write the VALID contrapositive: 'not Q -> not P'. For induction this means truth propagates FORWARD (k to k+1), so FALSITY propagates BACKWARD: if S(k+1) is false then S(k) is false. A false case forces all SMALLER cases false, NOT larger ones.
3. Do NOT assume the converse 'Q -> P', and do NOT assume the inverse 'not P -> not Q' -- neither is valid.
4. Test EACH option using ONLY the rule and its contrapositive; reject any option that would need the converse or the inverse.
Then on a new line write 'Answer: X' (X = A, B, C, or D). Plain text, no LaTeX.
```

> 注：当检索触发时，每个策略还会在题块上方插入一个*引用知识*证据块（`_build_context_block`），指示模型
> 根据检索到的来源作答、而非教科书默认答案。上面展示的 MCQ 题块与指令本身不变。
