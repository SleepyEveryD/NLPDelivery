"""Prompt construction. The lever of prompt engineering, this module is.

Zero-shot, few-shot, chain-of-thought -- templates we swap, and benchmark each we do.
Per question-type and difficulty, specialise the prompt we may (an open question the rubric asks).
"""
from __future__ import annotations

from schemas import Question, QuestionType, RetrievedDoc


# --- Strategy implementations ---

def _build_context_block(context: list[RetrievedDoc]) -> str:
    """A list of retrieved docs -> a formatted evidence block, this turns.

    Raw text only, injected it is -- never an LLM-generated answer (D-008).
    """
    # Each doc's raw text, on its own numbered line it goes.
    lines = ["Referenced knowledge (retrieved sources -- your PRIMARY evidence):"]
    for i, doc in enumerate(context, start=1):
        lines.append(f"[{i}] {doc.text.strip()}")
    # GROUNDING directive -- the "prior-default" miss this targets (qid 10630: the model picked the
    # textbook cause "earthquake" over the article's specific one). When the sources state the answer
    # -- a name, place, cause, number, quote -- ANSWER FROM THEM, not from the most typical/general case.
    lines.append(
        "When these sources state the answer (a name, place, cause, number, or quote), base your answer "
        "on what they say -- NOT on the most common or textbook case. The article's specific detail wins. "
        "Only if the sources do not address the question, fall back on general knowledge."
    )
    lines.append("")  # A blank line, separation it provides.
    return "\n".join(lines)


def _zero_shot_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """The zero-shot baseline prompt, this builds.

    A single user-turn string returned it is -- the chat template, applied elsewhere it must be.
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    # The question text, first it comes.
    parts.append(f"Question: {question.text.strip()}")

    if question.qtype == QuestionType.OPEN or not question.options:
        # An open question, a short free-text answer it expects.
        parts.append("Answer briefly in one or two sentences.")
    else:
        # The options, in A-D order they appear -- missing letters, skipped they are.
        # A single letter, demand we must.
        option_lines: list[str] = []
        for letter in ("A", "B", "C", "D"):
            if letter in question.options:
                option_lines.append(f"{letter}) {question.options[letter]}")
        parts.append("\n".join(option_lines))
        parts.append(
            "Reply with ONLY the letter of the correct option (A, B, C, or D). "
            "No explanation, no punctuation -- the letter alone."
        )

    return "\n".join(parts)


def _render_mcq(text: str, options: dict[str, str]) -> str:
    """A question and its options -> the 'Question/A)/B)...' block, this renders.

    In A-D order the options appear; missing letters, skipped they are.
    """
    lines = [f"Question: {text.strip()}"]
    for letter in ("A", "B", "C", "D"):
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    return "\n".join(lines)


# A few solved exemplars -- the answer FORMAT they teach, and arithmetic they prime.
# From OUTSIDE the dev set drawn they are -- leakage into the benchmark, avoid we must.
_FEW_SHOT_EXAMPLES = [
    ("What is the capital of France?",
     {"A": "Madrid", "B": "Paris", "C": "Rome", "D": "Berlin"}, "B"),
    ("Which gas do plants absorb from the air for photosynthesis?",
     {"A": "Oxygen", "B": "Hydrogen", "C": "Carbon dioxide", "D": "Nitrogen"}, "C"),
    ("What is 6 multiplied by 7?",
     {"A": "42", "B": "36", "C": "48", "D": "49"}, "A"),
]


def _few_shot_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Solved exemplars before the target, this prepends -- the output format, they anchor.

    A single user-turn string returned it is -- the chat template, applied elsewhere it must be.
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    # The worked examples, first they come -- 'Answer: X' each one ends with.
    for ex_text, ex_opts, ex_gold in _FEW_SHOT_EXAMPLES:
        parts.append(_render_mcq(ex_text, ex_opts))
        parts.append(f"Answer: {ex_gold}")
        parts.append("")  # A blank line between examples, separation it gives.

    # The real question, last it stands.
    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Answer briefly in one or two sentences.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append("Answer with ONLY the letter (A, B, C, or D).")

    return "\n".join(parts)


def _cot_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Brief step-by-step reasoning, then 'Answer: X', this asks for.

    For arithmetic especially, helpful it is -- compute before committing, the model can.
    The parser keys on the 'Answer: X' marker (its highest-confidence pattern, that is).
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Think briefly, then answer in one or two sentences.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Think step by step briefly (one or two short sentences). "
            "Then on a new line, write your final choice as 'Answer: X', "
            "where X is one of A, B, C, or D."
        )

    return "\n".join(parts)


def _cot_v2(question: Question, context: list[RetrievedDoc] | None) -> str:
    """cot_v1 + an OPTION-MATCHING check + a HARD brevity cap, this is.

    Two motivating failures, this prompt answers:
      * run #7, qid 6702 (the OPTION-MATCHING slip): on a t-test MCQ the model REASONED correctly
        (df=17, ±2.110 -- option C's content) yet wrote 'Answer: B', whose only flaw was 'df=18'.
        B and C SHARED the conclusion ('do not reject'); the model matched the conclusion alone and
        never cross-checked the buried number against its own work. So: verify the chosen option
        matches EVERY computed detail -- not the conclusion only.
      * run #9, qid 6706 (the TRUNCATION loss): the model's set-up was CORRECT (z=-0.524/-0.842,
        the two equations) but it wrote ~5 paragraphs of LaTeX and hit the 256-token cap BEFORE the
        'Answer:' line -- so the parser fell back to a blind 'A'. At ~11 tok/s the 256 cap ≈ the 25s
        wall, so MORE tokens would only time out. The cure is FEWER tokens to the answer: cap the
        steps, BAN LaTeX (the token hog -- \\frac/\\mu/$...$ tripled the length), and DEMAND the
        'Answer:' line always be reached.
    For open questions, identical to cot_v1 it stays (no options).
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Think briefly, then answer in one or two sentences.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Solve in AT MOST 3 very short steps. Plain numbers ONLY -- NO LaTeX, no \\frac, "
            "no \\mu/\\sigma, no $...$; write 'mu'/'sigma' as words and keep each step under ~12 "
            "words. When two options share the same conclusion, pick the one whose numbers (values, "
            "signs, degrees of freedom) match your result EXACTLY -- not just the conclusion. You "
            "MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- always reach that line."
        )

    return "\n".join(parts)


# Maths-only worked exemplars -- the RATIO/PROPORTION setup, they teach (Q6777, level-2 ratio fail).
# The first example NO numbers in the question carries -- so introducing variables, it demonstrates
# (the live failure mode: the model picked a ratio-like number without setting up s/p first). The
# second a plain arithmetic case is, the format generality it preserves.
_MATHS_EXAMPLES = [
    (
        "Yesterday, a worker took three times as long to finish a task and produced half as many "
        "items. The worker's items-per-hour rate today is what percent of yesterday's rate?",
        {"A": "150", "B": "200", "C": "300", "D": "600"},
        "Let today hours=h, items=i; yesterday hours=3h, items=i/2.\n"
        "Today rate = i/h. Yesterday rate = (i/2)/(3h) = i/(6h).\n"
        "Today / Yesterday = (i/h) / (i/(6h)) = 6, so 600%.\n"
        "Answer: D",
    ),
    (
        "If a rectangle has length 15 cm and width 8 cm, what is its area in square centimetres?",
        {"A": "46", "B": "90", "C": "120", "D": "160"},
        "Area = length x width = 15 x 8 = 120.\n"
        "Answer: C",
    ),
]


def _cot_maths_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """cot_v2 + two worked Maths exemplars -- the level-2 ratio fail (Q6777), this targets.

    Q6777 ("computer speed-to-price ratio is what percent...") the model answered B (32) instead
    of D (400) because cot_v2 ALONE never set up variables: the question carries NO numbers, so
    without an example to anchor the "let s=..., p=..." move, the chain skipped straight to a
    plausible-looking number. Two exemplars we prepend:
      * a ratio question with NO numbers (introduce variables, then compute) -- Q6777's mode.
      * a plain area calculation -- the format on simple arithmetic, also it covers.
    The directives stay cot_v2's (brevity cap + option-matching + 'Answer:' must be reached).
    For open questions, identical to cot_v1 it stays (no options).
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do (Maths usually has none).
    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Think briefly, then answer in one or two sentences.")
        return "\n".join(parts)

    # The worked Maths exemplars, first they come -- the reasoning shape, they teach.
    for ex_text, ex_opts, ex_solution in _MATHS_EXAMPLES:
        parts.append(_render_mcq(ex_text, ex_opts))
        parts.append(ex_solution)
        parts.append("")  # A blank line between examples, separation it gives.

    # The real question, last it stands.
    parts.append(_render_mcq(question.text, question.options))
    parts.append(
        "Solve in AT MOST 3 very short steps. If the question has no numbers, introduce "
        "variables (e.g. let s = speed, p = price) and write the relationships first. "
        "Plain numbers ONLY -- NO LaTeX, no \\frac, no \\mu/\\sigma, no $...$; write "
        "'mu'/'sigma' as words and keep each step under ~12 words. When two options share "
        "the same conclusion, pick the one whose numbers (values, signs, degrees of freedom) "
        "match your result EXACTLY -- not just the conclusion. You MUST end on a new line with "
        "'Answer: X' (X = A, B, C, or D) -- always reach that line."
    )

    return "\n".join(parts)


# Entertainment-domain exemplars -- film / music / TV, the three biggest subdomains they cover.
# From OUTSIDE the dev set drawn they are (the dev set holds Pulp Fiction, Dark Side of the Moon, the
# Hunger Games, the iPhone -- none reused here, leakage we avoid). Their job is TWOFOLD: anchor the
# 'Answer: X' format AND prime the pop-culture register (a director, an album's artist, a show's setting)
# so the small model retrieves the RIGHT kind of fact, not a generic plausible-sounding one.
_ENTERTAINMENT_EXAMPLES = [
    ("Who directed the 1975 film 'Jaws'?",
     {"A": "George Lucas", "B": "Steven Spielberg",
      "C": "Francis Ford Coppola", "D": "Ridley Scott"}, "B"),
    ("Which artist recorded the 1982 album 'Thriller'?",
     {"A": "Prince", "B": "Stevie Wonder",
      "C": "Michael Jackson", "D": "Lionel Richie"}, "C"),
    ("In the US sitcom 'The Office', what is the name of the paper company?",
     {"A": "Dunder Mifflin", "B": "Sabre",
      "C": "Wernham Hogg", "D": "Initech"}, "A"),
]


def _few_shot_entertainment(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Few-shot tuned for the Entertainment race -- pop-culture recall, this serves (comp 0).

    Entertainment is single-fact recall across film, music, TV, books, video games and sport -- the very
    shape where explicit chain-of-thought HURTS (the small model invents justification and drifts off a
    name it already knew). So NO reasoning is asked for: domain exemplars prime the register, the
    retrieved Wikipedia/FAISS evidence (when the gate fired) grounds the harder later-level facts, and a
    single committed letter is all we demand. Identical to `few_shot_v1` in shape, but its three generic
    exemplars (capital-of, photosynthesis, 6x7) are swapped for film/music/TV ones AND a domain-aware
    instruction is added -- so the prime matches the questions actually asked.

    For open questions, a short free-text answer it keeps (the race is MCQ in practice).
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do (entertainment facts Wikipedia covers well).
    if context:
        parts.append(_build_context_block(context))

    # The entertainment exemplars, first they come -- 'Answer: X' each one ends with.
    for ex_text, ex_opts, ex_gold in _ENTERTAINMENT_EXAMPLES:
        parts.append(_render_mcq(ex_text, ex_opts))
        parts.append(f"Answer: {ex_gold}")
        parts.append("")  # A blank line between examples, separation it gives.

    # The real question, last it stands.
    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Answer briefly -- the name, title, or fact only.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "This is an Entertainment trivia question (film, music, TV, books, video games, or sport). "
            "Identify the specific work, person, year, or fact asked for. If reference sources are given "
            "above, base the answer on them; otherwise use well-known facts. Answer with ONLY the letter "
            "(A, B, C, or D) -- no explanation, the letter alone."
        )

    return "\n".join(parts)


# ===========================================================================
# Adaptive-routing research strategies -- the four experimental conditions, these are.
#
# The adaptive prompt routing experiment (src/experiments/adaptive_routing.py) tests one claim:
# a prompt that HELPS one reasoning category may HURT another. So four named, self-contained
# strategies it needs -- each a deliberate point on the "how much explicit reasoning?" axis:
#
#   direct_answer               -- minimal reasoning (recall/commonsense; overthinking, it avoids).
#   generic_cot                 -- plain "think step by step" (the universal CoT baseline).
#   structured_enumeration_cot  -- enumerate every case/event, ordered, with boundary checks,
#                                  count ONLY after listing (the clock-chime / interval-counting fix).
#   checklist_cot               -- a verification checklist: assumptions, skipped cases, final
#                                  cross-check against the options (logical / multi-hop questions).
#
# Distinct from cot_v2 they deliberately are: cot_v2 carries a HARD ≤3-step brevity cap (born of the
# t-test LaTeX token-blowup) -- and that very cap is what made the model GUESS the clock-chime answer
# before it had counted. These research strategies separate the two regimes the cap conflated.
# ===========================================================================

def _direct_answer(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Concise, minimal-reasoning answer -- recall and commonsense, this serves.

    The hypothesis it embodies: for factual recall and everyday judgement, explicit chains
    HURT (they invite hallucinated justification and arithmetic drift on a non-arithmetic Q).
    A single committed answer, demand we do -- no scratch-work the small model can wander in.
    """
    parts: list[str] = []

    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Answer in as few words as possible -- the fact only, no explanation.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Answer immediately with ONLY the letter (A, B, C, or D). No reasoning, no "
            "explanation, no punctuation -- the single letter alone."
        )

    return "\n".join(parts)


def _generic_cot(question: Question, context: list[RetrievedDoc] | None) -> str:
    """The plain 'think step by step' baseline -- the universal CoT, this is.

    NO brevity cap, NO option-matching directive, NO domain exemplars: the vanilla chain-of-thought
    every paper reaches for first. The control against which the SPECIALISED chains (enumeration,
    checklist) and the ADAPTIVE router are measured. Open questions, a brief free-text answer keep.
    """
    parts: list[str] = []

    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Let's think step by step, then give the final answer in one sentence.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Let's think step by step. Work through the reasoning, then on a new line write "
            "your final choice as 'Answer: X', where X is one of A, B, C, or D."
        )

    return "\n".join(parts)


def _structured_enumeration_cot(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Enumerate-first counting -- the clock-chime / interval-counting failure, this targets.

    The motivating loss (qid 6712): "how many chimes between 5:10 and 7:35?" -- under cot_v2's ≤3-step
    cap the model wrote "Step 2: count the chimes" and then GUESSED, never listing them. The cure is the
    opposite of a brevity cap: force an explicit, ordered enumeration of EVERY case/event BEFORE any
    count, and a boundary check on the endpoints (off-by-one, the classic counting bug it is).

    For open questions, a brief free-text answer it keeps (enumeration suits options/counts, not prose).

    The TRUNCATION fix (live Maths run, qids 6993 + 6830): both chains were on the CORRECT track --
    6993 had found the two critical points, 6830 had the SE and both z-scores -- but each wrote ~6
    paragraphs of LaTeX (\\(...\\), \\[...\\]) and hit the 300-token cap BEFORE the 'Answer:' line, so
    the parser fell back to a blind guess (both LOST a winnable question). At ~16 tok/s the 300 cap ≈ the
    25s wall (both ran 26-27s), so MORE tokens would only time out -- the cure is FEWER tokens to the
    answer. The trailing "no LaTeX" line this prompt HAD was ignored; cot_v2's forceful, FRONT-LOADED
    ban demonstrably suppresses LaTeX (every cot_v2 chain in that run was plain text), so port it here.
    """
    parts: list[str] = []

    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("List each relevant item or event in order, then give the answer in one sentence.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Plain text ONLY -- NO LaTeX, no \\frac, no \\(...\\), no \\[...\\], no $...$; write "
            "'mu'/'sigma' as words and keep EVERY line under ~12 words (LaTeX overruns the token "
            "budget before the answer -- a guaranteed loss).\n"
            "Solve by EXPLICIT ENUMERATION -- do NOT guess a total.\n"
            "1. List EVERY relevant case/event/item ONE PER LINE, in order (chronological for times, "
            "ascending for numbers). Write the value beside each.\n"
            "2. Boundary check: state the first and last item that qualify, and confirm each endpoint "
            "is inside the asked range (watch the off-by-one).\n"
            "3. ONLY NOW add them up -- show the running total.\n"
            "You MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- ALWAYS reach that line."
        )

    return "\n".join(parts)


def _implication_cot(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Explicit logical-DIRECTION scaffold -- implication / induction / contrapositive questions.

    The motivating loss (live qid 6737, level 11): "whenever S(k) is true, S(k+1) is true; S(n0) is
    false; strongest conclusion?". Qwen-7B wrote "n0 is a counterexample, so all HIGHER values are too"
    -> C, the WRONG direction (falsity propagates BACKWARD, not forward). It failed under cot_v2,
    generic_cot, checklist_cot AND a 5-vote self-consistency -- a systematic directional misconception,
    not a format or sampling problem. So this prompt scaffolds the ONE thing those all skipped: write the
    implication, write its VALID contrapositive, and forbid the converse/inverse outright.

    For open questions, a brief reasoned answer it keeps.
    """
    parts: list[str] = []

    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("State the implication and its contrapositive, then answer in one sentence.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "This is a logical-implication question -- reason about DIRECTION explicitly:\n"
            "1. Write the rule as an implication 'P -> Q'. For an induction rule write it as "
            "'S(k) true -> S(k+1) true'.\n"
            "2. Write the VALID contrapositive: 'not Q -> not P'. For induction this means truth "
            "propagates FORWARD (k to k+1), so FALSITY propagates BACKWARD: if S(k+1) is false then "
            "S(k) is false. A false case forces all SMALLER cases false, NOT larger ones.\n"
            "3. Do NOT assume the converse 'Q -> P', and do NOT assume the inverse 'not P -> not Q' -- "
            "neither is valid.\n"
            "4. Test EACH option using ONLY the rule and its contrapositive; reject any option that "
            "would need the converse or the inverse.\n"
            "Then on a new line write 'Answer: X' (X = A, B, C, or D). Plain text, no LaTeX."
        )

    return "\n".join(parts)


def _checklist_cot(question: Question, context: list[RetrievedDoc] | None) -> str:
    """A verification checklist -- logical-reasoning and multi-hop questions, this serves.

    The failure mode it answers: on "which of the following is true?" / multi-step questions the small
    model commits early to a plausible option and never tests the OTHERS, nor cross-checks its chosen
    option's buried details (the cot_v2 option-matching slip, generalised). So a checklist we impose:
    restate, surface hidden assumptions, evaluate EACH option / hop, then validate the pick.

    For open questions, a brief reasoned answer it keeps.
    """
    parts: list[str] = []

    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append(
            "Reason in a short checklist (what is asked / what is assumed / the conclusion), "
            "then give the final answer in one sentence."
        )
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Work through this checklist:\n"
            "1. Restate what is being asked in one line.\n"
            "2. List any assumptions or hidden constraints.\n"
            "3. Evaluate EACH option (or EACH reasoning hop) in turn -- mark it true or false and why.\n"
            "4. Validate: does the surviving option match EVERY detail (numbers, signs, scope), not just "
            "the broad conclusion? Re-check any you skipped.\n"
            "Then on a new line write 'Answer: X' (X = A, B, C, or D)."
        )

    return "\n".join(parts)


# --- Strategy registry ---

# A name -> builder function, this dict is.
# To add a new strategy, register it here you must (the human record, in memory/prompts.md it lives).
_REGISTRY: dict[str, object] = {
    "zero_shot_v1": _zero_shot_v1,
    "few_shot_v1": _few_shot_v1,
    "cot_v1": _cot_v1,
    "cot_v2": _cot_v2,
    "cot_maths_v1": _cot_maths_v1,
    "few_shot_entertainment": _few_shot_entertainment,
    # Adaptive-routing research conditions (src/experiments/adaptive_routing.py).
    "direct_answer": _direct_answer,
    "generic_cot": _generic_cot,
    "structured_enumeration_cot": _structured_enumeration_cot,
    "checklist_cot": _checklist_cot,
    "implication_cot": _implication_cot,
}


class PromptBuilder:
    """A named strategy -> a prompt string. By name, strategies are registered.

    In memory/prompts.md the human-readable record lives; here, its code form.
    Drift between the two, allow we must not.
    """

    def __init__(self, strategy: str = "zero_shot_v1"):
        # The chosen strategy name, stored it is -- validated at build time.
        self.strategy = strategy

    def build(self, question: Question, context: list[RetrievedDoc] | None = None) -> str:
        """A question (and optional raw evidence) -> the final prompt string, this turns.

        Context, when given, as RAW retrieved text it is injected -- never a generated answer.
        The chat template, NOT applied here it is -- that, the inference engine does (D-003).
        """
        # The strategy function, looked up it is -- unknown names, loudly rejected they are.
        builder_fn = _REGISTRY.get(self.strategy)
        if builder_fn is None:
            known = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"Unknown prompt strategy: {self.strategy!r}. "
                f"Known strategies, these are: {known}"
            )
        return builder_fn(question, context)  # type: ignore[operator]


class RoutingPromptBuilder:
    """A drop-in PromptBuilder that PICKS the strategy per question -- adaptive routing in live play.

    The `QAPipeline` only calls `.build(question, context)` and reads `.strategy` for the log; a duck-type
    of `PromptBuilder`, this is. On each build we ask a `ReasoningRouter` which strategy this question's
    reasoning shape wants, set `self.strategy` to that name (so the EvalRecord logs WHICH prompt actually
    ran -- per question it now varies), and delegate to that strategy's builder.

    The POLICY (category -> strategy) and the FALLBACK are the router's. For Maths live play the
    conservative policy is: re-route ONLY the counting/temporal/enumeration shapes to
    `structured_enumeration_cot` (the proven clock-chime fix) and leave everything else on the
    known-good `cot_v2` -- so concept/stats questions never regress.
    """

    def __init__(self, router=None, policy=None, fallback_strategy: str = "cot_v2"):
        # Imported here (not at module top) -- the classify package importing prompting would otherwise
        # risk a cycle, and most PromptBuilder users never need the router.
        from classify.reasoning_router import ReasoningRouter
        self.router = router or ReasoningRouter(policy=policy, fallback_strategy=fallback_strategy)
        # Set per-build to the chosen strategy; before the first build, a label it carries.
        self.strategy: str = "adaptive"
        self._builders: dict[str, PromptBuilder] = {}

    def build(self, question: Question, context: list[RetrievedDoc] | None = None) -> str:
        signal, strat = self.router.route(question)
        self.strategy = strat   # the EvalRecord reads this AFTER build -> logs the routed strategy.
        if strat not in self._builders:
            self._builders[strat] = PromptBuilder(strat)
        return self._builders[strat].build(question, context)
