"""Reasoning-type classifier and prompt router -- the brain of the adaptive experiment, this is.

The existing `QuestionClassifier` (classifier.py) answers "what TOPIC / what TYPE / what LANGUAGE",
and gates retrieval and the calculator. This module answers a DIFFERENT question: "what shape of
REASONING does this need?" -- arithmetic, interval counting, factual recall, logic, and so on -- and
from that, "which PROMPT strategy should answer it?".

Why a separate axis: a prompt that helps one reasoning shape can hurt another (the experiment's whole
hypothesis). Structured enumeration rescues the clock-chime counting question; the very same prompt,
on a one-fact recall question, invites the small model to over-reason and drift. So we classify the
reasoning shape FIRST, then route -- rather than force one universal prompt onto every shape.

Rule-based and transparent it stays (regex + keyword precedence) -- comfortably under the latency
budget, and every decision auditable from its `evidence` string. A learned router, a future upgrade
it is; the seam (`ReasoningRouter.route`) the same would stay.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from schemas import Question


class ReasoningCategory(str, Enum):
    """The reasoning shapes we route on -- eight, these are."""
    ARITHMETIC = "arithmetic"                  # numeric computation: sums, percentages, word-number problems.
    TEMPORAL_REASONING = "temporal_reasoning"  # times/dates/durations, ordering before-after, elapsed time.
    INTERVAL_COUNTING = "interval_counting"    # counting events over a CLOCK/time range (the chime question).
    DISCRETE_ENUMERATION = "discrete_enumeration"  # counting discrete cases: ways, divisors, integers, subsets.
    FACTUAL_QA = "factual_qa"                   # single-fact recall: who/when/where/capital-of.
    COMMONSENSE = "commonsense"                 # everyday judgement, no recall and no computation.
    LOGICAL_REASONING = "logical_reasoning"     # if-then, "which is true", validity, implication.
    MULTI_HOP = "multi_hop"                     # chained reasoning across several facts/relations.


@dataclass
class ReasoningSignal:
    """The classifier's verdict -- the category, and WHY (the rule that fired), for the log it carries."""
    category: ReasoningCategory
    evidence: str            # The pattern/cue that decided it -- audit any routing choice with this, we can.


# ---------------------------------------------------------------------------
# Cue patterns -- compiled once, matched in a fixed PRECEDENCE order (below).
# Specific shapes (interval counting) before general ones (factual recall), we test.
# ---------------------------------------------------------------------------

# A clock time (5:10, 12:45) or am/pm/o'clock marker -- the strongest "this is about times" tell.
_CLOCK_RE = re.compile(
    r"\b\d{1,2}:\d{2}\b"
    r"|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b"
    r"|\bo'?clock\b",
    re.IGNORECASE,
)

# Softer temporal vocabulary -- hours/minutes/days/clocks/schedules, these are.
# Plurals matter ("seconds", "minutes") -- the bare singular missed them and dropped interval-counting
# questions to discrete enumeration; `s?` on the units, the fix it is.
_TIME_WORD_RE = re.compile(
    r"\b(?:hours?|minutes?|seconds?|clock|chimes?|schedule|noon|midnight|"
    r"days?|weeks?|months?|years?|calendar|"
    r"morning|afternoon|evening|elapsed|duration)\b",
    re.IGNORECASE,
)

# A "count something" intent -- how many / number of / total / count.
_COUNT_RE = re.compile(
    r"\bhow\s+many\b"
    r"|\bnumber\s+of\b"
    r"|\btotal\s+number\b"
    r"|\bhow\s+much\b"
    r"|\bcount\s+(?:the|how|all)\b",
    re.IGNORECASE,
)

# A range / interval phrasing -- "between X and Y", "from X to Y", "every N minutes".
# The "every N <unit>" branch accepts a SPELLED-OUT number too (every TWO seconds), not only digits:
# the cyclic-coincidence counting questions ("red blinks every two seconds ... how many times do all
# coincide") write the period in words, so the digit-only pattern missed them and they fell through to
# commonsense -> cot_v2, which botched the inclusive off-by-one (live qid 6861, a recurring death).
_RANGE_RE = re.compile(
    r"\bbetween\b[\s\S]*?\band\b"
    r"|\bfrom\b[\s\S]*?\bto\b"
    r"|\bevery\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"\s*-?\s*(?:minute|hour|second|day)",
    re.IGNORECASE,
)

# The GRE-style PAIRED True/False format ("Statement 1 | ... Statement 2 | ..."). These are CONCEPT
# judgements -- often abstract algebra ('subset', 'factor group') -- NOT combinatorics; but those very
# words trip the enumeration cue below, and the enumerate-then-count scaffold produces GARBAGE on a
# True/False judgement (live qid 6768 wrote "Running total: 1"). Detected at TOP precedence and sent to
# logical reasoning (-> cot_v2 under the live Maths policy). Both halves required -> never a false hit.
_STATEMENT_TF_RE = re.compile(
    r"\bstatement\s+1\b[\s\S]*\bstatement\s+2\b",
    re.IGNORECASE,
)

# "Formula math" signals -- abstract algebra (factor/quotient groups) and inferential statistics (normal
# distribution, z-scores, confidence intervals, standard deviation, variance). These are COMPUTED from a
# formula, NOT list-and-counted -- yet a single word in them ('factor group', 'how many ... between X and
# Y') trips the enumeration cue, and the enumerate-every-case scaffold then drives the small model into
# long LaTeX that overruns the 300-token cap. Every such turn was a WINNABLE question lost to truncation
# (live qids 6767 factor-group, 6954 normal-dist, 6830 / 7004 stats -- each one step from the answer when
# the cap cut it off). Routed to cot_v2 instead they stay terse and FINISH (cot_v2 reliably suppresses
# LaTeX -- every stats chain it produced stayed plain text). NARROW on purpose: bare 'factor' is NOT here
# (so 'how many factors of 360' stays enumeration) and ring/field notation (Z_n) is NOT here (so the
# find-all-zeros questions are untouched) -- only the two vocabularies that demonstrably LaTeX-truncate.
_FORMULA_MATH_RE = re.compile(
    r"\b(?:factor|quotient)\s+group"
    r"|normal(?:ly)?\s+distribut"
    r"|\bz-?scores?\b"
    r"|confidence\s+(?:interval|level)"
    r"|standard\s+deviation"
    r"|\bvariance\b",
    re.IGNORECASE,
)

# Discrete-enumeration vocabulary -- combinatorics and "how many <countable>" without a clock.
_ENUMERATION_RE = re.compile(
    r"\bhow\s+many\s+ways\b"
    r"|\bnumber\s+of\s+ways\b"
    r"|\bpermutation|\bcombination"
    r"|\bdivisors?\b|\bmultiples?\b|\bfactors?\b"
    r"|\bsubsets?\b|\barrangements?\b"
    r"|\bdistinct\b"
    r"|\bhow\s+many\s+(?:integers?|numbers?|primes?|digits?|pairs?|triangles?|sides?|"
    r"diagonals?|elements?|students?|people|items?)\b",
    re.IGNORECASE,
)

# Arithmetic operators / computation words -- a number nearby, the router additionally requires.
_ARITH_OP_RE = re.compile(
    r"[+\-*/×÷=]"
    r"|\bplus\b|\bminus\b|\btimes\b|\bdivided\s+by\b|\bmultiplied\b"
    r"|\bsquared\b|\bcubed\b|\bsquare\s+root\b|\bpower\b"
    r"|\bsum\s+of\b|\bproduct\s+of\b|\baverage\s+of\b|\bmean\s+of\b"
    r"|\bpercent(?:age)?\b|\bratio\b|\bremainder\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")

# Logical-reasoning cues -- validity, implication, "which is true".
_LOGIC_RE = re.compile(
    r"\bwhich\s+of\s+the\s+following\s+(?:is|must)\b"  # "...is true" / "...must (also) be true".
    r"|\bif\b[\s\S]*?\bthen\b"
    r"|\bwhenever\b"                                   # "whenever S(k) is true ..." -- the induction tell.
    r"|\bstrongest\s+conclusion\b"
    r"|\bcan\s+be\s+(?:concluded|drawn)\b"             # "...which can be drawn", "what can be concluded".
    r"|\bwhat\s+can\s+(?:we|you|be)\s+conclude"
    r"|\bwhich\s+must\s+(?:be|also)\b|\bwhich\s+must\s+be\s+true\b"
    r"|\bmust\s+(?:also\s+)?be\s+(?:true|false)\b"
    r"|\bcannot\s+be\s+(?:true|false)\b"
    r"|\bmust\s+(?:also\s+)?(?:be\s+true|follow|hold)\b"
    r"|\bimplies?\b|\bimplication\b"
    r"|\bvalid\s+argument\b|\bvalid\s+conclusion\b"
    r"|\bcontradict|\bsyllogism\b"
    r"|\bit\s+follows\s+that\b"
    r"|\b(?:all|some|no|none|every|each)\s+\w+\s+(?:is|are|can|cannot)\b"
    r"|\bnecessar(?:y|ily)\b",
    re.IGNORECASE,
)

# Factual-recall cues -- mirror the retrieval classifier's, deliberately (one fact, recalled it is).
_FACTUAL_RE = re.compile(
    r"\bwho\s+(?:was|is|were|are|invented|discovered|wrote|created|founded|led|won|painted|composed)\b"
    r"|\bwhen\s+(?:was|did|were|is)\b"
    r"|\bwhere\s+(?:was|did|is|are)\b"
    r"|\bwhich\s+year\b|\bwhat\s+year\b"
    r"|\bcapital\s+of\b|\bauthor\s+of\b|\binventor\s+of\b"
    r"|\bchemical\s+symbol\b"
    r"|\bknown\s+as\b|\bcalled\b"
    r"|\bwhich\s+(?:planet|element|country|city|animal|gas|metal|ocean|river|mountain|language|continent)\b"
    r"|\bwhat\s+is\s+the\s+(?:capital|symbol|largest|smallest|longest|tallest|name)\b",
    re.IGNORECASE,
)

# Multi-hop cues -- a nested relative clause ("the X of the Y that Z"), explicit chaining, or a
# possessive entity-chain ("the <attr> of the <entity> that/which ..." / "named after the ...").
# Checked BEFORE factual recall on purpose: these all ALSO carry a factual cue ("capital of",
# "chemical symbol"), so without first-look priority they would mis-route to single-fact recall.
_MULTI_HOP_RE = re.compile(
    r"\b(?:who|what|which|where)\b[\s\S]*?\b(?:that|who|which|whose)\b[\s\S]*?\b(?:is|was|are|were|won|wrote|invented|ruled|founded|directed|discovered|hosted|forms?|located)\b"
    r"|\b(?:of|in|by|after|on|from)\s+the\s+\w+\s+(?:that|which|who|whose|where)\b"   # "in the country that ..."
    r"|\bnamed\s+after\b"
    r"|\bthe\s+\w+\s+who\s+(?:wrote|invented|discovered|painted|composed|founded|directed|ruled|won|created|built|led|designed)\b"  # "the author who wrote ..."
    r"|\bfirst\b[\s\S]*?\bthen\b"
    r"|\band\s+then\b"
    r"|\bafter\s+that\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# The default routing policy -- category -> prompt strategy.
# The experiment's hypothesis, encoded here it is. One edit, the whole policy it changes.
# ---------------------------------------------------------------------------
DEFAULT_ROUTING_POLICY: dict[ReasoningCategory, str] = {
    # Recall and everyday judgement: explicit chains HURT (hallucinated justification, drift).
    ReasoningCategory.FACTUAL_QA: "direct_answer",
    ReasoningCategory.COMMONSENSE: "direct_answer",
    # Plain computation: step through it, but do NOT over-enumerate.
    ReasoningCategory.ARITHMETIC: "generic_cot",
    # Anything that must lay out events/cases in order, then count: enumerate first.
    ReasoningCategory.TEMPORAL_REASONING: "structured_enumeration_cot",
    ReasoningCategory.INTERVAL_COUNTING: "structured_enumeration_cot",
    ReasoningCategory.DISCRETE_ENUMERATION: "structured_enumeration_cot",
    # Multi-clause / validity reasoning: a verification checklist over each option/hop.
    ReasoningCategory.LOGICAL_REASONING: "checklist_cot",
    ReasoningCategory.MULTI_HOP: "checklist_cot",
}

# When the classifier is unsure (no rule fires confidently), this strategy we fall back to.
_FALLBACK_STRATEGY = "generic_cot"


# The CONSERVATIVE policy for live Maths (notebook 03, comp 3). The offline experiment proved structured
# enumeration fixes interval-counting (the clock-chime death, 0.4 -> 1.0) and helps temporal/discrete; it
# did NOT test the formal-maths/stats questions the live Maths comp also throws (group theory, t-tests),
# which the battle-tested `cot_v2` already handles. So we re-route ONLY the shapes we have evidence for,
# and EVERYTHING ELSE (arithmetic, logic, concept/stats -> the fallback) stays on `cot_v2`. Minimal blast
# radius, maximal targeting of the documented failure. Pair with max_new_tokens>=512 so the longer
# structured chains reach their 'Answer:' line.
# B3 (2026-06-02, the STRUCTURAL fix): only the TIME-interval shapes route to structured enumeration.
# DISCRETE_ENUMERATION was REMOVED -> it falls back to cot_v2. Rationale: the discrete bucket is the
# polluted one -- 'divisor' / 'subset' / 'factor' / 'distinct' / 'how-many-X' are shared vocabulary between
# genuine combinatorics AND formula/number-theory (factor groups, GCDs, power sets, normal distributions),
# and NO regex separates them. Every live truncation death was a non-counting question that leaked into the
# enumerate-and-count scaffold and then ran LaTeX past the 300-token cap. cot_v2 NEVER truncates, so routing
# the whole discrete bucket to it STRUCTURALLY ends the bleed (vs B1a/B1b/B2 cue-patching, which leaked a new
# truncation every run). INTERVAL_COUNTING / TEMPORAL_REASONING need a clock/time signal -> genuinely
# time-ordered, short, no LaTeX -> safe in structured, where the boundary check fixes the off-by-one. Cost:
# genuine combinatorics loses the scaffold, but it is rare and cot_v2 handles it terse. This partly reverses
# the OFFLINE 'discrete enumeration helped' finding -- but offline had no 30s-wall truncation. See
# [[maths-live-routing-stack]] for the full evidence trail.
MATHS_LIVE_POLICY: dict[ReasoningCategory, str] = {
    ReasoningCategory.INTERVAL_COUNTING: "structured_enumeration_cot",
    ReasoningCategory.TEMPORAL_REASONING: "structured_enumeration_cot",
    # DISCRETE_ENUMERATION + arithmetic / logical / multi_hop / factual / commonsense -> cot_v2 fallback.
}
MATHS_LIVE_FALLBACK = "cot_v2"


class ReasoningClassifier:
    """A question -> its reasoning shape, in a single cheap pass this decides.

    Precedence matters: the SPECIFIC shapes first we test (interval counting, enumeration, arithmetic,
    logic, multi-hop), then the GENERAL ones (factual recall, commonsense). The first rule to fire wins
    -- and its cue we keep as `evidence`, so any routing choice later we can explain.
    """

    def classify(self, question: Question) -> ReasoningSignal:
        text = question.text or ""

        has_clock = bool(_CLOCK_RE.search(text))
        has_time_word = bool(_TIME_WORD_RE.search(text))
        has_count = bool(_COUNT_RE.search(text))
        has_range = bool(_RANGE_RE.search(text))

        # 0. STATEMENT-PAIR True/False -- the GRE "Statement 1 | ... Statement 2 |" concept format.
        #    HIGHEST precedence on purpose: 'subset' / 'factor (group)' / 'distinct' inside these would
        #    otherwise hijack the enumeration cue, and the enumerate-and-count scaffold produces nonsense
        #    on a True/False judgement. A clean two-statement validity check (-> cot_v2) is the right shape.
        if _STATEMENT_TF_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.LOGICAL_REASONING,
                "paired True/False statements (Statement 1 | ... Statement 2 |)",
            )

        # 0b. FORMULA MATH -- abstract algebra (factor/quotient groups) and inferential stats. These are
        #     COMPUTED from a formula, not enumerated; but their vocabulary trips the enumeration cue and
        #     the enumerate-and-count scaffold then runs long LaTeX past the 300-token cap -- a guaranteed
        #     loss (live 6767/6954/6830/7004). Classify as ARITHMETIC -> cot_v2 (terse, no LaTeX, finishes).
        #     NARROW: genuine combinatorics ('how many ways', divisors) and clock counting carry none of
        #     these signals, so they still reach structured enumeration below.
        if _FORMULA_MATH_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.ARITHMETIC,
                "formula-math (factor/quotient group or inferential stats) -- compute, do not enumerate",
            )

        # 1. INTERVAL_COUNTING -- counting over a CLOCK/time range (the chime question's exact shape).
        #    A count intent AND a clock/time signal AND a range, all three together it needs.
        if has_count and (has_clock or has_time_word) and (has_range or has_clock):
            return ReasoningSignal(
                ReasoningCategory.INTERVAL_COUNTING,
                "count + time/clock + range cue (e.g. 'how many ... between 5:10 and 7:35')",
            )

        # 2. DISCRETE_ENUMERATION -- combinatorics / "how many <countable>" with NO clock.
        if _ENUMERATION_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.DISCRETE_ENUMERATION,
                "enumeration cue (ways / divisors / how-many-integers / distinct)",
            )
        #    A bare "how many" over a numeric range (no time) -- enumerate the candidates, still.
        if has_count and has_range and not (has_clock or has_time_word):
            return ReasoningSignal(
                ReasoningCategory.DISCRETE_ENUMERATION,
                "count + numeric range, no time cue",
            )

        # 3. TEMPORAL_REASONING -- times/dates/durations, but NOT a pure count (those caught above).
        if has_clock or (has_time_word and (has_range or re.search(r"\bbefore\b|\bafter\b|\bhow\s+long\b|\belapsed\b|\bwhat\s+(?:time|day)\b", text, re.IGNORECASE))):
            return ReasoningSignal(
                ReasoningCategory.TEMPORAL_REASONING,
                "time/clock/duration reasoning (ordering or elapsed time)",
            )

        # 4. LOGICAL_REASONING -- validity / implication / "which is true" / induction.
        #    BEFORE arithmetic on purpose: an induction question ("whenever S(k) true, S(k+1) must be
        #    true ... strongest conclusion?") carries an incidental '+1' and digits, and a stats MCQ
        #    ("two-sample t-test, p = 0.03 ... which is true?") carries hyphens that look like the minus
        #    operator -- so the arithmetic rule would HIJACK both. The reasoning here is logical, not
        #    computational, so the logic cue wins first. (live qid 6737 was the motivating misroute.)
        if _LOGIC_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.LOGICAL_REASONING,
                "logic cue (if-then / which-is-true / whenever / strongest-conclusion / validity)",
            )

        # 5. ARITHMETIC -- an operator/computation word WITH a digit present.
        if _ARITH_OP_RE.search(text) and _DIGIT_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.ARITHMETIC,
                "arithmetic operator/word + a number",
            )
        #    A numeric word-problem "how many ... left/remain/total" with two numbers -- arithmetic too.
        if has_count and len(_DIGIT_RE.findall(text)) >= 2 and re.search(r"\b(?:left|remain|remaining|altogether|in\s+total|each)\b", text, re.IGNORECASE):
            return ReasoningSignal(
                ReasoningCategory.ARITHMETIC,
                "numeric word-problem (count + >=2 numbers + total/left cue)",
            )

        # 6. MULTI_HOP -- a nested relative clause or explicit chaining of steps.
        if _MULTI_HOP_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.MULTI_HOP,
                "chained/nested clause (e.g. 'the X of the Y that Z')",
            )

        # 7. FACTUAL_QA -- single-fact recall cues, or a dense run of proper nouns.
        if _FACTUAL_RE.search(text):
            return ReasoningSignal(
                ReasoningCategory.FACTUAL_QA,
                "factual recall cue (who/when/where/capital-of)",
            )
        tokens = text.split()
        proper_nouns = [t for t in tokens[1:] if t and t[0].isupper() and t.isalpha()]
        if len(proper_nouns) >= 3:
            return ReasoningSignal(
                ReasoningCategory.FACTUAL_QA,
                f"proper-noun density ({len(proper_nouns)} capitalised tokens)",
            )

        # 8. COMMONSENSE -- nothing specific fired; everyday judgement, assume we do.
        return ReasoningSignal(
            ReasoningCategory.COMMONSENSE,
            "no specific reasoning cue -- everyday judgement assumed",
        )


class ReasoningRouter:
    """A question -> (reasoning category, prompt-strategy name). The adaptive policy, this is.

    The policy a plain dict is (`DEFAULT_ROUTING_POLICY`) -- swap it for an ablation, or hand a custom
    one in, you may. Unknown / unmapped category -> the fallback strategy (`generic_cot`), so a NEW
    category added to the enum never silently picks nothing.
    """

    def __init__(
        self,
        classifier: ReasoningClassifier | None = None,
        policy: dict[ReasoningCategory, str] | None = None,
        fallback_strategy: str = _FALLBACK_STRATEGY,
    ):
        self.classifier = classifier or ReasoningClassifier()
        self.policy = policy or DEFAULT_ROUTING_POLICY
        self.fallback_strategy = fallback_strategy

    def route(self, question: Question) -> tuple[ReasoningSignal, str]:
        """The signal AND the chosen strategy, both we return -- the signal for the log, the strategy to run."""
        signal = self.classifier.classify(question)
        strategy = self.policy.get(signal.category, self.fallback_strategy)
        return signal, strategy
