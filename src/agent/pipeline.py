"""The QA pipeline -- the orchestrator the notebook calls, this is.

Question -> classify -> (retrieve?) -> prompt -> generate -> (tool?) -> parse -> Prediction.
The single seam between 'the system' and 'the experiment harness', it forms.
Against the 30s budget, every stage is timed.
"""
from __future__ import annotations

import json
import re

from agent.voting import majority_vote
from classify.classifier import QuestionClassifier
from inference.engine import LLMEngine
from prompting.builder import PromptBuilder
from schemas import Prediction, Question
from utils.timing import LatencyGuard, stopwatch


def _render_mcq_block(question: Question) -> str:
    """A Question -> the 'Question/A)/B)...' block, for the tool prompts this renders (A-D order)."""
    lines = [f"Question: {question.text.strip()}"]
    for letter in ("A", "B", "C", "D"):
        if letter in question.options:
            lines.append(f"{letter}) {question.options[letter]}")
    return "\n".join(lines)


def _extract_first_json(text: str) -> dict | None:
    """The first balanced {...} object in the text, parse it we do -- chatter around it, tolerate we must.

    Brace-counting (not a greedy regex), so trailing prose after the JSON, never swallow it we do.
    None it returns when no parseable object there is.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        break  # This candidate failed -- the next '{' try we do.
        start = text.find("{", start + 1)
    return None


class QAPipeline:
    """Compose the modules, this does -- but own none of their internals, it must.

    Injected dependencies, all of them are -- so swap, mock and benchmark each part, we can.
    """

    # Below this many seconds left, no time for another ~CoT sampling pass there is -- stop, and the
    # votes we already have, keep them. (cot_v1 a sample ~4s costs; 5s of margin, comfortably safe it is.)
    _SC_MIN_MARGIN_S: float = 5.0

    # Numbers embedded in an option's text, this extracts -- commas stripped ("15,300" -> 15300).
    # The calculator-as-verifier (D-017) maps its result to an option through these.
    _NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

    def __init__(
        self,
        engine: LLMEngine,
        prompt_builder: PromptBuilder,
        classifier: QuestionClassifier | None = None,
        retriever=None,
        tools=None,
        self_consistency_n: int = 1,
        self_consistency_temperature: float = 0.7,
        latency_budget_s: float = 30.0,
        max_new_tokens: int | None = None,
        solver=None,
    ):
        self.engine = engine
        self.prompt_builder = prompt_builder
        self.classifier = classifier
        self.retriever = retriever
        self.tools = tools
        # A deterministic, type-specific MATHS solver: `solve_maths(question) -> (letter, evidence) | None`.
        # When present (only pipeline_maths injects it) it runs BEFORE the LLM and, on a confident single-
        # option hit for a computational type (finite-field roots, gcd, characteristic, sum/product,
        # reflection, triangle, %), answers from it and SKIPS the model -- correct by construction there.
        # It ABSTAINS (None) on every other type, so the LLM path is untouched: it can only ADD answers,
        # never override the model on the knowledge/abstract subset (validated: 0 regressions on the logs).
        self.solver = solver
        # The generation cap, optional it is. None (default) -> pass NOTHING to engine.generate, so the
        # engine's own default (256) stands -- EXACT current behaviour for every existing pipeline, no
        # change at all. Set it (e.g. 512) when a verbose strategy needs room to reach its 'Answer:' line:
        # at n=1 the 256 default TRUNCATED structured/CoT chains mid-reasoning (the no_answer failure the
        # offline routing experiment surfaced). With the game's 130s budget, 512 is comfortably affordable.
        self.max_new_tokens = max_new_tokens
        # Self-consistency: N>1 -> sample N CoT chains and majority-vote (Phase 5 / the Maths bet).
        # n=1 (the default) -> a single greedy pass: ZERO behaviour change for every existing pipeline.
        self.self_consistency_n = max(1, int(self_consistency_n))
        self.self_consistency_temperature = self_consistency_temperature
        self.latency_budget_s = latency_budget_s

    def _gen(self, prompt: str, **kwargs) -> str:
        """engine.generate, with `max_new_tokens` injected ONLY when this pipeline set one.

        None -> we pass nothing, so the engine default stands (byte-for-byte the old behaviour). A caller
        that already passed max_new_tokens explicitly, we never override.
        """
        if self.max_new_tokens is not None and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = self.max_new_tokens
        return self.engine.generate(prompt, **kwargs)

    def answer(self, question: Question) -> Prediction:
        """One question in, one Prediction out -- the whole system, in a single call this is.

        Seven stages in sequence, the pipeline runs:
          classify -> retrieve -> prompt -> generate -> tool (Phase 3) -> parse.
        Optional via DI, every collaborator is -- dormant they stay when None they are.
        Crash-safe the whole body is -- live play must always submit something, even on error.
        """
        # The latency guard and per-stage breakdown, created at the top they must be.
        guard = LatencyGuard(self.latency_budget_s)
        breakdown: dict[str, float] = {}

        # Retrieval state, initialised here -- dormant until a retriever exists it stays.
        retrieval_used: bool = False
        docs = None
        retrieved_doc_ids: list[str] = []
        retrieved_snippets: list[str] = []

        # Tool state, initialised here -- Phase 3 hook awaits, for now None it is.
        tool_used: str | None = None

        # Raw generation output -- empty string a safe default is.
        raw: str = ""

        # The parsed answer/confidence -- set by the parse stage (single pass) OR the vote (self-consistency).
        ans: str = ""
        conf: float = 0.0

        try:
            # --- Stage: classify ---
            # The question, enriched with topic/type/language -- only when a classifier present is.
            with stopwatch(breakdown, "classify"):
                if self.classifier:
                    question = self.classifier.classify(question)

            # --- Stage: solver short-circuit (Maths) ---
            # A deterministic type-specific solver, when injected, runs BEFORE the LLM. On a confident
            # single-option hit (a type it solves correctly by construction) we answer from it and SKIP
            # the model -- deterministic and instant. On abstain (None) the normal LLM path runs untouched.
            if self.solver is not None:
                with stopwatch(breakdown, "solver"):
                    try:
                        sol = self.solver(question)
                    except Exception:
                        sol = None
                if sol is not None:
                    letter, evidence = sol
                    return Prediction(
                        qid=question.qid,
                        answer=letter,
                        confidence=1.0,
                        raw_output=f"[math_solver] {evidence}",
                        model=getattr(self.engine, "name", ""),
                        prompt_strategy=getattr(self.prompt_builder, "strategy", ""),
                        retrieval_used=False,
                        retrieved_doc_ids=[],
                        retrieved_snippets=[],
                        tool_used="math_solver",
                        latency_s=guard.elapsed(),
                        tokens_in=0,
                        tokens_out=0,
                        error=None,
                    )

            # --- Stage: retrieve ---
            # Evidence, fetched only when a retriever exists AND retrieval needed it is.
            with stopwatch(breakdown, "retrieve"):
                if self.retriever and (
                    not self.classifier
                    or self.classifier.needs_retrieval(question)
                ):
                    # Raw document chunks, retrieved here they are -- never an LLM answer (D-008).
                    docs = self.retriever.retrieve(question)
                    retrieval_used = True
                    # The doc_id of each retrieved chunk, collected for the EvalRecord it is.
                    retrieved_doc_ids = [
                        doc.doc_id for doc in docs if hasattr(doc, "doc_id")
                    ]
                    # ...and the TEXT too (source-tagged) -- so a wrong-answer log shows whether the
                    # evidence carried the answer (retrieval miss) or the model ignored it (grounding miss).
                    retrieved_snippets = [
                        f"[{getattr(doc, 'source', '?')}#{getattr(doc, 'doc_id', i)}] {doc.text}"
                        for i, doc in enumerate(docs) if hasattr(doc, "text")
                    ]

            # --- Stage: prompt ---
            # The user-turn string, built from question and optional evidence context it is.
            with stopwatch(breakdown, "prompt"):
                prompt: str = self.prompt_builder.build(question, docs)

            # --- Stage: generate ---
            # n=1 (default): ONE pass, engine defaults (max_new_tokens/temperature not overridden).
            # n>1 (self-consistency): N SAMPLED chains, parsed and majority-voted -- for the hard
            # reasoning of Maths, more robust than one greedy pass it is (the occasional good chain,
            # the vote surfaces it; the confidence the VOTE SHARE becomes -- a real calibration signal).
            with stopwatch(breakdown, "generate"):
                if self.self_consistency_n > 1:
                    ans, conf, raw = self._self_consistency_answer(prompt, question, guard)
                else:
                    raw = self._gen(prompt)

            # --- Stage: tool ---
            # The calculator loop (D-013): only when tools present AND the classifier says arithmetic
            # is needed. The model a JSON call emits, run it we do, then with the number re-answer it does.
            # On any failure the plain-generate `raw` above stands -- crash-safe the maths path stays.
            with stopwatch(breakdown, "tool"):
                # Skip the tool when retrieval already fired (D-NEWS): the calculator re-answer prompt
                # carries only the bare MCQ -- NOT the retrieved docs -- so letting it override an
                # evidence-grounded answer would silently discard the web snippets (the News-clobber bug).
                # Skip too under self-consistency: the N CoT chains compute their arithmetic INLINE and
                # vote on it -- the tool's single context-free re-answer would fight the vote (and a wrong
                # SET-UP the calculator cannot save anyway -- our Maths logs, repeatedly this they showed).
                if (
                    self.self_consistency_n <= 1
                    and self.tools
                    and self.classifier
                    and not retrieval_used
                    and self.classifier.needs_calculator(question)
                ):
                    tool_raw, used = self._run_calculator_tool(question, guard)
                    if tool_raw is not None:
                        raw = tool_raw      # The tool-grounded answer, the plain one it replaces.
                        tool_used = used

            # --- Stage: parse ---
            # Single-pass mode parses the raw here; self-consistency already voted (ans/conf set above).
            with stopwatch(breakdown, "parse"):
                if self.self_consistency_n <= 1:
                    ans, conf = QAPipeline.parse_answer(raw, question)

        except Exception as e:
            # A crash, caught here it is -- the live game must never go silent.
            # The first available option letter, a safe fallback answer it makes.
            fallback_ans = sorted(question.options)[0] if question.options else ""
            return Prediction(
                qid=question.qid,
                answer=fallback_ans,
                confidence=0.0,
                raw_output=raw,
                model=getattr(self.engine, "name", ""),
                prompt_strategy=getattr(self.prompt_builder, "strategy", ""),
                retrieval_used=retrieval_used,
                retrieved_doc_ids=retrieved_doc_ids,
                retrieved_snippets=retrieved_snippets,
                tool_used=tool_used,
                latency_s=guard.elapsed(),
                tokens_in=getattr(self.engine, "last_tokens_in", 0),
                tokens_out=getattr(self.engine, "last_tokens_out", 0),
                error=str(e),
            )

        # The completed Prediction, assembled and returned it is.
        # Note: breakdown not stored on Prediction (schema is frozen); the runner re-times at its level.
        return Prediction(
            qid=question.qid,
            answer=ans,
            confidence=conf,
            raw_output=raw,
            model=getattr(self.engine, "name", ""),
            prompt_strategy=getattr(self.prompt_builder, "strategy", ""),
            retrieval_used=retrieval_used,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_snippets=retrieved_snippets,
            tool_used=tool_used,
            latency_s=guard.elapsed(),
            tokens_in=getattr(self.engine, "last_tokens_in", 0),
            tokens_out=getattr(self.engine, "last_tokens_out", 0),
            error=None,
        )

    def _self_consistency_answer(
        self, prompt: str, question: Question, guard: LatencyGuard
    ) -> tuple[str, float, str]:
        """N sampled chains of the SAME prompt -> a majority vote (the self-consistency technique).

        Each sample at `self_consistency_temperature` we draw (so the chains DIVERGE -- a vote of N
        identical greedy passes, meaningless it would be). Each we parse to a letter; over those
        letters `majority_vote` decides. Budget-aware: a further sample we skip once under
        `_SC_MIN_MARGIN_S` the guard falls -- but at LEAST one we always keep (the loop's first pass,
        the margin check never blocks).

        Returns: (answer, confidence, raw_output) -- the confidence the VOTE SHARE is (winners/total),
        and the raw the most-confident winning chain's text (a representative the log keeps).
        """
        candidates: list[Prediction] = []
        for _ in range(self.self_consistency_n):
            # Time for another ~CoT pass, is there? Below the margin -> the votes we have, settle for them.
            if candidates and guard.remaining() < self._SC_MIN_MARGIN_S:
                break
            sample = self._gen(
                prompt, temperature=self.self_consistency_temperature
            )
            a, c = QAPipeline.parse_answer(sample, question)
            candidates.append(
                Prediction(qid=question.qid, answer=a, confidence=c, raw_output=sample)
            )
        winner = majority_vote(candidates)  # >=1 candidate always -- the empty-list guard never trips.
        return winner.answer, winner.confidence, winner.raw_output

    def _run_calculator_tool(
        self, question: Question, guard: LatencyGuard
    ) -> tuple[str | None, str | None]:
        """The calculator as a VERIFIER (D-017): single-turn, the model an arithmetic call emits, run it
        we do -- then the number to an OPTION we map, and the chain's answer OVERRIDE we do ONLY on a
        unique, in-tolerance match.

        Why a match-gate (the run-#8 clobber, thus avoided): re-answering blindly from the result HURT
        concept/stats Qs (a t-test where the answer is a CONCLUSION using a number, not the number). So:
        override only when the result IS one of the options (mean=65 == the option "65"); else the cot_v2
        reasoning we KEEP. Default = keep the chain; override only on confidence (run #10 motivated this).

        Returns:
            ("Answer: X" + trace, "calculator") -- a valid call made AND the result UNIQUELY matched option X.
            (None, None)                        -- otherwise, so the cot_v2 answer untouched it stays.
        Crash-safe entirely: a bad JSON, a forbidden expression, a calc error -- all swallowed, None returned.
        """
        calc = self.tools.get("calculator") if isinstance(self.tools, dict) else None
        if calc is None:
            return None, None
        # One more generation the tool costs -- below the wall too close, then skip it we do.
        if guard.remaining() < 4.0:
            return None, None

        try:
            mcq = _render_mcq_block(question)

            # The tool offer. ONLY-JSON or ONLY-a-letter, the model we ask for. The FULL computation in the
            # expression we want (every raw number listed) -- the model's own mental arithmetic NOT trusted
            # it is (run #10: ten scores it summed to 620, not 650). The adding, the tool must do.
            offer = (
                f"{mcq}\n\n"
                "A calculator tool you may use for arithmetic. To call it, reply with ONLY this JSON "
                "and nothing else:\n"
                '{"name": "calculator", "arguments": {"expression": "<arithmetic expression>"}}\n'
                "Pure arithmetic the expression must be (digits and + - * / ** % ( ) only). Put the FULL "
                "computation in it -- list EVERY original number and let the tool add; pre-compute nothing.\n"
                "No calculation needed? Then reply with ONLY the letter (A, B, C, or D)."
            )
            first = self._gen(offer)
            call = _extract_first_json(first)
            if not call or call.get("name") != "calculator":
                return None, None  # No tool call -- the cot_v2 answer, let it stand.

            args = call.get("arguments") or {}
            expr = str(args.get("expression", "")).strip()
            if not expr:
                return None, None
            result = calc(expr)  # safe-AST calculate(); on anything non-arithmetic, it raises.

            # The number -> an OPTION, map it we do. Override ONLY on a unique, in-tolerance match.
            letter = self._option_for_value(question, result)
            if letter is None:
                return None, None  # No single option the result matched -> the cot_v2 reasoning we keep.
            # "Answer: X" the parser reads at conf 1.0; the trace (no A-D letter in it), for the log we keep.
            return f"Answer: {letter}\n[calculator: {expr} = {result}]", "calculator"
        except Exception:
            # Any slip -- the maths path must never crash the turn; the cot_v2 answer, fall back to it we do.
            return None, None

    @staticmethod
    def _numbers_in(text: str) -> list[float]:
        """Every number embedded in an option's text -> floats, commas stripped ("15,300" -> 15300.0)."""
        out: list[float] = []
        for m in QAPipeline._NUM_RE.findall(text or ""):
            try:
                out.append(float(m.replace(",", "")))
            except ValueError:
                pass  # A lone "-" or a stray match -- skip it we do.
        return out

    @staticmethod
    def _option_for_value(question: Question, value: float) -> str | None:
        """A computed number -> the option letter it UNIQUELY matches (else None) -- the verifier's gate.

        Override the chain ONLY when exactly ONE option carries a number within ~0.5% of the result AND no
        OTHER option is within that tolerance. "The answer IS a number" (mean=65 -> option "65") it catches;
        "a conclusion that merely uses a number" (a t-test "df=17") it rejects (the result equals no option
        value) -> the cot_v2 answer kept it is. Densely-spaced numeric options (e.g. 6100/6200/6300/6400),
        the uniqueness check still separates -- the nearest within tol, the rivals outside it.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        # Tolerance: 0.5% of the magnitude, but >= 1.0 (so integers like 65 match exactly-ish).
        tol = max(1.0, 0.005 * abs(v))
        best_letter: str | None = None
        best_dist = float("inf")
        second_dist = float("inf")
        for letter in sorted(question.options):
            nums = QAPipeline._numbers_in(question.options[letter])
            if not nums:
                continue
            d = min(abs(v - n) for n in nums)  # this option's NEAREST number to the result.
            if d < best_dist:
                second_dist = best_dist
                best_dist = d
                best_letter = letter
            elif d < second_dist:
                second_dist = d
        # Within tol AND no rival within tol -> a confident, unique match it is.
        if best_letter is not None and best_dist <= tol and second_dist > tol:
            return best_letter
        return None

    @staticmethod
    def parse_answer(raw_output: str, question: Question) -> tuple[str, float]:
        """The model's text -> (answer, confidence). Robust to chatter, this must be.

        From the raw generation, a single valid option letter we extract.
        A rich set of patterns -- 'Answer: B', '(C)', 'Option D', bare 'a' -- handled they all are.
        Restricted to the letters actually present in question.options, the result is.
        For an open question with no options, the cleaned raw text and low confidence returned are.
        """
        # The valid letters, from the question's options determined they are.
        valid_letters: set[str] = set(question.options.keys()) if question.options else set()

        # An open question with no options -- return cleaned text and low confidence we do.
        if not valid_letters:
            return raw_output.strip(), 0.3

        # The text to search, stripped of leading/trailing whitespace it is.
        text = raw_output.strip()

        # Helper: a candidate letter, validated against valid_letters it is.
        def _valid(letter: str) -> str | None:
            # Uppercased and checked against the allowed set, the letter is.
            up = letter.upper()
            return up if up in valid_letters else None

        # --- Pattern group 1: Explicit answer markers (highest confidence) ---
        # Patterns like "Answer: B", "Answer:B", "The answer is C", "answer is: D" caught here they are.
        # The old regex a SPACE before the colon demanded -- so "Answer: B" (cot's own format!) it MISSED,
        # and the prose fell to pattern-7, the article "a" as "A" grabbing (P2-bug). Two clean branches now:
        # a colon straight after "answer" (space optional), OR " is" then an optional colon. Bare "answer a
        # question", never the article it snags -- a separator (':' or 'is'), each branch demands.
        _EXPLICIT = re.compile(
            r"\b(?:the\s+)?answer\b(?:\s*:\s*|\s+is\s*:?\s*)([A-Da-d])\b",
            re.IGNORECASE,
        )
        m = _EXPLICIT.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 2: Parenthesised letter -- "(B)" or "(b)" ---
        # A single letter in parentheses, a strong answer signal it is.
        _PAREN = re.compile(r"\(([A-Da-d])\)", re.IGNORECASE)
        m = _PAREN.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 3: "Option X" or "option X" ---
        # The word 'Option' followed by a letter, matched here it is.
        _OPTION_WORD = re.compile(r"\boption\s+([A-Da-d])\b", re.IGNORECASE)
        m = _OPTION_WORD.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 4: Letter followed by closing paren -- "B)" or "b)" ---
        # Common in enumerated list continuations, this form is.
        _LETTER_PAREN = re.compile(r"\b([A-Da-d])\)", re.IGNORECASE)
        m = _LETTER_PAREN.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 0.9

        # --- Pattern group 5: Letter followed by option text (e.g. "B Rome") ---
        # The model echoes "B Rome" or "C: Paris" -- the letter we take, the text we ignore.
        if valid_letters and question.options:
            for letter_key in sorted(valid_letters):
                # The option text for this letter, escaped for regex it is.
                opt_text = re.escape(question.options[letter_key][:20].strip())
                _ECHO = re.compile(
                    rf"\b({re.escape(letter_key)})[):\s]+{opt_text}",
                    re.IGNORECASE,
                )
                m = _ECHO.search(text)
                if m:
                    letter = _valid(m.group(1))
                    if letter:
                        return letter, 0.9

        # --- Pattern group 6: Standalone letter on its own line or at start of text ---
        # A bare "A" or "b" as the sole token on a line -- the cleanest single-letter output it is.
        _STANDALONE_LINE = re.compile(
            r"(?:^|\n)\s*([A-Da-d])\s*(?:\n|$)",
            re.IGNORECASE,
        )
        m = _STANDALONE_LINE.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 7: Any isolated letter bounded by non-alpha chars ---
        # The first letter in the text that is surrounded by word boundaries, grabbed we do.
        # Lower confidence: context around it, we cannot verify.
        _ISOLATED = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", re.IGNORECASE)
        matches = _ISOLATED.findall(text)
        # Only valid candidates, kept they are.
        valid_matches = [_valid(ltr) for ltr in matches if _valid(ltr)]
        if valid_matches:
            # The first valid match, returned with medium confidence -- ambiguous it may be.
            return valid_matches[0], 0.5

        # --- Fallback: no letter found -- the first available option, submitted we must ---
        # Live play must always answer something; silent submission, allowed it is not.
        fallback = sorted(valid_letters)[0]
        return fallback, 0.0
