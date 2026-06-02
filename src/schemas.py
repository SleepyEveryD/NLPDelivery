"""Core data contracts shared across all modules.

The single source of truth for the shapes that flow through the pipeline, this file is.
Change a shape here you must, when an interface evolves -- so drift between modules, prevent we do.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionType(str, Enum):
    # The kinds of questions, these are.
    MCQ = "mcq"          # Multiple choice, one correct letter it has.
    OPEN = "open"        # Free-text short answer, this is.
    UNKNOWN = "unknown"


# Friendly aliases, so the notebook need not memorise the canonical words.
# Module-level frozensets these are -- inside a str-Enum, plain-set attributes
# misread as members would be, so out here they live.
_OFFLINE_ALIASES = frozenset({"offline", "test", "dev", "our", "ours", "own", "local", "our_test", "our test", "our own test"})
_LIVE_ALIASES = frozenset({"live", "real", "game", "online", "server", "real_test", "real test", "realtest"})


class RunMode(str, Enum):
    """The two ways a run can be driven, these are.

    OFFLINE -- our OWN test: the hand-crafted dev set, where the gold is known and
    accuracy locally computed it is.
    LIVE    -- the REAL test: the actual game API, where the truth only after submitting
    revealed it is (correct from AnswerResult comes).
    """
    OFFLINE = "offline"  # our own dev-set test (gold known up front).
    LIVE = "live"        # the real game (gold known only post-submit).

    @classmethod
    def normalize(cls, value) -> "RunMode":
        """A loose string ('our own test', 'real') into a canonical RunMode, this turns.

        Already a RunMode it may be -- then untouched it passes. Unknown it is -- loudly we fail.
        """
        # Already canonical, the value is -- return it unchanged we do.
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        # The exact-alias sets, first we consult.
        if key in _OFFLINE_ALIASES:
            return cls.OFFLINE
        if key in _LIVE_ALIASES:
            return cls.LIVE
        # A keyword fallback -- "real"/"live"/"game" beat "test", so "real test" → LIVE it is.
        if any(w in key for w in ("real", "live", "game", "online", "server")):
            return cls.LIVE
        if any(w in key for w in ("our", "offline", "dev")) or "test" in key:
            return cls.OFFLINE
        # Neither matched -- a clear error, raise we must.
        raise ValueError(
            f"Unknown run mode {value!r}. Use 'offline' (our own test) or 'live' (real test)."
        )


@dataclass
class Question:
    """A single quiz question, as received from the game or a dev set.

    From the game API or an offline dataset, populated this is.
    """
    qid: str
    text: str
    options: dict[str, str] = field(default_factory=dict)  # {"A": "...", ...}; empty for open, it is.
    option_ids: dict[str, int] = field(default_factory=dict)  # {"A": 101, ...}; the server's Option.id per letter.
    qtype: QuestionType = QuestionType.MCQ
    level: Optional[int] = None        # Difficulty rung 1..15, the game may tell us.
    topic: Optional[str] = None        # Filled by the classifier later, it is.
    language: Optional[str] = None     # "en" / "it", detected or known.
    gold: Optional[str] = None         # The truth -- known only for dev sets, None in live play it is.


@dataclass
class RetrievedDoc:
    # A raw chunk of evidence, retrieved this was -- never an LLM answer, it must be.
    doc_id: str
    text: str
    source: str          # URL or corpus name, the origin is.
    score: float = 0.0


@dataclass
class Prediction:
    """What a single model produced for one question.

    The atomic unit of a prediction, this is. Over many of these, the ensemble votes.
    """
    qid: str
    answer: str                        # "A".."D" for MCQ, or text for open.
    confidence: float = 0.0            # 0..1, self-reported or derived it is.
    raw_output: str = ""               # The unparsed generation -- for debugging, kept it is.
    model: str = ""
    prompt_strategy: str = ""
    retrieval_used: bool = False
    retrieved_doc_ids: list[str] = field(default_factory=list)
    # The retrieved chunks' TEXT (not just ids) -- so a wrong answer's log shows whether the evidence
    # actually carried the answer (a retrieval miss) or the model ignored it (a grounding miss). News
    # diagnosis needs this; the doc_ids alone (`guardian:0`..) reveal nothing about the content.
    retrieved_snippets: list[str] = field(default_factory=list)
    tool_used: Optional[str] = None    # e.g. "calculator"; None if no tool called, it is.
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None


@dataclass
class EvalRecord:
    """One row of the benchmark log -- everything the rubric asks us to track, it holds.

    Append-only to JSONL these go. Reproduce any experiment from them, we can.
    """
    run_id: str
    timestamp: float
    qid: str
    question_text: str
    qtype: str
    topic: Optional[str]
    level: Optional[int]
    language: Optional[str]
    model: str
    prompt_strategy: str
    retrieval_used: bool
    retrieved_doc_ids: list[str]
    tool_used: Optional[str]
    predicted_answer: str
    gold_answer: Optional[str]
    correct: Optional[bool]            # None when the gold is unknown (live game), it is.
    confidence: float
    latency_s: float
    latency_breakdown: dict[str, float]
    tokens_in: int
    tokens_out: int
    raw_output: str
    error: Optional[str] = None
    options: dict[str, str] = field(default_factory=dict)  # {"A": "...", ...}; the choices shown, for a full replay kept they are. Empty for open / old logs, it is.
    # The server's level telemetry -- LIVE play only. Our pipeline knows it NOT; from `AnswerResult` it comes.
    # The REAL leaderboard metric, `reached_level` is (by it we are scored, D-014) -- so capture it we must.
    # None for offline rows / old logs / when the server withholds it, these stay.
    reached_level: Optional[int] = None    # How high the RUN climbed -- across the game, its max the score is.
    current_level: Optional[int] = None    # The level of THIS turn, as the server counted it.
    # The retrieved chunks' TEXT (source-tagged) -- empty for old logs / no-retrieval rows. The "was the
    # answer even in the evidence?" question, this answers when a News turn goes wrong.
    retrieved_snippets: list[str] = field(default_factory=list)

    @staticmethod
    def now() -> float:
        # The current wall-clock time, this gives.
        return time.time()
