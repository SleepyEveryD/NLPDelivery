"""Run harness -- two modes, one logging format, this gives.

Two ways a run drives the SAME pipeline, there are:
  - OFFLINE ("our own test"): over the hand-crafted dev set we loop; the gold known,
    so `correct` locally we compute.  -> `BenchmarkRunner`
  - LIVE ("the real test"): the actual game API drives the loop; the truth only after
    submitting revealed, so `correct` from `AnswerResult` we take.  -> `LiveRunner`

Both write the SAME `EvalRecord` JSONL, so `metrics.py` reads either without caring which.
Pick the mode with `run_session(...)`, or call a runner directly you may.
One run = one config; the config itself, into meta.json we write.
"""
from __future__ import annotations

import copy
import datetime
import subprocess
import time
from typing import Callable, Optional

from agent.pipeline import QAPipeline
from config import RunConfig
from evaluation.logger import ExperimentLogger
from schemas import EvalRecord, Prediction, Question, RunMode


# --------------------------------------------------------------------------- #
# Shared helpers -- both runners lean on these, duplication we avoid.
# --------------------------------------------------------------------------- #

def _collect_meta(config: RunConfig) -> dict:
    """The run's meta dict: config + best-effort git/hardware/timestamp probes.

    Crash-safe every probe is -- a missing tool, None it leaves, never an exception.
    """
    # The config's plain dict, the foundation it is (mode + game settings, it already carries).
    meta = config.to_dict()

    # The current git commit, best-effort we fetch.
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        git_commit = None

    # The CUDA device name, best-effort we probe.
    hardware = None
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            hardware = torch.cuda.get_device_name(0)
    except Exception:
        hardware = None

    meta["git_commit"] = git_commit
    meta["hardware"] = hardware
    meta["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return meta


def _qtype_str(question: Question) -> str:
    """qtype: a str the EvalRecord wants; an enum the Question may hold -- bridge it we do."""
    try:
        return question.qtype.value
    except AttributeError:
        return str(question.qtype)


def _build_record(
    run_id: str,
    question: Question,
    pred: Prediction,
    correct: Optional[bool],
    reached_level: Optional[int] = None,
    current_level: Optional[int] = None,
) -> EvalRecord:
    """One question + its prediction + a known-or-unknown correctness -> one EvalRecord.

    `correct` the CALLER decides: from gold (offline) or from AnswerResult (live), or None.
    `reached_level`/`current_level` LIVE only -- the server's `AnswerResult` they carry; None offline.
    """
    return EvalRecord(
        run_id=run_id,
        timestamp=EvalRecord.now(),
        qid=question.qid,
        question_text=question.text,
        qtype=_qtype_str(question),
        topic=question.topic,
        level=question.level,
        language=question.language,
        model=pred.model,
        prompt_strategy=pred.prompt_strategy,
        retrieval_used=pred.retrieval_used,
        retrieved_doc_ids=pred.retrieved_doc_ids,
        retrieved_snippets=getattr(pred, "retrieved_snippets", []),
        tool_used=pred.tool_used,
        predicted_answer=pred.answer,
        gold_answer=question.gold,
        correct=correct,
        confidence=pred.confidence,
        latency_s=pred.latency_s,
        # No breakdown the Prediction carries; empty dict we pass.
        latency_breakdown={},
        tokens_in=pred.tokens_in,
        tokens_out=pred.tokens_out,
        raw_output=pred.raw_output,
        error=pred.error,
        options=question.options,   # The choices shown, logged for a full replay they are (the letter the model picked, by its text we can read).
        reached_level=reached_level,   # The server's climb telemetry (LIVE) -- the leaderboard metric, this is.
        current_level=current_level,
    )


def _grade_offline(pred: Prediction, question: Question) -> Optional[bool]:
    """Offline correctness: the predicted letter against the dev-set gold, compared it is.

    No gold the question has -> None we return (an open question, or a live-style row).
    """
    if question.gold is None:
        return None
    # Case-insensitive letter comparison, fair it is.
    return pred.answer.strip().upper() == question.gold.strip().upper()


# --------------------------------------------------------------------------- #
# OFFLINE -- "our own test": the dev set, looped and graded locally.
# --------------------------------------------------------------------------- #

class BenchmarkRunner:
    """Over a dev set the pipeline run, and each EvalRecord log -- the reproducible offline loop, it is."""

    def __init__(self, pipeline: QAPipeline, config: RunConfig, log_root: str = "experiments/runs"):
        self.pipeline = pipeline
        self.config = config
        self.log_root = log_root

    def run(self, questions: list) -> str:
        """Over a dataset run the pipeline and log. The run path, return it does."""
        meta = _collect_meta(self.config)
        # Belt-and-braces: whatever the config said, offline this runner is.
        meta["mode"] = RunMode.OFFLINE.value

        logger = ExperimentLogger(self.log_root, self.config.run_id, meta=meta)
        try:
            for i, question in enumerate(questions, start=1):
                # Progress: one line per question, print we do.
                print(f"[{i}/{len(questions)}] qid={question.qid} ...")

                pred = self.pipeline.answer(question)
                correct = _grade_offline(pred, question)
                record = _build_record(self.config.run_id, question, pred, correct)
                logger.log(record)
        finally:
            # Always close the logger; a crash mid-loop, data we must not lose.
            logger.close()

        return str(logger.dir)


# --------------------------------------------------------------------------- #
# LIVE -- "the real test": the game API drives the loop; correct comes post-submit.
# --------------------------------------------------------------------------- #

class LiveRunner:
    """Against the REAL game one session play, and each EvalRecord log -- the live loop, it is.

    The game API the questions feeds and the answers grades; our pipeline the strategy is.
    A logged-in `GameClient` (src/game/client.py) injected it must be -- the HTTP it owns, not us.
    """

    def __init__(
        self,
        pipeline: QAPipeline,
        config: RunConfig,
        game_client,
        log_root: str = "experiments/runs",
    ):
        self.pipeline = pipeline
        self.config = config
        self.game_client = game_client
        self.log_root = log_root

    def run(self) -> str:
        """One full live game play, every turn logged. The run path, return it does."""
        meta = _collect_meta(self.config)
        meta["mode"] = RunMode.LIVE.value

        # The 30s wall is real now -- a network margin below it, aim we do (D-014).
        # The pipeline's budget, to the live target we set; deterministic for this session it stays.
        self.pipeline.latency_budget_s = self.config.game.aim_seconds

        logger = ExperimentLogger(self.log_root, self.config.run_id, meta=meta)

        # The pipeline's Prediction, per qid we stash -- so on_result the full row can build.
        preds: dict[str, Prediction] = {}
        counter = {"n": 0}  # A tiny mutable, the closures share it.

        def answer_fn(q: Question) -> str:
            # Here our pipeline decides; the full Prediction, for logging we keep.
            pred = self.pipeline.answer(q)
            preds[q.qid] = pred
            return pred.answer

        def on_result(q: Question, letter: str, result, time_left) -> None:
            # The pipeline's Prediction for this qid, recover it we do.
            pred = preds.get(q.qid)
            if pred is None:
                # Defensive: a Prediction missing should never be -- but lose the row we will not.
                return
            # The truth, the server now reveals -- None if it withholds it (e.g. timed out).
            correct = getattr(result, "correct", None)
            # The server's level telemetry, capture it we now do -- the leaderboard by reached_level
            # scores us (D-014), and our pipeline knows it not. From the AnswerResult, lifted it is
            # (None it stays when the server withholds it).
            reached_level = getattr(result, "reached_level", None)
            current_level = getattr(result, "current_level", None)
            record = _build_record(
                self.config.run_id, q, pred, correct,
                reached_level=reached_level, current_level=current_level,
            )
            logger.log(record)

            counter["n"] += 1
            timed_out = getattr(result, "timed_out", False)
            print(
                f"[{counter['n']}] qid={q.qid} lvl={q.level} reached={reached_level} "
                f"-> {letter} | correct={correct} | timed_out={timed_out} | "
                f"latency={pred.latency_s:.1f}s (left was {time_left})"
            )

        try:
            self.game_client.play(
                competition_id=self.config.game.competition_id,
                answer_fn=answer_fn,
                on_result=on_result,
                mode=self.config.game.game_mode,
            )
        finally:
            logger.close()

        return str(logger.dir)


# --------------------------------------------------------------------------- #
# Dispatch -- the ONE entry point that picks the mode for you.
# --------------------------------------------------------------------------- #

def run_session(
    pipeline: QAPipeline,
    config: RunConfig,
    *,
    questions: Optional[list] = None,
    game_client=None,
    log_root: str = "experiments/runs",
) -> str:
    """On `config.mode`, the right runner pick -- the single switch between our two tests, this is.

    OFFLINE -> BenchmarkRunner over `questions` (or, if None, loaded from `config.dataset_path`).
    LIVE    -> LiveRunner over a logged-in `game_client`.
    The run-directory path, either way return it does.
    """
    mode = RunMode.normalize(config.mode)

    if mode is RunMode.OFFLINE:
        # No questions handed in -> from the dev set on disk, load them we do.
        if questions is None:
            from evaluation.dataset import load_questions
            questions = load_questions(config.dataset_path)
        return BenchmarkRunner(pipeline, config, log_root=log_root).run(questions)

    # mode is LIVE
    if game_client is None:
        raise ValueError(
            "Live mode a logged-in GameClient requires -- pass game_client=... you must "
            "(see src/game/client.py::GameClient). Offline you wanted? Set config.mode='offline'."
        )
    return LiveRunner(pipeline, config, game_client, log_root=log_root).run()


def run_all_competitions(
    pipeline: QAPipeline,
    config: RunConfig,
    game_client,
    *,
    competition_ids=(0, 1, 2, 3, 4, 5),
    log_root: str = "experiments/runs",
    pause_s: float = 8.0,
    on_competition=None,
    pipeline_for: Optional[Callable[[int], QAPipeline]] = None,
) -> list[tuple[int, Optional[str]]]:
    """All six competitions, one LIVE game each, in sequence play -- the whole sweep, this is.

    Each competition its OWN run dir gets (run_id `live_comp{id}`), so `metrics.load_runs` per-competition
    reads them. The caller's `config` we never mutate -- a deep copy per game, with `mode='live'` + the
    competition id set, we drive.

    `pipeline_for`: a per-competition pipeline picker, optional it is. Given a `cid`, the pipeline for THAT
    game it returns -- so Maths (comp 3) a `cot_v1` + no-retrieval pipeline can use, while the others the
    shared `few_shot_v1` + RAG one keep. None (the default) -> the single `pipeline` for every competition
    (back-compatible). The reliable LIVE topic signal is the competition id, not the unset `question.topic`.

    Polite to the proof-of-concept server we stay (the assignment asks it): `pause_s` seconds BETWEEN games
    we sleep. Resilient too -- one competition's failure (a RateLimitError, say) the rest it never sinks;
    `(id, None)` for the failed one we record, and onward we go.

    Returns: a list of `(competition_id, run_path-or-None)`, in play order.
    """
    ids = list(competition_ids)
    results: list[tuple[int, Optional[str]]] = []
    for i, cid in enumerate(ids):
        cfg = copy.deepcopy(config)         # The caller's config, untouched it stays.
        cfg.mode = "live"
        cfg.game.competition_id = cid
        cfg.run_id = f"live_comp{cid}"
        if on_competition:
            on_competition(cid)
        # The pipeline for THIS competition -- the per-cid picker when given, else the shared one.
        pipe = pipeline_for(cid) if pipeline_for else pipeline
        try:
            run_path = LiveRunner(pipe, cfg, game_client, log_root=log_root).run()
            results.append((cid, run_path))
        except Exception as e:  # One game falls -- the sweep survives, the failure we note.
            print(f"  [comp {cid}] FAILED -- {type(e).__name__}: {e}")
            results.append((cid, None))
        if pause_s and i < len(ids) - 1:    # Between games (not after the last), polite we pause.
            time.sleep(pause_s)
    return results
