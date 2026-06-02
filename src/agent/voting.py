"""Ensemble voting. Many Predictions into one, this combines.

Majority vote, or confidence-weighted -- more reliable answers, the rubric hopes for.
The SAME primitive two callers serve it does:
  * self-consistency -- N sampled CoT chains of ONE model, voted (the Maths bet, `pipeline.py`).
  * the Phase-5 ensemble -- one Prediction per DIFFERENT model, voted (latency-permitting).
Only if the 30s budget the extra forward passes allows, run the multi-pass forms we will.
"""
from __future__ import annotations

from collections import Counter

from schemas import Prediction


def majority_vote(predictions: list[Prediction]) -> Prediction:
    """Many Predictions -> one. The most-voted answer wins; ties, by mean confidence broken they are.

    RAW it stays -- a NEW Prediction from the winners it builds, so the inputs untouched remain.

    The returned Prediction:
      * answer     -- the winning letter/text.
      * confidence -- the VOTE SHARE (winners / total). A REAL calibration signal, this is: unlike a
                      single greedy pass (where every clean letter reads 1.0), a 2/3 vote a genuine
                      'two chains agreed, one dissented' uncertainty it carries.
      * the most-confident winning sample as the FACE -- its raw_output/model/prompt_strategy carried,
        so a representative generation the log keeps.
      * tool_used / retrieval_used -- OR'd across the winners (any winner used it -> True).

    Note: latency_s / tokens_* the representative's are -- a single member's, not the sum. The
    self-consistency caller OVERWRITES them with the run's real totals (it owns the latency guard);
    the Phase-5 caller, refine them it may. Empty input -> a ValueError, loudly we raise.
    """
    if not predictions:
        raise ValueError("majority_vote of an empty list -- at least one Prediction, give it must.")

    # Votes per answer, tallied they are.
    counts = Counter(p.answer for p in predictions)
    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]

    # A tie -> by the MEAN confidence of each tied answer's members, break it we do.
    if len(tied) > 1:
        def _mean_conf(ans: str) -> float:
            confs = [p.confidence for p in predictions if p.answer == ans]
            return sum(confs) / len(confs) if confs else 0.0
        winner_answer = max(tied, key=_mean_conf)
    else:
        winner_answer = tied[0]

    # The winning members; the most-confident of them, the representative it is.
    winners = [p for p in predictions if p.answer == winner_answer]
    rep = max(winners, key=lambda p: p.confidence)

    vote_share = top / len(predictions)

    return Prediction(
        qid=rep.qid,
        answer=winner_answer,
        confidence=vote_share,
        raw_output=rep.raw_output,
        model=rep.model,
        prompt_strategy=rep.prompt_strategy,
        retrieval_used=any(p.retrieval_used for p in winners),
        retrieved_doc_ids=rep.retrieved_doc_ids,
        tool_used=next((p.tool_used for p in winners if p.tool_used), None),
        latency_s=rep.latency_s,
        tokens_in=rep.tokens_in,
        tokens_out=rep.tokens_out,
        error=rep.error,
    )
