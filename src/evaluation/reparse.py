"""Re-parse saved records with the CURRENT parser -- the true numbers, no model re-run it costs.

Every record its model `raw_output` keeps. So fix `parse_answer` you do, and the SAVED generations
re-judge you can -- the [P2-bug] fix (cot's "Answer: X" now read, not the article "a" as "A" grabbed)
verified WITHOUT a GPU, this lets you. The model, re-run we never -- only the parsing, replayed it is.

Usage (run where the logs live -- on Colab, that is):
    # As a script (one or more run dirs):
    python -m evaluation.reparse experiments/runs/prompt_eng
    # In a notebook:
    from evaluation.reparse import reparse_runs
    reparse_runs(["experiments/runs/prompt_eng"])

Only OFFLINE records (a known `gold_answer`) re-judged are -- live rows, no gold they carry, skipped they are.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from schemas import Question, QuestionType
from agent.pipeline import QAPipeline


def _load_dev(dev_path: str | None) -> dict[str, Question]:
    """The dev set, by qid indexed -- old logs lack the `options`, so from here borrow them we do."""
    out: dict[str, Question] = {}
    if dev_path and Path(dev_path).exists():
        for line in Path(dev_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[str(d["qid"])] = Question(
                qid=str(d["qid"]),
                text=d.get("text", ""),
                options=d.get("options", {}) or {},
                qtype=QuestionType.MCQ,
                gold=d.get("gold"),
            )
    return out


def _question_from_record(rec: dict, dev_by_qid: dict[str, Question]) -> Question:
    """A logged row -> a minimal Question, enough for `parse_answer` (its valid letters it needs).

    Options: from the record (new logs carry them), else the dev set by qid, else a bare A-D fallback
    (the dev set 4-option MCQ is -- so the valid-letter set, this safely gives).
    """
    opts = rec.get("options") or {}
    if not opts:
        dev_q = dev_by_qid.get(str(rec.get("qid")))
        opts = dev_q.options if dev_q else {"A": "", "B": "", "C": "", "D": ""}
    return Question(
        qid=str(rec.get("qid")),
        text=rec.get("question_text", ""),
        options=opts,
        qtype=QuestionType.MCQ,
        gold=rec.get("gold_answer"),
    )


def reparse_runs(run_dirs: list[str], dev_path: str | None = "data/dev_questions.jsonl") -> dict:
    """Each run dir's records.jsonl, with the CURRENT parser re-judge -- old vs new accuracy, per strategy."""
    dev = _load_dev(dev_path)
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # strategy -> [old_correct, new_correct, n]
    flipped: list[tuple] = []  # the rows the fix changed.

    for rd in run_dirs:
        path = Path(rd) / "records.jsonl"
        if not path.exists():
            print(f"skip (no records.jsonl): {rd}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold = rec.get("gold_answer")
            if gold is None:  # Live rows -- no gold, no accuracy. Skip them we do.
                continue
            q = _question_from_record(rec, dev)
            new_ans, _ = QAPipeline.parse_answer(rec.get("raw_output", ""), q)
            strat = rec.get("prompt_strategy", "?")
            old_pred = str(rec.get("predicted_answer", "")).upper()
            new_pred = str(new_ans).upper()
            goldU = str(gold).upper()
            s = stats[strat]
            s[0] += int(old_pred == goldU)
            s[1] += int(new_pred == goldU)
            s[2] += 1
            if old_pred != new_pred:
                flipped.append((strat, rec.get("qid"), old_pred, new_pred, goldU, rec.get("raw_output", "")[:90]))

    print(f"{'strategy':16} {'old acc':>9} {'new acc':>9} {'n':>4}")
    for strat, (oc, nc, n) in sorted(stats.items()):
        if n:
            print(f"{strat:16} {oc / n:>8.1%} {nc / n:>8.1%} {n:>4}")

    print(f"\nrows the fix flipped: {len(flipped)}")
    for strat, qid, old, new, gold, raw in flipped[:40]:
        ok = "OK " if new == gold else "still-wrong"
        print(f"  [{strat}] qid={qid}: {old} -> {new} (gold {gold}) [{ok}]  raw={raw!r}")

    return dict(stats)


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["experiments/runs/prompt_eng"]
    reparse_runs(dirs)
