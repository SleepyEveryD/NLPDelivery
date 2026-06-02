"""Metrics & breakdowns. The investigation's evidence, these compute.

Accuracy overall and by topic/level/type, latency percentiles, overconfidence (confidence vs correctness).
From the JSONL logs all of it derives -- so re-analyse without re-running models, we can.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# The budget wall the game enforces -- 30 seconds it is.
_LATENCY_BUDGET_S: float = 30.0


def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    """All run dirs into one tidy DataFrame, this merges.

    Each dir must contain a records.jsonl (one JSON object per line).
    Missing or empty files, skipped with a warning they are.
    """
    # The collected rows, here they accumulate.
    rows: list[dict] = []

    for run_dir in run_dirs:
        # The JSONL path, from the dir we build.
        jsonl_path = Path(run_dir) / "records.jsonl"

        if not jsonl_path.exists():
            # Missing file -- warn and skip, crash we must not.
            warnings.warn(f"records.jsonl not found, skipping we are: {jsonl_path}")
            continue

        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    # Blank lines, safely we skip.
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # One JSON object per line, parse we do.
                        record = json.loads(line)
                        rows.append(record)
                    except json.JSONDecodeError as exc:
                        # A corrupt line -- warn and skip, all else we keep.
                        warnings.warn(
                            f"JSON parse error at {jsonl_path}:{line_no} -- {exc}"
                        )
        except OSError as exc:
            # Unreadable file -- warn and skip.
            warnings.warn(f"Could not read {jsonl_path}, skipping: {exc}")
            continue

    if not rows:
        # No data found -- an empty DataFrame with no columns we return.
        return pd.DataFrame()

    # One tidy DataFrame from all rows, build we do.
    df = pd.DataFrame(rows)

    # run_id already in each record (EvalRecord.run_id), present it should be.
    # If somehow absent, a placeholder we add so downstream code never breaks.
    if "run_id" not in df.columns:
        df["run_id"] = None

    return df


def accuracy_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Grouped accuracy by the given column, this computes.

    Rows where correct is None/NaN, excluded they are (live play rows, those are).
    Returns a DataFrame with columns [column, 'accuracy', 'n'].
    """
    if df.empty or column not in df.columns or "correct" not in df.columns:
        # Nothing to compute -- an empty result we return.
        return pd.DataFrame(columns=[column, "accuracy", "n"])

    # Only rows where correct is known, keep we do.
    known = df[df["correct"].notna()].copy()

    if known.empty:
        # All correct values None -- no accuracy to report.
        return pd.DataFrame(columns=[column, "accuracy", "n"])

    # correct may be stored as bool or int (0/1) in JSON; numeric it must be for mean.
    known["correct"] = known["correct"].astype(float)

    # Group by column; accuracy = mean of correct, n = count of rows.
    grouped = (
        known.groupby(column, dropna=False)["correct"]
        .agg(accuracy="mean", n="count")
        .reset_index()
    )

    # Column order: [column, accuracy, n] -- tidy and predictable it is.
    result = grouped[[column, "accuracy", "n"]].copy()

    return result


def latency_summary(df: pd.DataFrame) -> dict:
    """Median, p95, max, mean, and budget violations -- the 30s story, this tells.

    Returns a dict with keys:
        median_s, p95_s, max_s, mean_s, over_budget (count), budget_violation_rate.
    An empty df, safe defaults it yields.
    """
    if df.empty or "latency_s" not in df.columns:
        # Empty input -- zeroed-out summary we return.
        return {
            "median_s": None,
            "p95_s": None,
            "max_s": None,
            "mean_s": None,
            "over_budget": 0,
            "budget_violation_rate": 0.0,
        }

    # NaN rows from latency_s, drop we do -- broken records, those are.
    latencies = df["latency_s"].dropna()

    if latencies.empty:
        # All NaN -- same safe defaults.
        return {
            "median_s": None,
            "p95_s": None,
            "max_s": None,
            "mean_s": None,
            "over_budget": 0,
            "budget_violation_rate": 0.0,
        }

    # The core statistics, from numpy/pandas we draw.
    median_s: float = float(latencies.median())
    p95_s: float = float(latencies.quantile(0.95))
    max_s: float = float(latencies.max())
    mean_s: float = float(latencies.mean())

    # Budget violations: rows exceeding the 30s wall, count them we do.
    over_budget: int = int((latencies > _LATENCY_BUDGET_S).sum())
    budget_violation_rate: float = over_budget / len(latencies)

    return {
        "median_s": median_s,
        "p95_s": p95_s,
        "max_s": max_s,
        "mean_s": mean_s,
        "over_budget": over_budget,
        "budget_violation_rate": budget_violation_rate,
    }
