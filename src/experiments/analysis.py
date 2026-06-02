"""Analysis & visualisation for the adaptive-routing experiment -- the evidence, these turn into figures.

From the one `records.jsonl` everything derives (re-analyse without re-running the model, we can):
  * strategy-comparison table       -- overall accuracy / latency / tokens / reasoning length per condition.
  * category x condition accuracy   -- the heatmap that shows WHERE each prompt helps or hurts.
  * latency-vs-accuracy scatter     -- the cost of reasoning, plotted.
  * routing-accuracy report         -- did the classifier label the reasoning shape correctly?
  * failure taxonomy                -- overthinking / skipped-case / boundary / drift, counted per category.

matplotlib only (the repo's one plotting dep) -- no seaborn, so portable to Colab it stays.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# The eight categories, in a fixed display order -- so every table/heatmap reads the same way.
CATEGORY_ORDER = [
    "arithmetic", "temporal_reasoning", "interval_counting", "discrete_enumeration",
    "factual_qa", "commonsense", "logical_reasoning", "multi_hop",
]
CONDITION_ORDER = ["A_universal", "B_generic_cot", "C_structured", "D_adaptive"]


def load_records(records_path: str = "experiments/adaptive_routing/records.jsonl") -> pd.DataFrame:
    """The experiment JSONL -> a tidy DataFrame. One row per (condition, question), it holds."""
    rows: list[dict] = []
    with open(records_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty and "correct" in df.columns:
        # JSON may carry bool or null; numeric for means it must be (null -> NaN, excluded later).
        df["correct"] = df["correct"].astype("float")
    return df


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def strategy_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition: overall accuracy, mean latency, mean tokens-out, mean reasoning length.

    The headline table -- "does adaptive (D) beat the universal prompt (A) and the always-on chains
    (B, C), and at what latency/token cost?" -- in one glance it answers.
    """
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby("condition")
        .agg(
            accuracy=("correct", "mean"),
            n=("correct", "count"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", lambda s: float(np.nanpercentile(s, 95))),
            mean_tokens_out=("tokens_out", "mean"),
            mean_reasoning_lines=("reasoning_lines", "mean"),
        )
        .reset_index()
    )
    # Stable, meaningful row order.
    g["__o"] = g["condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)}).fillna(99)
    return g.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def category_accuracy_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy with rows = true_category, columns = condition. The heatmap's data, this is."""
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index="true_category", columns="condition", values="correct", aggfunc="mean"
    )
    pivot = pivot.reindex(index=[c for c in CATEGORY_ORDER if c in pivot.index])
    pivot = pivot.reindex(columns=[c for c in CONDITION_ORDER if c in pivot.columns])
    return pivot


def best_strategy_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """For each TRUE category, the FIXED strategy that scored highest -- the oracle the router chases.

    Only the fixed-strategy rows we consider (each carries its own strategy column even in the adaptive
    arm, but here we read the three single-strategy conditions A/B/C plus the strategy actually run).
    Reported: best fixed strategy, its accuracy, and what the adaptive arm achieved -- the gap, it shows.
    """
    if df.empty:
        return pd.DataFrame()
    # Per (category, strategy-actually-used) accuracy, over ALL rows (every arm logs its strategy).
    per = (
        df.groupby(["true_category", "strategy"])["correct"].mean().reset_index()
    )
    rows = []
    for cat in [c for c in CATEGORY_ORDER if c in set(per["true_category"])]:
        sub = per[per["true_category"] == cat]
        best = sub.loc[sub["correct"].idxmax()]
        adaptive_acc = df[(df["true_category"] == cat) & (df["condition"] == "D_adaptive")]["correct"].mean()
        rows.append({
            "category": cat,
            "best_strategy": best["strategy"],
            "best_accuracy": float(best["correct"]),
            "adaptive_accuracy": float(adaptive_acc) if pd.notna(adaptive_acc) else None,
        })
    return pd.DataFrame(rows)


def routing_report(df: pd.DataFrame) -> dict:
    """How often the ReasoningClassifier's label matched the hand-labelled truth, this measures.

    On the adaptive arm we read it (one verdict per question there); the routed_category column,
    against true_category we compare. A confusion table too, for the error analysis it gives.
    """
    if df.empty:
        return {"overall_routing_accuracy": None, "confusion": pd.DataFrame()}
    arm = df[df["condition"] == "D_adaptive"]
    if arm.empty:
        arm = df.drop_duplicates("qid")
    match = (arm["routed_category"] == arm["true_category"]).mean()
    confusion = pd.crosstab(arm["true_category"], arm["routed_category"])
    confusion = confusion.reindex(
        index=[c for c in CATEGORY_ORDER if c in confusion.index],
        columns=[c for c in CATEGORY_ORDER if c in confusion.columns],
        fill_value=0,
    )
    return {"overall_routing_accuracy": float(match), "confusion": confusion}


# --------------------------------------------------------------------------- #
# Failure taxonomy -- WHY a prompt got it wrong, heuristically labelled from the raw output.
# --------------------------------------------------------------------------- #

_NO_ANSWER_RE = re.compile(r"answer\s*:", re.IGNORECASE)


def classify_failure(row: pd.Series) -> str | None:
    """One wrong row -> a failure-mode label (None when correct). Heuristic, transparent, it is.

    Modes (the rubric's list, operationalised):
      overthinking       -- a recall/commonsense Q answered with a long chain (reasoning where none helps).
      boundary_error     -- a counting/interval Q wrong despite enumerating (off-by-one at an endpoint).
      skipped_case       -- a counting/logic Q wrong with a SHORT chain (cases never enumerated).
      arithmetic_drift   -- an arithmetic Q wrong despite showing work (a slip mid-computation).
      no_answer_parsed   -- the generation never reached an 'Answer:' line (truncation / format miss).
      hallucinated/other -- a wrong factual/multi-hop answer (a confident wrong fact).
    """
    if row.get("correct") == 1.0 or row.get("correct") is True:
        return None
    cat = row.get("true_category", "")
    raw = str(row.get("raw_output", ""))
    lines = row.get("reasoning_lines", 0) or 0
    strategy = row.get("strategy", "")
    conf = row.get("confidence", None)

    # A genuine parse failure -- NOT merely "the raw lacks an 'Answer:' string". The bare-letter
    # strategies (direct_answer / few_shot_v1) are SUPPOSED to omit it, and they parse fine, so the
    # old "no 'answer:' -> no_answer_parsed" rule mislabelled EVERY wrong few-shot row. The honest
    # signal is the parser's own fallback: `parse_answer` returns confidence 0.0 ONLY when no letter
    # could be extracted at all. We also flag a CoT strategy that never wrote an 'Answer:' line
    # (truncation) -- there the missing marker IS the failure.
    _COT = {"generic_cot", "structured_enumeration_cot", "checklist_cot", "cot_v1", "cot_v2", "cot_maths_v1"}
    if conf is not None and float(conf) == 0.0:
        return "no_answer_parsed"
    if strategy in _COT and not _NO_ANSWER_RE.search(raw):
        return "no_answer_parsed"
    if cat in ("factual_qa", "commonsense"):
        return "overthinking" if lines >= 3 else "hallucinated/other"
    if cat in ("interval_counting", "discrete_enumeration", "temporal_reasoning"):
        return "boundary_error" if lines >= 3 else "skipped_case"
    if cat == "arithmetic":
        return "arithmetic_drift" if lines >= 2 else "skipped_case"
    if cat in ("logical_reasoning", "multi_hop"):
        return "skipped_case" if lines < 3 else "hallucinated/other"
    return "hallucinated/other"


def failure_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """A count of failure modes per condition -- the error story, tabulated."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["failure_mode"] = work.apply(classify_failure, axis=1)
    fails = work[work["failure_mode"].notna()]
    if fails.empty:
        return pd.DataFrame()
    tab = pd.crosstab(fails["condition"], fails["failure_mode"])
    tab = tab.reindex(index=[c for c in CONDITION_ORDER if c in tab.index], fill_value=0)
    return tab


# --------------------------------------------------------------------------- #
# Figures -- saved to PNG so the notebook OR a headless run both produce them.
# --------------------------------------------------------------------------- #

def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_category_heatmap(df: pd.DataFrame, out_path: str = "experiments/adaptive_routing/fig_category_heatmap.png"):
    """Accuracy heatmap: rows = category, cols = condition. WHERE each prompt wins, it shows."""
    import matplotlib.pyplot as plt

    mat = category_accuracy_matrix(df)
    if mat.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    data = mat.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black")
    ax.set_title("Accuracy by reasoning category x condition")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    _ensure_dir(Path(out_path))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_latency_accuracy(df: pd.DataFrame, out_path: str = "experiments/adaptive_routing/fig_latency_accuracy.png"):
    """Per-condition mean latency (x) vs accuracy (y) -- the cost of reasoning, scattered."""
    import matplotlib.pyplot as plt

    tab = strategy_comparison_table(df)
    if tab.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(tab["mean_latency_s"], tab["accuracy"], s=80, zorder=3)
    for _, r in tab.iterrows():
        ax.annotate(r["condition"], (r["mean_latency_s"], r["accuracy"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(tab["accuracy"].max(), ls="--", lw=0.7, color="grey", zorder=1)
    ax.set_xlabel("mean latency per question (s)")
    ax.set_ylabel("overall accuracy")
    ax.set_title("Latency vs accuracy by condition")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_dir(Path(out_path))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_condition_accuracy_bars(df: pd.DataFrame, out_path: str = "experiments/adaptive_routing/fig_condition_accuracy.png"):
    """Overall accuracy per condition -- the headline bar chart."""
    import matplotlib.pyplot as plt

    tab = strategy_comparison_table(df)
    if tab.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#888", "#5b9bd5", "#ed7d31", "#70ad47"][: len(tab)]
    ax.bar(tab["condition"], tab["accuracy"], color=colors)
    for i, v in enumerate(tab["accuracy"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("overall accuracy")
    ax.set_title("Overall accuracy by condition")
    fig.tight_layout()
    _ensure_dir(Path(out_path))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def run_full_analysis(records_path: str = "experiments/adaptive_routing/records.jsonl") -> dict:
    """Load -> all tables + all figures -> a dict the notebook (or a script) can print/inspect."""
    df = load_records(records_path)
    out = {
        "comparison": strategy_comparison_table(df),
        "category_matrix": category_accuracy_matrix(df),
        "best_per_category": best_strategy_per_category(df),
        "routing": routing_report(df),
        "failures": failure_taxonomy(df),
        "figures": {
            "category_heatmap": plot_category_heatmap(df),
            "latency_accuracy": plot_latency_accuracy(df),
            "condition_accuracy": plot_condition_accuracy_bars(df),
        },
    }
    return out
