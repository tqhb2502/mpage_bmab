"""Generate balanced tables and figures for all five experiment setups.

The script reads run artifacts directly from the experiment result tree and
does not depend on a pre-existing summary.csv. It writes only tables and
figures; it intentionally does not generate narrative analysis text.

Run from the repository root:

    MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
        mpage_bmab/experiments/analyze_all_setups_overview.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
_PROJECT_ROOT = _PKG_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mpage_bmab.experiments.aggregate import aggregate  # noqa: E402


METHODS = ["full", "full_old", "mpage_orig", "dense_reward", "hybrid_reward"]
TASKS = ["bi_tsp", "tri_tsp", "bi_cvrp", "bi_kp"]
BUDGETS = [25, 50, 100, 200]
METRICS = [
    "aubc",
    "hv_final",
    "valid_yield_per_100_calls",
    "invalid_null_proxy_rate",
]
PRIMARY_SUMMARY_METRICS = [
    "aubc",
    "hv_final",
    "valid_yield_per_100_calls",
]
DIAGNOSTIC_SUMMARY_METRICS = [
    "invalid_null_proxy_rate",
]

HIGHER_IS_BETTER = {
    "aubc": True,
    "hv_final": True,
    "valid_yield_per_100_calls": True,
    "invalid_null_proxy_rate": False,
}

TASK_LABELS = {
    "bi_tsp": "Bi-TSP",
    "tri_tsp": "Tri-TSP",
    "bi_cvrp": "Bi-CVRP",
    "bi_kp": "Bi-KP",
}

METHOD_LABELS = {
    "full": "Current full",
    "full_old": "Previous full",
    "mpage_orig": "MPaGE-orig",
    "dense_reward": "Dense reward",
    "hybrid_reward": "Hybrid reward",
}

METRIC_LABELS = {
    "aubc": "AUBC",
    "hv_final": "Final HV",
    "valid_yield_per_100_calls": "Valid yield",
    "invalid_null_proxy_rate": "Invalid/null proxy",
}

METRIC_SHORT = {
    "aubc": "AUBC",
    "hv_final": "Final HV",
    "valid_yield_per_100_calls": "Valid yield",
    "invalid_null_proxy_rate": "Invalid rate",
}

COLORS = {
    "full": "#2474A6",
    "full_old": "#8C6BB1",
    "mpage_orig": "#D55E00",
    "dense_reward": "#6C757D",
    "hybrid_reward": "#2CA25F",
}

MARKERS = {
    "full": "o",
    "full_old": "s",
    "mpage_orig": "^",
    "dense_reward": "D",
    "hybrid_reward": "P",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def fmt(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return mean(vals) if vals else math.nan


def std_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return stdev(vals) if len(vals) > 1 else 0.0


def sample_counts(run_dir: Path) -> tuple[int, int, int]:
    total = valid = null = 0
    for sample_file in (run_dir / "samples").glob("samples_*.json"):
        try:
            data = json.loads(sample_file.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        total += len(data)
        for sample in data:
            if not isinstance(sample, dict) or sample.get("score") is None:
                null += 1
            else:
                valid += 1
    return total, valid, null


def bmab_generation_attempts(run_dir: Path) -> tuple[int, int]:
    try:
        history = json.loads((run_dir / "budget_history.json").read_text())
    except Exception:
        return 0, 0
    generation_calls = 0
    non_generation_calls = 0
    for entry in history if isinstance(history, list) else []:
        label = str(entry.get("label", ""))
        if label == "init" or label.startswith("mutate") or label.startswith("crossover"):
            generation_calls += 1
        else:
            non_generation_calls += 1
    return generation_calls, non_generation_calls


def enrich_rows(rows: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for row in rows:
        if row["ablation"] not in METHODS:
            continue
        d = dict(row)
        run_dir = Path(d["run_dir"])
        sample_count, valid_count, null_count = sample_counts(run_dir)
        total_budget = float(d.get("total_budget", 0.0) or 0.0)
        recorded_null_rate = null_count / sample_count if sample_count else math.nan
        valid_yield = valid_count / total_budget * 100.0 if total_budget else math.nan

        gen_calls, non_gen_calls = bmab_generation_attempts(run_dir)
        if d["ablation"] == "mpage_orig":
            invalid_proxy = recorded_null_rate
            unregistered = ""
            gen_calls_out = ""
            non_gen_calls_out = ""
        else:
            unregistered_count = max(0, gen_calls - valid_count)
            invalid_proxy = unregistered_count / gen_calls if gen_calls else math.nan
            unregistered = unregistered_count
            gen_calls_out = gen_calls
            non_gen_calls_out = non_gen_calls

        d.update({
            "sample_count": sample_count,
            "valid_count": valid_count,
            "null_score_count": null_count,
            "recorded_null_rate": recorded_null_rate,
            "valid_yield_per_100_calls": valid_yield,
            "generation_attempts": gen_calls_out,
            "non_generation_calls": non_gen_calls_out,
            "unregistered_generation_count": unregistered,
            "invalid_null_proxy_rate": invalid_proxy,
        })
        enriched.append(d)
    return enriched


def build_cell_means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["ablation"], row["task"], int(row["budget"]))].append(row)

    out: list[dict] = []
    for method in METHODS:
        for task in TASKS:
            for budget in BUDGETS:
                rs = groups.get((method, task, budget), [])
                rec: dict = {
                    "method": method,
                    "task": task,
                    "budget": budget,
                    "n": len(rs),
                }
                for metric in [
                    "aubc",
                    "hv_final",
                    "pareto_size",
                    "valid_yield_per_100_calls",
                    "invalid_null_proxy_rate",
                    "recorded_null_rate",
                    "valid_count",
                    "null_score_count",
                    "sample_count",
                ]:
                    vals = [
                        float(r[metric])
                        for r in rs
                        if r.get(metric) not in ("", None)
                        and not math.isnan(float(r[metric]))
                    ]
                    rec[f"{metric}_mean"] = mean_or_nan(vals)
                    rec[f"{metric}_std"] = std_or_zero(vals)
                out.append(rec)
    return out


def rank_values(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    ordered = sorted(
        values.items(),
        key=lambda kv: kv[1],
        reverse=higher_is_better,
    )
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and math.isclose(ordered[j][1], ordered[i][1], rel_tol=1e-12, abs_tol=1e-12):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def normalized_scores(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    finite = {k: v for k, v in values.items() if not math.isnan(v)}
    if not finite:
        return {m: math.nan for m in values}
    if higher_is_better:
        best = max(finite.values())
        return {
            method: (value / best * 100.0 if best else math.nan)
            for method, value in values.items()
        }
    lo = min(finite.values())
    hi = max(finite.values())
    if math.isclose(hi, lo):
        return {method: 100.0 for method in values}
    return {
        method: (hi - value) / (hi - lo) * 100.0
        for method, value in values.items()
    }


def build_rank_tables(cell_means: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by = {
        (row["method"], row["task"], int(row["budget"])): row
        for row in cell_means
    }
    rank_rows: list[dict] = []
    for task in TASKS:
        for budget in BUDGETS:
            for metric in METRICS:
                values = {
                    method: float(by[(method, task, budget)][f"{metric}_mean"])
                    for method in METHODS
                }
                ranks = rank_values(values, HIGHER_IS_BETTER[metric])
                normalized = normalized_scores(values, HIGHER_IS_BETTER[metric])
                for method in METHODS:
                    rank_rows.append({
                        "method": method,
                        "task": task,
                        "budget": budget,
                        "metric": metric,
                        "value": values[method],
                        "rank": ranks[method],
                        "normalized_score": normalized[method],
                        "is_best": ranks[method] == 1.0,
                        "is_top2": ranks[method] <= 2.0,
                    })

    overall: list[dict] = []
    for method in METHODS:
        for metric in METRICS:
            rows = [r for r in rank_rows if r["method"] == method and r["metric"] == metric]
            ranks = [float(r["rank"]) for r in rows]
            scores = [float(r["normalized_score"]) for r in rows]
            overall.append({
                "method": method,
                "metric": metric,
                "n_cells": len(rows),
                "mean_normalized_score": mean_or_nan(scores),
                "median_normalized_score": median(scores),
                "mean_rank": mean_or_nan(ranks),
                "median_rank": median(ranks),
                "best_cells": sum(1 for r in rows if r["is_best"]),
                "top2_cells": sum(1 for r in rows if r["is_top2"]),
                "worst_cells": sum(1 for r in rows if math.isclose(float(r["rank"]), float(len(METHODS)))),
            })

    dashboard: list[dict] = []
    for method in METHODS:
        rec: dict = {"method": method}
        method_rows = [r for r in overall if r["method"] == method]
        for r in method_rows:
            metric = r["metric"]
            rec[f"{metric}_normalized_score"] = r["mean_normalized_score"]
            rec[f"{metric}_mean_rank"] = r["mean_rank"]
            rec[f"{metric}_best_cells"] = r["best_cells"]
            rec[f"{metric}_top2_cells"] = r["top2_cells"]
        rec["performance_score_mean"] = mean_or_nan([
            rec["aubc_normalized_score"],
            rec["hv_final_normalized_score"],
        ])
        rec["diagnostic_score_mean"] = mean_or_nan([
            rec["valid_yield_per_100_calls_normalized_score"],
            rec["invalid_null_proxy_rate_normalized_score"],
        ])
        rec["all_metric_score_mean"] = mean_or_nan([
            rec["aubc_normalized_score"],
            rec["hv_final_normalized_score"],
            rec["valid_yield_per_100_calls_normalized_score"],
            rec["invalid_null_proxy_rate_normalized_score"],
        ])
        rec["performance_rank_mean"] = mean_or_nan([
            rec["aubc_mean_rank"],
            rec["hv_final_mean_rank"],
        ])
        rec["all_metric_rank_mean"] = mean_or_nan([
            rec["aubc_mean_rank"],
            rec["hv_final_mean_rank"],
            rec["valid_yield_per_100_calls_mean_rank"],
            rec["invalid_null_proxy_rate_mean_rank"],
        ])
        dashboard.append(rec)
    return rank_rows, overall, dashboard


def build_best_count_table(rank_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for method in METHODS:
        for task in TASKS:
            for metric in METRICS:
                rows = [
                    r for r in rank_rows
                    if r["method"] == method and r["task"] == task and r["metric"] == metric
                ]
                out.append({
                    "method": method,
                    "task": task,
                    "metric": metric,
                    "best_cells": sum(1 for r in rows if r["is_best"]),
                    "top2_cells": sum(1 for r in rows if r["is_top2"]),
                    "mean_rank": mean_or_nan(float(r["rank"]) for r in rows),
                    "mean_normalized_score": mean_or_nan(float(r["normalized_score"]) for r in rows),
                })
    return out


def build_pairwise_matrices(cell_means: list[dict], rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_cell_mean = {
        (row["method"], row["task"], int(row["budget"])): row
        for row in cell_means
    }
    cell_matrix_rows: list[dict] = []
    for metric in METRICS:
        for row_method in METHODS:
            rec = {"metric": metric, "row_method": row_method}
            for col_method in METHODS:
                wins = ties = losses = 0
                for task in TASKS:
                    for budget in BUDGETS:
                        a = float(by_cell_mean[(row_method, task, budget)][f"{metric}_mean"])
                        b = float(by_cell_mean[(col_method, task, budget)][f"{metric}_mean"])
                        if not HIGHER_IS_BETTER[metric]:
                            a, b = -a, -b
                        if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                            ties += 1
                        elif a > b:
                            wins += 1
                        else:
                            losses += 1
                rec[f"{col_method}_wins"] = wins
                rec[f"{col_method}_ties"] = ties
                rec[f"{col_method}_losses"] = losses
                rec[col_method] = f"{wins}/{wins + ties + losses}"
            cell_matrix_rows.append(rec)

    idx = {
        (r["ablation"], r["task"], int(r["budget"]), int(r["seed"])): r
        for r in rows
    }
    seed_matrix_rows: list[dict] = []
    for metric in METRICS:
        for row_method in METHODS:
            rec = {"metric": metric, "row_method": row_method}
            for col_method in METHODS:
                wins = ties = losses = 0
                for task in TASKS:
                    for budget in BUDGETS:
                        for seed in [2025, 2026, 2027, 2028, 2029]:
                            a = float(idx[(row_method, task, budget, seed)][metric])
                            b = float(idx[(col_method, task, budget, seed)][metric])
                            if not HIGHER_IS_BETTER[metric]:
                                a, b = -a, -b
                            if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                                ties += 1
                            elif a > b:
                                wins += 1
                            else:
                                losses += 1
                rec[f"{col_method}_wins"] = wins
                rec[f"{col_method}_ties"] = ties
                rec[f"{col_method}_losses"] = losses
                rec[col_method] = f"{wins}/{wins + ties + losses}"
            seed_matrix_rows.append(rec)
    return cell_matrix_rows, seed_matrix_rows


def read_curve(run_dir: Path) -> list[tuple[float, float]]:
    try:
        data = json.loads((run_dir / "budget_curve.json").read_text())
    except Exception:
        return []
    total_budget = None
    try:
        aubc = json.loads((run_dir / "aubc.json").read_text())
        total_budget = float(aubc.get("total_budget", 0.0) or 0.0)
    except Exception:
        total_budget = None
    points: list[tuple[float, float]] = []
    for point in data if isinstance(data, list) else []:
        consumed = float(
            point.get(
                "budget_consumed",
                point.get("consumed_budget", point.get("budget", 0.0)),
            )
            or 0.0
        )
        hv = float(point.get("hv", 0.0) or 0.0)
        denom = total_budget or consumed or 1.0
        x = max(0.0, min(1.0, consumed / denom))
        points.append((x, hv))
    if not points:
        return []
    points.sort()
    compact: list[tuple[float, float]] = []
    for x, hv in points:
        if compact and abs(compact[-1][0] - x) < 1e-12:
            compact[-1] = (x, hv)
        else:
            compact.append((x, hv))
    if compact[0][0] > 0.0:
        # Match BMABProfiler.aubc(): AUBC integrates from an initial
        # (budget=0, HV=0) point, not from the first observed HV.
        compact.insert(0, (0.0, 0.0))
    return compact


def mean_curve(rows: list[dict], method: str, task: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
    curves = [
        read_curve(Path(r["run_dir"]))
        for r in rows
        if r["ablation"] == method and r["task"] == task and int(r["budget"]) == budget
    ]
    curves = [curve for curve in curves if curve]
    xs = np.linspace(0.0, 1.0, 101)
    if not curves:
        return xs, np.full_like(xs, np.nan)
    ys = []
    for curve in curves:
        cx = np.array([p[0] for p in curve], dtype=float)
        cy = np.array([p[1] for p in curve], dtype=float)
        ys.append(np.interp(xs, cx, cy))
    return xs, np.vstack(ys).mean(axis=0)


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_metric_lines(cell_means: list[dict], metric: str, out: Path) -> None:
    by = {(row["method"], row["task"], int(row["budget"])): row for row in cell_means}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    for ax, task in zip(axes.ravel(), TASKS):
        for method in METHODS:
            vals = [by[(method, task, budget)][f"{metric}_mean"] for budget in BUDGETS]
            ax.plot(
                BUDGETS,
                vals,
                marker="o",
                linewidth=2.0,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Budget")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(BUDGETS)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.045))
    fig.suptitle(f"Mean {METRIC_LABELS[metric]} by Task and Budget", y=1.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def metric_offsets(metrics: list[str], width: float) -> list[float]:
    center = (len(metrics) - 1) / 2.0
    return [(i - center) * width for i in range(len(metrics))]


def plot_overall_normalized_bars(overall: list[dict], out: Path, metrics: list[str], title: str) -> None:
    by = {(row["method"], row["metric"]): row for row in overall}
    x = np.arange(len(METHODS))
    width = min(0.22, 0.82 / max(1, len(metrics)))
    fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    metric_colors = {
        "aubc": "#2474A6",
        "hv_final": "#E69F00",
        "valid_yield_per_100_calls": "#2CA25F",
        "invalid_null_proxy_rate": "#7570B3",
    }
    for offset, metric in zip(metric_offsets(metrics, width), metrics):
        vals = [by[(method, metric)]["mean_normalized_score"] for method in METHODS]
        ax.bar(x + offset, vals, width, label=METRIC_SHORT[metric], color=metric_colors[metric])
    ax.set_ylim(0, 105)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Mean normalized score across task-budget cells")
    ax.set_title(title)
    ax.legend(ncol=len(metrics), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_mean_rank_bars(overall: list[dict], out: Path, metrics: list[str], title: str) -> None:
    by = {(row["method"], row["metric"]): row for row in overall}
    x = np.arange(len(METHODS))
    width = min(0.22, 0.82 / max(1, len(metrics)))
    fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    metric_colors = {
        "aubc": "#2474A6",
        "hv_final": "#E69F00",
        "valid_yield_per_100_calls": "#2CA25F",
        "invalid_null_proxy_rate": "#7570B3",
    }
    for offset, metric in zip(metric_offsets(metrics, width), metrics):
        vals = [by[(method, metric)]["mean_rank"] for method in METHODS]
        ax.bar(x + offset, vals, width, label=METRIC_SHORT[metric], color=metric_colors[metric])
    ax.set_ylim(1, len(METHODS) + 0.4)
    ax.invert_yaxis()
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Mean rank across task-budget cells (lower is better)")
    ax.set_title(title)
    ax.legend(ncol=len(metrics), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_best_counts(overall: list[dict], out: Path, metrics: list[str], title: str) -> None:
    by = {(row["method"], row["metric"]): row for row in overall}
    x = np.arange(len(METHODS))
    width = min(0.22, 0.82 / max(1, len(metrics)))
    fig, ax = plt.subplots(figsize=(11, 5.4), constrained_layout=True)
    metric_colors = {
        "aubc": "#2474A6",
        "hv_final": "#E69F00",
        "valid_yield_per_100_calls": "#2CA25F",
        "invalid_null_proxy_rate": "#7570B3",
    }
    for offset, metric in zip(metric_offsets(metrics, width), metrics):
        vals = [by[(method, metric)]["best_cells"] for method in METHODS]
        ax.bar(x + offset, vals, width, label=METRIC_SHORT[metric], color=metric_colors[metric])
    ax.set_ylim(0, 16)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Number of best task-budget cells")
    ax.set_title(title)
    ax.legend(ncol=len(metrics), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_heatmap(rank_rows: list[dict], metric: str, out: Path) -> None:
    cell_labels = [f"{TASK_LABELS[t]}\nB={b}" for t in TASKS for b in BUDGETS]
    values = np.zeros((len(METHODS), len(cell_labels)), dtype=float)
    for i, method in enumerate(METHODS):
        for j, (task, budget) in enumerate((t, b) for t in TASKS for b in BUDGETS):
            row = next(
                r for r in rank_rows
                if r["method"] == method
                and r["task"] == task
                and int(r["budget"]) == budget
                and r["metric"] == metric
            )
            values[i, j] = float(row["normalized_score"])

    fig, ax = plt.subplots(figsize=(14.5, 4.2), constrained_layout=True)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cell_labels)), cell_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    ax.set_title(f"Normalized {METRIC_LABELS[metric]} Score by Task-Budget Cell")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center",
                    color="white" if values[i, j] < 58 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized score (best cell = 100)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_matrix(matrix_rows: list[dict], metric: str, out: Path, title_suffix: str) -> None:
    rows = [r for r in matrix_rows if r["metric"] == metric]
    values = np.zeros((len(METHODS), len(METHODS)), dtype=float)
    labels = [["" for _ in METHODS] for _ in METHODS]
    for i, row_method in enumerate(METHODS):
        row = next(r for r in rows if r["row_method"] == row_method)
        for j, col_method in enumerate(METHODS):
            values[i, j] = float(row[f"{col_method}_wins"])
            labels[i][j] = row[col_method]
    vmax = 80 if "Seed" in title_suffix else 16
    fig, ax = plt.subplots(figsize=(7.8, 6.4), constrained_layout=True)
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS], rotation=35, ha="right")
    ax.set_yticks(range(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    ax.set_xlabel("Column setup")
    ax.set_ylabel("Row setup")
    ax.set_title(f"{METRIC_LABELS[metric]} Pairwise Wins ({title_suffix})")
    for i in range(len(METHODS)):
        for j in range(len(METHODS)):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    color="white" if values[i, j] > vmax * 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row setup wins")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_budget_curves(rows: list[dict], budget: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    for ax, task in zip(axes.ravel(), TASKS):
        for method in METHODS:
            xs, ys = mean_curve(rows, method, task, budget)
            ax.plot(xs, ys, linewidth=2.0, color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Normalized budget")
        ax.set_ylabel("Outer HV")
        ax.set_xlim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.045))
    fig.suptitle(f"Mean Budget Curves Across All Setups at B={budget}", y=1.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_bmab_zoom_budget_curves(rows: list[dict], budget: int, out: Path) -> None:
    methods = ["full", "full_old", "dense_reward", "hybrid_reward"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    for ax, task in zip(axes.ravel(), TASKS):
        task_values: list[float] = []
        for method in methods:
            xs, ys = mean_curve(rows, method, task, budget)
            ax.plot(xs, ys, linewidth=2.0, color=COLORS[method], label=METHOD_LABELS[method])
            task_values.extend(float(v) for v in ys if not math.isnan(float(v)))
        if task_values:
            lo, hi = min(task_values), max(task_values)
            pad = max((hi - lo) * 0.08, hi * 0.01 if hi else 1.0)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Normalized budget")
        ax.set_ylabel("Outer HV")
        ax.set_xlim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.045))
    fig.suptitle(f"Zoomed BMAB-Style Budget Curves at B={budget}", y=1.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(dashboard: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    xs: list[float] = []
    ys: list[float] = []
    for row in dashboard:
        method = row["method"]
        xs.append(float(row["aubc_normalized_score"]))
        ys.append(float(row["hv_final_normalized_score"]))
        ax.scatter(
            row["aubc_normalized_score"],
            row["hv_final_normalized_score"],
            s=155,
            marker=MARKERS[method],
            facecolors="none",
            edgecolors=COLORS[method],
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Mean normalized AUBC score")
    ax.set_ylabel("Mean normalized final-HV score")
    ax.set_title("Performance Trade-off: AUBC vs Final HV")
    ax.set_xlim(max(0, min(xs) - 4), min(105, max(xs) + 4))
    ax.set_ylim(max(0, min(ys) - 4), min(105, max(ys) + 4))
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_scatter(dashboard: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    xs: list[float] = []
    ys: list[float] = []
    for row in dashboard:
        method = row["method"]
        xs.append(float(row["valid_yield_per_100_calls_normalized_score"]))
        ys.append(float(row["invalid_null_proxy_rate_normalized_score"]))
        ax.scatter(
            row["valid_yield_per_100_calls_normalized_score"],
            row["invalid_null_proxy_rate_normalized_score"],
            s=155,
            marker=MARKERS[method],
            facecolors="none",
            edgecolors=COLORS[method],
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Mean normalized valid-yield score")
    ax.set_ylabel("Mean normalized invalid-rate score")
    ax.set_title("Validity Diagnostics Trade-off")
    ax.set_xlim(max(0, min(xs) - 4), min(105, max(xs) + 4))
    ax.set_ylim(max(0, min(ys) - 4), min(105, max(ys) + 4))
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def write_latex_table(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def write_latex_tables(out_dir: Path, overall: list[dict], dashboard: list[dict], cell_matrices: list[dict]) -> None:
    ensure_dir(out_dir)
    by = {(r["method"], r["metric"]): r for r in overall}

    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Setup & AUBC score & AUBC rank & Final-HV score & Final-HV rank & Valid score & Invalid score & Perf. score & All score \\\\",
        "\\midrule",
    ]
    for row in dashboard:
        method = row["method"]
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{fmt(row['aubc_normalized_score'])} & {fmt(row['aubc_mean_rank'], 2)} & "
            f"{fmt(row['hv_final_normalized_score'])} & {fmt(row['hv_final_mean_rank'], 2)} & "
            f"{fmt(row['valid_yield_per_100_calls_normalized_score'])} & "
            f"{fmt(row['invalid_null_proxy_rate_normalized_score'])} & "
            f"{fmt(row['performance_score_mean'])} & {fmt(row['all_metric_score_mean'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex_table(out_dir / "overall_normalized_scores.tex", lines)

    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Setup & AUBC best & AUBC top-2 & Final-HV best & Final-HV top-2 & Valid best & Valid top-2 & Invalid best & Invalid top-2 \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{by[(method, 'aubc')]['best_cells']} & {by[(method, 'aubc')]['top2_cells']} & "
            f"{by[(method, 'hv_final')]['best_cells']} & {by[(method, 'hv_final')]['top2_cells']} & "
            f"{by[(method, 'valid_yield_per_100_calls')]['best_cells']} & {by[(method, 'valid_yield_per_100_calls')]['top2_cells']} & "
            f"{by[(method, 'invalid_null_proxy_rate')]['best_cells']} & {by[(method, 'invalid_null_proxy_rate')]['top2_cells']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex_table(out_dir / "best_and_top2_counts.tex", lines)

    for metric in ["aubc", "hv_final"]:
        rows = [r for r in cell_matrices if r["metric"] == metric]
        lines = [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Row setup & Current full & Previous full & MPaGE-orig & Dense reward & Hybrid reward \\\\",
            "\\midrule",
        ]
        for method in METHODS:
            row = next(r for r in rows if r["row_method"] == method)
            lines.append(
                f"{METHOD_LABELS[method]} & "
                f"{row['full']} & {row['full_old']} & {row['mpage_orig']} & "
                f"{row['dense_reward']} & {row['hybrid_reward']} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        write_latex_table(out_dir / f"pairwise_cell_wins_{metric}.tex", lines)


def run(results_root: Path, out_dir: Path) -> None:
    configure_matplotlib()
    fig_dir = ensure_dir(out_dir / "figures")
    table_dir = ensure_dir(out_dir / "tables")

    rows = enrich_rows(aggregate(str(results_root)))
    rows = [row for row in rows if row["ablation"] in METHODS]

    cell_means = build_cell_means(rows)
    rank_rows, overall, dashboard = build_rank_tables(cell_means)
    best_counts = build_best_count_table(rank_rows)
    cell_matrices, seed_matrices = build_pairwise_matrices(cell_means, rows)

    write_csv(out_dir / "summary_all_setups.csv", rows)
    write_csv(out_dir / "cell_means_all_setups.csv", cell_means)
    write_csv(out_dir / "rank_by_cell.csv", rank_rows)
    write_csv(out_dir / "overall_metric_summary.csv", overall)
    write_csv(out_dir / "overall_dashboard.csv", dashboard)
    write_csv(out_dir / "best_counts_by_task.csv", best_counts)
    write_csv(out_dir / "pairwise_cell_win_matrices.csv", cell_matrices)
    write_csv(out_dir / "pairwise_seed_win_matrices.csv", seed_matrices)

    write_latex_tables(table_dir, overall, dashboard, cell_matrices)

    for metric in METRICS:
        plot_metric_lines(cell_means, metric, fig_dir / f"{metric}_mean_all_setups.png")
        plot_normalized_heatmap(rank_rows, metric, fig_dir / f"{metric}_normalized_heatmap.png")
        plot_pairwise_matrix(
            cell_matrices,
            metric,
            fig_dir / f"{metric}_pairwise_cell_win_matrix.png",
            "task-budget cells",
        )
        plot_pairwise_matrix(
            seed_matrices,
            metric,
            fig_dir / f"{metric}_pairwise_seed_win_matrix.png",
            "seed pairs",
        )

    plot_overall_normalized_bars(
        overall,
        fig_dir / "overall_normalized_scores.png",
        PRIMARY_SUMMARY_METRICS,
        "Overall Normalized Scores by Setup",
    )
    plot_overall_normalized_bars(
        overall,
        fig_dir / "invalid_null_diagnostic_normalized_scores.png",
        DIAGNOSTIC_SUMMARY_METRICS,
        "Invalid/Null Diagnostic Scores by Setup",
    )
    plot_mean_rank_bars(
        overall,
        fig_dir / "mean_rank_by_metric.png",
        PRIMARY_SUMMARY_METRICS,
        "Mean Rank by Setup and Primary Metric",
    )
    plot_mean_rank_bars(
        overall,
        fig_dir / "invalid_null_diagnostic_mean_rank.png",
        DIAGNOSTIC_SUMMARY_METRICS,
        "Mean Rank by Setup for Invalid/Null Diagnostics",
    )
    plot_best_counts(
        overall,
        fig_dir / "best_cell_counts.png",
        PRIMARY_SUMMARY_METRICS,
        "Best-Cell Counts by Setup and Primary Metric",
    )
    plot_best_counts(
        overall,
        fig_dir / "invalid_null_diagnostic_best_cell_counts.png",
        DIAGNOSTIC_SUMMARY_METRICS,
        "Best-Cell Counts by Setup for Invalid/Null Diagnostics",
    )
    plot_tradeoff_scatter(dashboard, fig_dir / "aubc_vs_final_hv_tradeoff.png")
    plot_diagnostic_scatter(dashboard, fig_dir / "valid_yield_vs_invalid_rate_tradeoff.png")
    for budget in BUDGETS:
        plot_budget_curves(rows, budget, fig_dir / f"budget_curves_all_setups_B{budget}.png")
        plot_bmab_zoom_budget_curves(rows, budget, fig_dir / f"budget_curves_bmab_zoom_B{budget}.png")

    print(f"[all-setups] Aggregated {len(rows)} runs from {results_root}")
    print(f"[all-setups] Wrote tables and figures to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default=str(_PKG_ROOT / "experiments" / "results"),
        help="Experiment results root containing all five setup folders.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(_PKG_ROOT / "experiments" / "results" / "all_setups_overview"),
        help="Directory for generated tables and figures.",
    )
    args = parser.parse_args(argv)
    run(Path(args.results_root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
