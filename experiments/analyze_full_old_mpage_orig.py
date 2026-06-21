"""Analyze current BMAB results against full_old and mpage_orig.

This script is intentionally dependency-light: it reads the experiment run
directories directly, recomputes cell-level summaries, writes thesis-ready
tables, and generates figures under a separate output directory.

Run from the repository root:

    MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
        mpage_bmab/experiments/analyze_full_old_mpage_orig.py
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
from statistics import mean, stdev
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
_PROJECT_ROOT = _PKG_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mpage_bmab.experiments.aggregate import aggregate  # noqa: E402


TASKS = ["bi_tsp", "tri_tsp", "bi_cvrp", "bi_kp"]
BUDGETS = [25, 50, 100, 200]
METHODS = ["full", "full_old", "mpage_orig", "dense_reward", "hybrid_reward"]
CORE_METHODS = ["full", "full_old", "mpage_orig"]
COMPARATORS = ["full_old", "mpage_orig", "dense_reward", "hybrid_reward"]
METRICS = ["aubc", "hv_final"]

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
}

COLORS = {
    "full": "#2474A6",
    "full_old": "#8C6BB1",
    "mpage_orig": "#D55E00",
    "dense_reward": "#6C757D",
    "hybrid_reward": "#2CA25F",
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


def fmt(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{digits}f}"


def pct_fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.1f}\\%"


def mean_or_nan(values: Iterable[float]) -> float:
    values = [float(v) for v in values if not math.isnan(float(v))]
    return mean(values) if values else math.nan


def std_or_zero(values: Iterable[float]) -> float:
    values = [float(v) for v in values if not math.isnan(float(v))]
    return stdev(values) if len(values) > 1 else 0.0


def rel_delta(current: float, baseline: float) -> float:
    return (current / baseline - 1.0) * 100.0 if baseline else math.nan


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
    out: list[dict] = []
    for row in rows:
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
        else:
            unregistered_count = max(0, gen_calls - valid_count)
            invalid_proxy = unregistered_count / gen_calls if gen_calls else math.nan
            unregistered = unregistered_count

        d.update({
            "sample_count": sample_count,
            "valid_count": valid_count,
            "null_score_count": null_count,
            "recorded_null_rate": recorded_null_rate,
            "valid_yield_per_100_calls": valid_yield,
            "generation_attempts": gen_calls if d["ablation"] != "mpage_orig" else "",
            "non_generation_calls": non_gen_calls if d["ablation"] != "mpage_orig" else "",
            "unregistered_generation_count": unregistered,
            "invalid_null_proxy_rate": invalid_proxy,
        })
        out.append(d)
    return out


def cell_key(row: dict) -> tuple[str, str, int, int]:
    return (row["ablation"], row["task"], int(row["budget"]), int(row["seed"]))


def build_cell_means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["ablation"], row["task"], int(row["budget"]))].append(row)

    records: list[dict] = []
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
                ]:
                    vals = [float(r[metric]) for r in rs if r.get(metric) not in ("", None)]
                    rec[f"{metric}_mean"] = mean_or_nan(vals)
                    rec[f"{metric}_std"] = std_or_zero(vals)
                records.append(rec)
    return records


def paired_comparisons(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    idx = {cell_key(row): row for row in rows}
    records: list[dict] = []
    paired_seed_records: list[dict] = []
    for comparator in COMPARATORS:
        for task in TASKS:
            for budget in BUDGETS:
                seeds = sorted({
                    int(r["seed"]) for r in rows
                    if r["task"] == task and int(r["budget"]) == budget
                    and r["ablation"] in {"full", comparator}
                })
                seeds = [
                    s for s in seeds
                    if ("full", task, budget, s) in idx
                    and (comparator, task, budget, s) in idx
                ]
                if not seeds:
                    continue
                for metric in METRICS:
                    full_vals = np.array([
                        float(idx[("full", task, budget, seed)][metric])
                        for seed in seeds
                    ])
                    comp_vals = np.array([
                        float(idx[(comparator, task, budget, seed)][metric])
                        for seed in seeds
                    ])
                    delta = full_vals - comp_vals
                    pct = np.array([
                        rel_delta(f, c) for f, c in zip(full_vals, comp_vals)
                    ])
                    p_value = math.nan
                    if wilcoxon is not None and np.count_nonzero(delta):
                        try:
                            p_value = float(wilcoxon(
                                full_vals,
                                comp_vals,
                                zero_method="wilcox",
                                alternative="two-sided",
                            ).pvalue)
                        except Exception:
                            p_value = math.nan
                    records.append({
                        "comparator": comparator,
                        "task": task,
                        "budget": budget,
                        "metric": metric,
                        "n_seeds": len(seeds),
                        "full_mean": float(full_vals.mean()),
                        "comparator_mean": float(comp_vals.mean()),
                        "delta_mean": float(delta.mean()),
                        "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                        "delta_pct_from_means": rel_delta(float(full_vals.mean()), float(comp_vals.mean())),
                        "paired_delta_pct_mean": float(np.nanmean(pct)),
                        "wins": int((delta > 0).sum()),
                        "ties": int((delta == 0).sum()),
                        "losses": int((delta < 0).sum()),
                        "p_value": p_value,
                    })
                    for seed, f, c, d, p in zip(seeds, full_vals, comp_vals, delta, pct):
                        paired_seed_records.append({
                            "comparator": comparator,
                            "task": task,
                            "budget": budget,
                            "seed": seed,
                            "metric": metric,
                            "full": float(f),
                            "comparator_value": float(c),
                            "delta": float(d),
                            "delta_pct": float(p),
                        })
    return records, paired_seed_records


def summarize_pairwise(pairwise: list[dict]) -> tuple[list[dict], list[dict]]:
    by_comp_metric: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_comp_task_metric: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in pairwise:
        by_comp_metric[(row["comparator"], row["metric"])].append(row)
        by_comp_task_metric[(row["comparator"], row["task"], row["metric"])].append(row)

    overall: list[dict] = []
    for (comparator, metric), rows in sorted(by_comp_metric.items()):
        total_wins = sum(int(r["wins"]) for r in rows)
        total_losses = sum(int(r["losses"]) for r in rows)
        total_ties = sum(int(r["ties"]) for r in rows)
        overall.append({
            "comparator": comparator,
            "metric": metric,
            "n_cells": len(rows),
            "positive_cells": sum(1 for r in rows if float(r["delta_mean"]) > 0),
            "negative_cells": sum(1 for r in rows if float(r["delta_mean"]) < 0),
            "seed_wins": total_wins,
            "seed_ties": total_ties,
            "seed_losses": total_losses,
            "mean_delta_pct": mean_or_nan(r["delta_pct_from_means"] for r in rows),
            "median_delta_pct": float(np.nanmedian([r["delta_pct_from_means"] for r in rows])),
            "min_delta_pct": min(float(r["delta_pct_from_means"]) for r in rows),
            "max_delta_pct": max(float(r["delta_pct_from_means"]) for r in rows),
        })

    task_summary: list[dict] = []
    for (comparator, task, metric), rows in sorted(by_comp_task_metric.items()):
        task_summary.append({
            "comparator": comparator,
            "task": task,
            "metric": metric,
            "n_cells": len(rows),
            "positive_cells": sum(1 for r in rows if float(r["delta_mean"]) > 0),
            "seed_wins": sum(int(r["wins"]) for r in rows),
            "seed_losses": sum(int(r["losses"]) for r in rows),
            "mean_delta_pct": mean_or_nan(r["delta_pct_from_means"] for r in rows),
        })
    return overall, task_summary


def best_by_cell(cell_means: list[dict]) -> list[dict]:
    out: list[dict] = []
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in cell_means:
        by_cell[(row["task"], int(row["budget"]))].append(row)
    for (task, budget), rows in sorted(by_cell.items()):
        for metric in METRICS:
            key = f"{metric}_mean"
            best = max(rows, key=lambda r: float(r[key]))
            current = next(r for r in rows if r["method"] == "full")
            out.append({
                "task": task,
                "budget": budget,
                "metric": metric,
                "best_method": best["method"],
                "best_mean": best[key],
                "current_full_mean": current[key],
                "current_is_best": best["method"] == "full",
                "current_gap_to_best_pct": rel_delta(float(current[key]), float(best[key])),
            })
    return out


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
        consumed = float(point.get("consumed_budget", point.get("budget", 0.0)) or 0.0)
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
        compact.insert(0, (0.0, compact[0][1]))
    return compact


def mean_curve(rows: list[dict], method: str, task: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
    curves = [
        read_curve(Path(r["run_dir"]))
        for r in rows
        if r["ablation"] == method and r["task"] == task and int(r["budget"]) == budget
    ]
    curves = [c for c in curves if c]
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
        "legend.fontsize": 9,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_metric_lines(cell_means: list[dict], metric: str, methods: list[str], out: Path) -> None:
    by = {(r["method"], r["task"], int(r["budget"])): r for r in cell_means}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, task in zip(axes.ravel(), TASKS):
        for method in methods:
            vals = [by[(method, task, b)][f"{metric}_mean"] for b in BUDGETS]
            ax.plot(BUDGETS, vals, marker="o", linewidth=2.0,
                    color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Budget")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(BUDGETS)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods),
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Mean {METRIC_LABELS[metric]} by Task and Budget", y=1.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(pairwise: list[dict], comparator: str, metric: str, out: Path) -> None:
    values = np.full((len(TASKS), len(BUDGETS)), np.nan)
    for row in pairwise:
        if row["comparator"] == comparator and row["metric"] == metric:
            values[TASKS.index(row["task"]), BUDGETS.index(int(row["budget"]))] = row["delta_pct_from_means"]
    vmax = np.nanmax(np.abs(values))
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    im = ax.imshow(values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(BUDGETS)), BUDGETS)
    ax.set_yticks(range(len(TASKS)), [TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("Budget")
    ax.set_title(
        f"Current full vs {METHOD_LABELS[comparator]}: {METRIC_LABELS[metric]} delta (%)"
    )
    for i in range(len(TASKS)):
        for j in range(len(BUDGETS)):
            ax.text(j, i, f"{values[i, j]:+.1f}%", ha="center", va="center",
                    color="black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta from comparator (%)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_overall_bars(overall: list[dict], out: Path) -> None:
    comparators = ["full_old", "mpage_orig", "dense_reward", "hybrid_reward"]
    x = np.arange(len(comparators))
    width = 0.36
    vals = {
        metric: [
            next(r["mean_delta_pct"] for r in overall if r["comparator"] == comp and r["metric"] == metric)
            for comp in comparators
        ]
        for metric in METRICS
    }
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.bar(x - width / 2, vals["aubc"], width, label="AUBC", color="#2474A6")
    ax.bar(x + width / 2, vals["hv_final"], width, label="Final HV", color="#E69F00")
    ax.set_xticks(x, [METHOD_LABELS[c] for c in comparators], rotation=15, ha="right")
    ax.set_ylabel("Mean cell-level delta of current full (%)")
    ax.set_title("Overall Current-Full Advantage by Comparator")
    ax.legend(frameon=False)
    for metric, offset in [("aubc", -width / 2), ("hv_final", width / 2)]:
        for xi, value in zip(x, vals[metric]):
            va = "bottom" if value >= 0 else "top"
            y = value + (0.7 if value >= 0 else -0.7)
            ax.text(xi + offset, y, f"{value:+.1f}%", ha="center", va=va, fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_cell_delta_boxplots(pairwise: list[dict], out: Path) -> None:
    comparators = ["full_old", "mpage_orig", "dense_reward", "hybrid_reward"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        data = [
            [r["delta_pct_from_means"] for r in pairwise
             if r["comparator"] == comp and r["metric"] == metric]
            for comp in comparators
        ]
        ax.axhline(0, color="#333333", linewidth=1)
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[METHOD_LABELS[c] for c in comparators])
        for patch, comp in zip(bp["boxes"], comparators):
            patch.set_facecolor(COLORS[comp])
            patch.set_alpha(0.45)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("Task-budget cell delta (%)")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Current full minus comparator: task-budget cell distribution")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_budget_curves(rows: list[dict], budget: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, task in zip(axes.ravel(), TASKS):
        for method in CORE_METHODS:
            xs, ys = mean_curve(rows, method, task, budget)
            ax.plot(xs, ys, linewidth=2.0, color=COLORS[method],
                    label=METHOD_LABELS[method])
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Normalized budget")
        ax.set_ylabel("Outer HV")
        ax.set_xlim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Mean Budget Curves at B={budget}", y=1.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: str) -> str:
    return str(text).replace("_", "\\_")


def write_latex_tables(out_dir: Path, overall: list[dict], task_summary: list[dict], pairwise: list[dict]) -> None:
    ensure_dir(out_dir)

    core_comparators = ["full_old", "mpage_orig"]
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Comparator & Metric & Positive cells & Seed wins & Mean $\\Delta\\%$ & Range $\\Delta\\%$ \\\\",
        "\\midrule",
    ]
    for comp in core_comparators:
        for metric in METRICS:
            row = next(r for r in overall if r["comparator"] == comp and r["metric"] == metric)
            lines.append(
                f"{METHOD_LABELS[comp]} & {METRIC_LABELS[metric]} & "
                f"{row['positive_cells']}/{row['n_cells']} & "
                f"{row['seed_wins']}/{row['seed_wins'] + row['seed_losses'] + row['seed_ties']} & "
                f"{pct_fmt(row['mean_delta_pct'])} & "
                f"{pct_fmt(row['min_delta_pct'])}--{pct_fmt(row['max_delta_pct'])} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out_dir / "overall_comparison_core.tex").write_text("\n".join(lines))

    for metric in METRICS:
        lines = [
            "\\begin{tabular}{llrrr}",
            "\\toprule",
            "Comparator & Task & Positive cells & Seed wins & Mean $\\Delta\\%$ \\\\",
            "\\midrule",
        ]
        for comp in core_comparators:
            for task in TASKS:
                row = next(r for r in task_summary if r["comparator"] == comp and r["task"] == task and r["metric"] == metric)
                total = row["seed_wins"] + row["seed_losses"]
                lines.append(
                    f"{METHOD_LABELS[comp]} & {TASK_LABELS[task]} & "
                    f"{row['positive_cells']}/{row['n_cells']} & "
                    f"{row['seed_wins']}/{total} & "
                    f"{pct_fmt(row['mean_delta_pct'])} \\\\"
                )
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        (out_dir / f"task_summary_{metric}.tex").write_text("\n".join(lines))

    selected = [
        r for r in pairwise
        if r["comparator"] in core_comparators
        and r["metric"] == "aubc"
    ]
    selected.sort(key=lambda r: abs(float(r["delta_pct_from_means"])), reverse=True)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Comparator & Cell & Current full & Comparator & $\\Delta\\%$ & Wins \\\\",
        "\\midrule",
    ]
    for row in selected[:12]:
        cell = f"{TASK_LABELS[row['task']]}, B={row['budget']}"
        total = row["wins"] + row["losses"] + row["ties"]
        lines.append(
            f"{METHOD_LABELS[row['comparator']]} & {cell} & "
            f"{fmt(row['full_mean'])} & {fmt(row['comparator_mean'])} & "
            f"{pct_fmt(row['delta_pct_from_means'])} & {row['wins']}/{total} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out_dir / "largest_aubc_differences_core.tex").write_text("\n".join(lines))


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected:
        vals = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value, 2)
            vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def write_report(
    out_path: Path,
    rows: list[dict],
    overall: list[dict],
    task_summary: list[dict],
    pairwise: list[dict],
    best: list[dict],
) -> None:
    def get_overall(comp: str, metric: str) -> dict:
        return next(r for r in overall if r["comparator"] == comp and r["metric"] == metric)

    def key_sentence(comp: str, metric: str) -> str:
        row = get_overall(comp, metric)
        return (
            f"Against {METHOD_LABELS[comp]}, current full improves {METRIC_LABELS[metric]} "
            f"in {row['positive_cells']}/{row['n_cells']} cells, wins "
            f"{row['seed_wins']}/{row['seed_wins'] + row['seed_losses'] + row['seed_ties']} "
            f"paired seed comparisons, and has mean cell-level delta "
            f"{row['mean_delta_pct']:+.1f}%."
        )

    current_best_aubc = sum(1 for r in best if r["metric"] == "aubc" and r["current_is_best"])
    current_best_hv = sum(1 for r in best if r["metric"] == "hv_final" and r["current_is_best"])

    largest = [r for r in pairwise if r["comparator"] in {"full_old", "mpage_orig"} and r["metric"] == "aubc"]
    largest.sort(key=lambda r: abs(float(r["delta_pct_from_means"])), reverse=True)
    largest_rows = []
    for r in largest[:10]:
        largest_rows.append({
            "Comparator": METHOD_LABELS[r["comparator"]],
            "Cell": f"{TASK_LABELS[r['task']]}, B={r['budget']}",
            "Current full": fmt(r["full_mean"]),
            "Comparator mean": fmt(r["comparator_mean"]),
            "Delta %": f"{r['delta_pct_from_means']:+.1f}%",
            "Wins": f"{r['wins']}/{r['wins'] + r['losses'] + r['ties']}",
        })

    report = f"""# Current Full vs Previous Full and MPaGE-orig

This report analyzes the run artifacts under `mpage_bmab/experiments/results`.
The analysis does **not** rely on the existing `summary.csv`, because that file
does not include the copied `full_old` and `mpage_orig` folders. Instead, the
analysis script re-aggregates the run directories directly and finds {len(rows)}
complete runs: 5 methods x 4 tasks x 4 budgets x 5 seeds.

## Compared Methods

- `full`: current finalized BMAB configuration.
- `full_old`: previous full-version result set copied from the earlier thesis.
- `mpage_orig`: original MPaGE baseline result set copied for comparison.
- `dense_reward` and `hybrid_reward`: current reward-mode variants; these are
  included as secondary context, but the main historical comparisons are
  current `full` versus `full_old` and `mpage_orig`.

## Main Findings

1. {key_sentence('full_old', 'aubc')}
2. {key_sentence('mpage_orig', 'aubc')}
3. {key_sentence('full_old', 'hv_final')}
4. {key_sentence('mpage_orig', 'hv_final')}

Current `full` is the best method in {current_best_aubc}/16 task-budget cells
for AUBC and {current_best_hv}/16 cells for final HV. This confirms that the
current implementation is not uniformly dominant at every terminal point, but
it is useful to distinguish budget-efficiency behavior (AUBC) from final
population quality (final HV).

## Thesis-Ready Figures

- `figures/aubc_mean_core_methods.png`: AUBC trends for current full,
  previous full, and MPaGE-orig.
- `figures/hv_final_mean_core_methods.png`: final-HV trends for the same
  three methods.
- `figures/aubc_delta_current_vs_full_old_heatmap.png` and
  `figures/aubc_delta_current_vs_mpage_orig_heatmap.png`: task-budget AUBC
  differences.
- `figures/hv_final_delta_current_vs_full_old_heatmap.png` and
  `figures/hv_final_delta_current_vs_mpage_orig_heatmap.png`: task-budget
  final-HV differences.
- `figures/overall_delta_bars.png`: compact overview of mean cell-level
  percentage deltas.
- `figures/cell_delta_boxplots.png`: task-budget cell-level delta
  distributions.
- `figures/budget_curves_core_B*.png`: mean budget curves for each budget.

## Overall Summary

{markdown_table(overall, [
    ('Comparator', 'comparator'),
    ('Metric', 'metric'),
    ('Cells +', 'positive_cells'),
    ('Cells -', 'negative_cells'),
    ('Seed wins', 'seed_wins'),
    ('Seed losses', 'seed_losses'),
    ('Mean delta %', 'mean_delta_pct'),
    ('Min delta %', 'min_delta_pct'),
    ('Max delta %', 'max_delta_pct'),
])}

## Largest AUBC Differences

{markdown_table(largest_rows, [
    ('Comparator', 'Comparator'),
    ('Cell', 'Cell'),
    ('Current full', 'Current full'),
    ('Comparator', 'Comparator mean'),
    ('Delta %', 'Delta %'),
    ('Wins', 'Wins'),
])}

## Interpretation

The current full configuration should be presented as the official finalized
BMAB method. The comparison with `full_old` is useful for documenting how the
final implementation behaves relative to the previous thesis result set. The
comparison with `mpage_orig` is the main baseline comparison. Because every
task-budget-method cell contains only five seeds, the Wilcoxon p-values in the
CSV tables should be interpreted cautiously; directional consistency, paired
seed wins, and cell-level deltas are more informative for thesis discussion
than strict significance claims.

When incorporating these assets into the thesis, the safest wording is:
current full improves or degrades a metric **in the analyzed result matrix**,
not that it proves universal superiority. This distinction is especially
important because AUBC and final HV measure different properties: AUBC measures
how quickly useful heuristic-population HV is obtained under budget, while
final HV measures only the terminal population at the last budget point.
"""
    out_path.write_text(report)


def run(results_root: Path, out_dir: Path) -> None:
    configure_matplotlib()
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    table_dir = ensure_dir(out_dir / "tables")

    rows = aggregate(str(results_root))
    rows = [r for r in rows if r["ablation"] in METHODS]
    rows = enrich_rows(rows)

    cell_means = build_cell_means(rows)
    pairwise, paired_seed_records = paired_comparisons(rows)
    overall, task_summary = summarize_pairwise(pairwise)
    best = best_by_cell(cell_means)

    write_csv(out_dir / "summary_all_methods.csv", rows)
    write_csv(out_dir / "cell_means_all_methods.csv", cell_means)
    write_csv(out_dir / "pairwise_current_full_vs_comparators.csv", pairwise)
    write_csv(out_dir / "paired_seed_deltas.csv", paired_seed_records)
    write_csv(out_dir / "overall_summary.csv", overall)
    write_csv(out_dir / "task_level_summary.csv", task_summary)
    write_csv(out_dir / "best_by_cell.csv", best)

    write_latex_tables(table_dir, overall, task_summary, pairwise)

    plot_metric_lines(cell_means, "aubc", CORE_METHODS, fig_dir / "aubc_mean_core_methods.png")
    plot_metric_lines(cell_means, "hv_final", CORE_METHODS, fig_dir / "hv_final_mean_core_methods.png")
    for comparator in ["full_old", "mpage_orig"]:
        for metric in METRICS:
            plot_delta_heatmap(
                pairwise,
                comparator,
                metric,
                fig_dir / f"{metric}_delta_current_vs_{comparator}_heatmap.png",
            )
    plot_overall_bars(overall, fig_dir / "overall_delta_bars.png")
    plot_cell_delta_boxplots(pairwise, fig_dir / "cell_delta_boxplots.png")
    # Keep the legacy filename harmless if an older generated folder already
    # contains it; the report points to cell_delta_boxplots.png.
    plot_cell_delta_boxplots(pairwise, fig_dir / "paired_seed_delta_boxplots.png")
    for budget in BUDGETS:
        plot_budget_curves(rows, budget, fig_dir / f"budget_curves_core_B{budget}.png")

    write_report(out_dir / "analysis_report.md", rows, overall, task_summary, pairwise, best)

    print(f"[analysis] Aggregated {len(rows)} runs from {results_root}")
    print(f"[analysis] Wrote outputs to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default=str(_PKG_ROOT / "experiments" / "results"),
        help="Experiment results root containing full, full_old, and mpage_orig.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(_PKG_ROOT / "experiments" / "results" / "full_old_mpage_orig_analysis"),
        help="Directory for generated tables, figures, and report.",
    )
    args = parser.parse_args(argv)
    run(Path(args.results_root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
