"""Generate analysis figures for the MPaGE vs mpage_bmab result set.

This is the plotting code used for the figures in:

    mpage_bmab/experiments/results/images/

The script reads:

    summary.csv
    <run_dir>/budget_curve.json
    <run_dir>/budget_history.json
    <run_dir>/samples/samples_*.json

and writes:

    images/*.png
    derived_cell_stats.csv
    validity_stats.csv
    validity_cell_stats.csv

Run from the project root:

    MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
        mpage_bmab/experiments/results/image_generation_code/generate_result_images.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover - only used when scipy is unavailable.
    wilcoxon = None


TASKS = ["bi_tsp", "tri_tsp", "bi_cvrp", "bi_kp"]
BUDGETS = [25, 50, 100, 200]

TASK_LABELS = {
    "bi_tsp": "Bi-TSP",
    "tri_tsp": "Tri-TSP",
    "bi_cvrp": "Bi-CVRP",
    "bi_kp": "Bi-KP",
}

METHOD_LABELS = {
    "mpage_orig": "MPaGE-orig",
    "full": "mpage_bmab",
}

COLORS = {
    "mpage_orig": "#D55E00",
    "full": "#2474A6",
}


def read_summary(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    with (results_root / "summary.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            d = dict(row)
            for key in ["budget", "seed", "pareto_size", "n_calls"]:
                d[key] = int(float(d[key]))
            for key in ["aubc", "hv_final", "consumed_budget", "total_budget"]:
                d[key] = float(d[key])
            rows.append(d)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_cell_stats(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    idx = {
        (r["ablation"], r["task"], r["budget"], r["seed"]): r
        for r in rows
    }

    cell_stats: list[dict] = []
    paired_records: list[dict] = []

    for task in TASKS:
        for budget in BUDGETS:
            seeds = sorted({
                r["seed"]
                for r in rows
                if r["task"] == task and r["budget"] == budget
            })
            full = [idx[("full", task, budget, seed)] for seed in seeds]
            orig = [idx[("mpage_orig", task, budget, seed)] for seed in seeds]
            rec: dict = {"task": task, "budget": budget, "n": len(seeds)}

            for metric in ["aubc", "hv_final"]:
                full_vals = np.array([x[metric] for x in full], dtype=float)
                orig_vals = np.array([x[metric] for x in orig], dtype=float)
                delta = full_vals - orig_vals
                pct = np.where(orig_vals != 0, delta / orig_vals * 100.0, np.nan)
                p_value = math.nan
                if wilcoxon is not None and np.count_nonzero(delta):
                    try:
                        p_value = float(
                            wilcoxon(
                                full_vals,
                                orig_vals,
                                zero_method="wilcox",
                                alternative="two-sided",
                            ).pvalue
                        )
                    except Exception:
                        p_value = math.nan

                rec.update({
                    f"{metric}_full_mean": float(full_vals.mean()),
                    f"{metric}_full_std": float(full_vals.std(ddof=1)),
                    f"{metric}_orig_mean": float(orig_vals.mean()),
                    f"{metric}_orig_std": float(orig_vals.std(ddof=1)),
                    f"{metric}_delta_mean": float(delta.mean()),
                    f"{metric}_delta_std": float(delta.std(ddof=1)),
                    f"{metric}_pct_delta": float(
                        (full_vals.mean() / orig_vals.mean() - 1.0) * 100.0
                    ) if orig_vals.mean() else math.nan,
                    f"{metric}_paired_pct_mean": float(np.nanmean(pct)),
                    f"{metric}_win_seeds": int((delta > 0).sum()),
                    f"{metric}_p": p_value,
                    f"{metric}_full_median": float(np.median(full_vals)),
                    f"{metric}_orig_median": float(np.median(orig_vals)),
                })

                for seed, full_val, orig_val, delta_val, pct_val in zip(
                    seeds, full_vals, orig_vals, delta, pct
                ):
                    paired_records.append({
                        "task": task,
                        "budget": budget,
                        "seed": seed,
                        "metric": metric,
                        "full": float(full_val),
                        "mpage_orig": float(orig_val),
                        "delta": float(delta_val),
                        "pct_delta": float(pct_val),
                    })

            cell_stats.append(rec)

    return cell_stats, paired_records


def sample_counts(run_dir: Path) -> tuple[int, int, int]:
    total = 0
    valid = 0
    null = 0

    for sample_file in (run_dir / "samples").glob("samples_*.json"):
        try:
            data = json.loads(sample_file.read_text())
        except Exception:
            continue
        total += len(data)
        for sample in data:
            if sample.get("score") is None:
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
    for entry in history:
        label = entry.get("label", "")
        if (
            label == "init"
            or label.startswith("mutate")
            or label.startswith("crossover")
        ):
            generation_calls += 1
        else:
            non_generation_calls += 1
    return generation_calls, non_generation_calls


def add_validity_fields(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    for row in rows:
        run_dir = Path(row["run_dir"])
        sample_count, valid_count, null_count = sample_counts(run_dir)

        row["sample_count"] = sample_count
        row["valid_count"] = valid_count
        row["null_score_count"] = null_count
        row["recorded_null_rate"] = (
            null_count / sample_count if sample_count else math.nan
        )
        row["valid_yield_per_budget"] = (
            valid_count / row["total_budget"] if row["total_budget"] else math.nan
        )

        if row["ablation"] == "full":
            gen_calls, non_gen_calls = bmab_generation_attempts(run_dir)
            row["generation_attempts"] = gen_calls
            row["non_generation_calls"] = non_gen_calls
            row["unregistered_generation_count"] = max(0, gen_calls - valid_count)
            row["invalid_null_proxy_rate"] = (
                row["unregistered_generation_count"] / gen_calls
                if gen_calls else math.nan
            )
        else:
            row["generation_attempts"] = ""
            row["non_generation_calls"] = ""
            row["unregistered_generation_count"] = ""
            row["invalid_null_proxy_rate"] = row["recorded_null_rate"]

    validity_rows = []
    for row in rows:
        validity_rows.append({
            "ablation": row["ablation"],
            "task": row["task"],
            "budget": row["budget"],
            "seed": row["seed"],
            "total_budget": row["total_budget"],
            "sample_count": row["sample_count"],
            "valid_count": row["valid_count"],
            "null_score_count": row["null_score_count"],
            "recorded_null_rate": row["recorded_null_rate"],
            "valid_yield_per_budget": row["valid_yield_per_budget"],
            "generation_attempts": row["generation_attempts"],
            "non_generation_calls": row["non_generation_calls"],
            "unregistered_generation_count": row["unregistered_generation_count"],
            "invalid_null_proxy_rate": row["invalid_null_proxy_rate"],
            "run_dir": row["run_dir"],
        })

    validity_cell_stats: list[dict] = []
    for task in TASKS:
        for budget in BUDGETS:
            rec: dict = {"task": task, "budget": budget}
            for method in ["full", "mpage_orig"]:
                method_rows = [
                    r for r in rows
                    if (
                        r["ablation"] == method
                        and r["task"] == task
                        and r["budget"] == budget
                    )
                ]
                for metric in [
                    "valid_yield_per_budget",
                    "recorded_null_rate",
                    "invalid_null_proxy_rate",
                    "valid_count",
                    "null_score_count",
                    "sample_count",
                ]:
                    values = np.array([
                        float(r[metric])
                        for r in method_rows
                        if r[metric] != ""
                        and not (
                            isinstance(r[metric], float)
                            and math.isnan(r[metric])
                        )
                    ], dtype=float)
                    rec[f"{method}_{metric}_mean"] = (
                        float(values.mean()) if values.size else math.nan
                    )
                    rec[f"{method}_{metric}_std"] = (
                        float(values.std(ddof=1)) if values.size > 1 else 0.0
                    )

            orig_yield = rec["mpage_orig_valid_yield_per_budget_mean"]
            full_yield = rec["full_valid_yield_per_budget_mean"]
            rec["yield_delta"] = full_yield - orig_yield
            rec["yield_delta_pct"] = (
                (full_yield / orig_yield - 1.0) * 100.0
                if orig_yield else math.nan
            )
            rec["invalid_proxy_delta"] = (
                rec["full_invalid_null_proxy_rate_mean"]
                - rec["mpage_orig_invalid_null_proxy_rate_mean"]
            )
            validity_cell_stats.append(rec)

    return validity_rows, validity_cell_stats


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def metric_mean_and_std(
    rows_by_group: dict[tuple[str, str, int], list[dict]],
    method: str,
    task: str,
    budget: int,
    metric: str,
) -> tuple[float, float]:
    values = np.array(
        [r[metric] for r in rows_by_group[(method, task, budget)]],
        dtype=float,
    )
    return float(values.mean()), float(values.std(ddof=1))


def plot_metric_by_budget(
    rows_by_group: dict[tuple[str, str, int], list[dict]],
    images_dir: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, task in zip(axes.ravel(), TASKS):
        for method in ["mpage_orig", "full"]:
            means = []
            errors = []
            for budget in BUDGETS:
                mean, std = metric_mean_and_std(
                    rows_by_group, method, task, budget, metric
                )
                means.append(mean)
                errors.append(std / math.sqrt(5))
            ax.errorbar(
                BUDGETS,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                capsize=3,
            )
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("LLM-call budget B")
        ax.set_ylabel(ylabel)
        ax.set_xticks(BUDGETS)
        ax.ticklabel_format(axis="y", style="plain")

    fig.suptitle(title, fontsize=14)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(images_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(
    cell_stats: list[dict],
    images_dir: Path,
    *,
    metric: str,
    title: str,
    filename: str,
    vmin: float,
    vmax: float,
) -> None:
    data = np.zeros((len(TASKS), len(BUDGETS)))
    pvals = np.zeros_like(data)
    wins = np.zeros_like(data)

    for i, task in enumerate(TASKS):
        for j, budget in enumerate(BUDGETS):
            rec = next(
                r for r in cell_stats
                if r["task"] == task and r["budget"] == budget
            )
            data[i, j] = rec[f"{metric}_pct_delta"]
            pvals[i, j] = rec[f"{metric}_p"]
            wins[i, j] = rec[f"{metric}_win_seeds"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(BUDGETS)), labels=[str(b) for b in BUDGETS])
    ax.set_yticks(range(len(TASKS)), labels=[TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("LLM-call budget B")
    ax.set_title(title)

    for i in range(len(TASKS)):
        for j in range(len(BUDGETS)):
            sig = "*" if pvals[i, j] < 0.05 else ("~" if pvals[i, j] <= 0.125 else "")
            ax.text(
                j,
                i,
                f"{data[i, j]:+.1f}%\n{int(wins[i, j])}/5{sig}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean delta vs MPaGE-orig (%)")
    fig.savefig(images_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_paired_aubc_boxplot(
    paired_records: list[dict],
    images_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    box_data = []
    labels = []
    for task in TASKS:
        values = [
            r["pct_delta"]
            for r in paired_records
            if r["metric"] == "aubc" and r["task"] == task
        ]
        box_data.append(values)
        labels.append(TASK_LABELS[task])

    ax.boxplot(
        box_data,
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#D7EAF7", "edgecolor": COLORS["full"]},
        medianprops={"color": "#111111"},
        meanprops={
            "marker": "D",
            "markerfacecolor": COLORS["mpage_orig"],
            "markeredgecolor": COLORS["mpage_orig"],
            "markersize": 5,
        },
    )

    rng = np.random.default_rng(2026)
    for i, values in enumerate(box_data, start=1):
        xs = i + rng.normal(0, 0.045, size=len(values))
        ax.scatter(xs, values, s=18, alpha=0.65, color="#444444")

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("Paired AUBC delta vs MPaGE-orig (%)")
    ax.set_title("Per-seed AUBC deltas across all budgets")
    fig.savefig(images_dir / "paired_aubc_delta_boxplot.png", bbox_inches="tight")
    plt.close(fig)


def load_budget_curve(row: dict) -> tuple[np.ndarray, np.ndarray]:
    curve_path = Path(row["run_dir"]) / "budget_curve.json"
    try:
        curve = json.loads(curve_path.read_text())
    except Exception:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    total_budget = row["total_budget"] or float(row["budget"])
    xs = [0.0]
    ys = [0.0]
    for point in sorted(curve, key=lambda x: x.get("budget_consumed", 0)):
        xs.append(float(point.get("budget_consumed", 0.0)) / total_budget)
        ys.append(float(point.get("hv", 0.0)))

    # Multiple curve points can share the same budget position. Keep the last.
    dedup = {}
    for x, y in zip(xs, ys):
        dedup[x] = y
    xs = np.array(sorted(dedup.keys()), dtype=float)
    ys = np.array([dedup[x] for x in xs], dtype=float)

    if xs[-1] < 1.0:
        xs = np.append(xs, 1.0)
        ys = np.append(ys, ys[-1])
    return xs, ys


def plot_budget_curves(rows: list[dict], images_dir: Path, budget: int) -> None:
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for ax, task in zip(axes.ravel(), TASKS):
        for method in ["mpage_orig", "full"]:
            curves = []
            for row in rows:
                if (
                    row["ablation"] == method
                    and row["task"] == task
                    and row["budget"] == budget
                ):
                    xs, ys = load_budget_curve(row)
                    curves.append(np.interp(grid, xs, ys))

            if not curves:
                continue

            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            se = (
                arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0])
                if arr.shape[0] > 1 else np.zeros_like(mean)
            )
            ax.plot(grid, mean, color=COLORS[method], linewidth=2, label=METHOD_LABELS[method])
            ax.fill_between(
                grid,
                mean - se,
                mean + se,
                color=COLORS[method],
                alpha=0.15,
                linewidth=0,
            )

        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel(f"Budget consumed / B (B={budget})")
        ax.set_ylabel("Heuristic-population HV")
        ax.ticklabel_format(axis="y", style="plain")

    fig.suptitle(f"Mean budget curves at B={budget}", fontsize=14)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(images_dir / f"budget_curves_B{budget}.png", bbox_inches="tight")
    plt.close(fig)


def plot_aubc_to_final_hv_ratio(
    rows_by_group: dict[tuple[str, str, int], list[dict]],
    images_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for ax, task in zip(axes.ravel(), TASKS):
        for method in ["mpage_orig", "full"]:
            means = []
            errors = []
            for budget in BUDGETS:
                values = []
                for row in rows_by_group[(method, task, budget)]:
                    if row["hv_final"] > 0:
                        values.append(row["aubc"] / row["hv_final"])
                arr = np.array(values, dtype=float)
                means.append(float(np.nanmean(arr)))
                errors.append(
                    float(np.nanstd(arr, ddof=1) / math.sqrt(np.count_nonzero(~np.isnan(arr))))
                )

            ax.errorbar(
                BUDGETS,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                capsize=3,
            )

        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("LLM-call budget B")
        ax.set_ylabel("AUBC / final HV")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(BUDGETS)

    fig.suptitle("How early final-quality gains appear (AUBC / final HV)", fontsize=14)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(images_dir / "aubc_to_final_hv_ratio.png", bbox_inches="tight")
    plt.close(fig)


def plot_valid_yield(rows: list[dict], images_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for ax, task in zip(axes.ravel(), TASKS):
        for method in ["mpage_orig", "full"]:
            means = []
            errors = []
            for budget in BUDGETS:
                values = np.array([
                    r["valid_yield_per_budget"]
                    for r in rows
                    if (
                        r["ablation"] == method
                        and r["task"] == task
                        and r["budget"] == budget
                    )
                ], dtype=float) * 100.0
                means.append(values.mean())
                errors.append(values.std(ddof=1) / math.sqrt(len(values)))

            ax.errorbar(
                BUDGETS,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )

        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("LLM-call budget B")
        ax.set_ylabel("Valid heuristics per 100 calls")
        ax.set_xticks(BUDGETS)

    fig.suptitle("Valid heuristic yield per budget", fontsize=14)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(images_dir / "valid_yield_by_budget.png", bbox_inches="tight")
    plt.close(fig)


def plot_invalid_null_rate(rows: list[dict], images_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    for ax, task in zip(axes.ravel(), TASKS):
        for method in ["mpage_orig", "full"]:
            means = []
            errors = []
            for budget in BUDGETS:
                values = np.array([
                    r["invalid_null_proxy_rate"]
                    for r in rows
                    if (
                        r["ablation"] == method
                        and r["task"] == task
                        and r["budget"] == budget
                    )
                ], dtype=float) * 100.0
                means.append(values.mean())
                errors.append(values.std(ddof=1) / math.sqrt(len(values)))

            ax.errorbar(
                BUDGETS,
                means,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )

        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("LLM-call budget B")
        ax.set_ylabel("Invalid/null proxy rate (%)")
        ax.set_xticks(BUDGETS)
        ax.set_ylim(0, 100)

    fig.suptitle("Invalid/null-score rate proxies by task and budget", fontsize=14)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(images_dir / "invalid_null_rate_by_budget.png", bbox_inches="tight")
    plt.close(fig)


def plot_valid_yield_delta_heatmap(
    validity_cell_stats: list[dict],
    images_dir: Path,
) -> None:
    data = np.zeros((len(TASKS), len(BUDGETS)))
    for i, task in enumerate(TASKS):
        for j, budget in enumerate(BUDGETS):
            rec = next(
                r for r in validity_cell_stats
                if r["task"] == task and r["budget"] == budget
            )
            data[i, j] = rec["yield_delta_pct"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=55)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(BUDGETS)), labels=[str(b) for b in BUDGETS])
    ax.set_yticks(range(len(TASKS)), labels=[TASK_LABELS[t] for t in TASKS])
    ax.set_xlabel("LLM-call budget B")
    ax.set_title("Valid-yield relative improvement: mpage_bmab vs MPaGE-orig")

    for i in range(len(TASKS)):
        for j in range(len(BUDGETS)):
            ax.text(j, i, f"{data[i, j]:+.1f}%", ha="center", va="center", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean valid-yield delta (%)")
    fig.savefig(images_dir / "valid_yield_delta_percent_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def generate_all(results_root: Path, images_dir: Path) -> None:
    rows = read_summary(results_root)
    rows_by_group: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_group[(row["ablation"], row["task"], row["budget"])].append(row)

    cell_stats, paired_records = compute_cell_stats(rows)
    write_csv(results_root / "derived_cell_stats.csv", cell_stats)

    validity_rows, validity_cell_stats = add_validity_fields(rows)
    write_csv(results_root / "validity_stats.csv", validity_rows)
    write_csv(results_root / "validity_cell_stats.csv", validity_cell_stats)

    configure_matplotlib()
    images_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_by_budget(
        rows_by_group,
        images_dir,
        metric="aubc",
        ylabel="AUBC",
        title="Budget efficiency: mean AUBC by task and budget",
        filename="aubc_mean_by_budget.png",
    )
    plot_metric_by_budget(
        rows_by_group,
        images_dir,
        metric="hv_final",
        ylabel="Final heuristic-population HV",
        title="Final quality: mean final HV by task and budget",
        filename="hv_final_mean_by_budget.png",
    )
    plot_delta_heatmap(
        cell_stats,
        images_dir,
        metric="aubc",
        title="AUBC relative improvement: mpage_bmab vs MPaGE-orig",
        filename="aubc_delta_percent_heatmap.png",
        vmin=-25,
        vmax=80,
    )
    plot_delta_heatmap(
        cell_stats,
        images_dir,
        metric="hv_final",
        title="Final HV relative improvement: mpage_bmab vs MPaGE-orig",
        filename="hv_final_delta_percent_heatmap.png",
        vmin=-25,
        vmax=25,
    )
    plot_paired_aubc_boxplot(paired_records, images_dir)

    for budget in BUDGETS:
        plot_budget_curves(rows, images_dir, budget)

    plot_aubc_to_final_hv_ratio(rows_by_group, images_dir)
    plot_valid_yield(rows, images_dir)
    plot_invalid_null_rate(rows, images_dir)
    plot_valid_yield_delta_heatmap(validity_cell_stats, images_dir)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Generate figures for MPaGE vs mpage_bmab result analysis."
    )
    parser.add_argument(
        "--results-root",
        default=str(default_results_root),
        help="Directory containing summary.csv and run subdirectories.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Output image directory. Defaults to <results-root>/images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    images_dir = (
        Path(args.images_dir).resolve()
        if args.images_dir
        else results_root / "images"
    )

    generate_all(results_root, images_dir)
    print(f"Wrote figures to: {images_dir}")
    print(f"Wrote derived tables to: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
