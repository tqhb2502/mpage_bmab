"""Generate focused tables and figures for the dense_reward Bi-CVRP B25 rerun.

This script is intentionally narrow: it documents the experiment cell that was
rerun after the old seed2029 result produced AUBC=0. It reads the current
result tree and writes only tables/figures, with no narrative report.

Run from the repository root:

    MPLCONFIGDIR=/private/tmp/mpl mpage_bmab/.venv/bin/python \
        mpage_bmab/experiments/analyze_rerun_cell_update.py
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
from mpage_bmab.experiments.analyze_all_setups_overview import (  # noqa: E402
    COLORS,
    METHODS,
    METHOD_LABELS,
    TASK_LABELS,
    enrich_rows,
    read_curve,
)


TARGET_METHOD = "dense_reward"
TARGET_TASK = "bi_cvrp"
TARGET_BUDGET = 25
TARGET_SEED = 2029
METRICS = ["aubc", "hv_final", "valid_count", "valid_yield_per_100_calls"]


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


def write_latex(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


def fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{float(value):.{digits}f}"


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


def std_or_zero(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def cell_mean_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["ablation"], row["task"], int(row["budget"]))].append(row)

    out: list[dict] = []
    for method in METHODS:
        rs = groups[(method, TARGET_TASK, TARGET_BUDGET)]
        rec: dict = {
            "method": method,
            "task": TARGET_TASK,
            "budget": TARGET_BUDGET,
            "n": len(rs),
        }
        for metric in METRICS:
            vals = [float(r[metric]) for r in rs]
            rec[f"{metric}_mean"] = mean(vals)
            rec[f"{metric}_std"] = std_or_zero(vals)
        out.append(rec)
    return out


def rank_cell_means(rows: list[dict]) -> list[dict]:
    ranked: list[dict] = []
    for metric in METRICS:
        higher = metric != "invalid_null_proxy_rate"
        ordered = sorted(
            rows,
            key=lambda r: float(r[f"{metric}_mean"]),
            reverse=higher,
        )
        for rank, row in enumerate(ordered, start=1):
            ranked.append({
                "metric": metric,
                "rank": rank,
                "method": row["method"],
                "value": row[f"{metric}_mean"],
            })
    return ranked


def load_curve_points(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    curve = read_curve(run_dir)
    xs = np.array([p[0] for p in curve], dtype=float)
    ys = np.array([p[1] for p in curve], dtype=float)
    return xs, ys


def mean_curve(rows: list[dict], method: str) -> tuple[np.ndarray, np.ndarray]:
    xs_grid = np.linspace(0.0, 1.0, 101)
    ys_all = []
    for row in rows:
        if (
            row["ablation"] == method
            and row["task"] == TARGET_TASK
            and int(row["budget"]) == TARGET_BUDGET
        ):
            xs, ys = load_curve_points(Path(row["run_dir"]))
            if len(xs) == 0:
                continue
            ys_all.append(np.interp(xs_grid, xs, ys))
    if not ys_all:
        return xs_grid, np.full_like(xs_grid, np.nan)
    return xs_grid, np.vstack(ys_all).mean(axis=0)


def plot_dense_per_seed_bars(rows: list[dict], fig_dir: Path) -> None:
    rs = sorted(rows, key=lambda r: int(r["seed"]))
    seeds = [str(r["seed"]) for r in rs]
    x = np.arange(len(rs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, [float(r["aubc"]) for r in rs], width,
           label="AUBC", color="#2474A6")
    ax.bar(x + width / 2, [float(r["hv_final"]) for r in rs], width,
           label="Final HV", color="#E69F00")
    for i, r in enumerate(rs):
        ax.text(i, max(float(r["aubc"]), float(r["hv_final"])) * 1.02,
                f"n={int(r['valid_count'])}", ha="center", va="bottom",
                fontsize=8)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Outer HV")
    ax.set_title("Dense Reward: Bi-CVRP B25 Per-Seed AUBC and Final HV")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(fig_dir / "dense_reward_bi_cvrp_B25_per_seed_aubc_hv.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_valid_samples(rows: list[dict], fig_dir: Path) -> None:
    rs = sorted(rows, key=lambda r: int(r["seed"]))
    seeds = [str(r["seed"]) for r in rs]
    vals = [float(r["valid_count"]) for r in rs]
    colors = [
        "#D55E00" if int(r["seed"]) == TARGET_SEED else COLORS[TARGET_METHOD]
        for r in rs
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    ax.bar(seeds, vals, color=colors)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Valid heuristic count")
    ax.set_title("Dense Reward: Bi-CVRP B25 Valid Heuristics by Seed")
    fig.savefig(fig_dir / "dense_reward_bi_cvrp_B25_valid_samples_by_seed.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_setup_mean_bars(rows: list[dict], fig_dir: Path) -> None:
    x = np.arange(len(METHODS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    ax.bar(
        x - width / 2,
        [float(r["aubc_mean"]) for r in rows],
        width,
        label="AUBC",
        color="#2474A6",
    )
    ax.bar(
        x + width / 2,
        [float(r["hv_final_mean"]) for r in rows],
        width,
        label="Final HV",
        color="#E69F00",
    )
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=18, ha="right")
    ax.set_ylabel("Outer HV")
    ax.set_title("Bi-CVRP B25 Mean AUBC and Final HV by Setup")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(fig_dir / "bi_cvrp_B25_setup_mean_aubc_hv.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_focused_budget_curves(rows: list[dict], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for method in METHODS:
        xs, ys = mean_curve(rows, method)
        ax.plot(xs, ys, linewidth=2.0, color=COLORS[method],
                label=METHOD_LABELS[method])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized budget")
    ax.set_ylabel("Outer HV")
    ax.set_title("Bi-CVRP B25 Mean Budget Curves by Setup")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(fig_dir / "bi_cvrp_B25_budget_curves_by_setup.png",
                bbox_inches="tight")
    plt.close(fig)


def plot_dense_seed_curves(rows: list[dict], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for row in sorted(rows, key=lambda r: int(r["seed"])):
        xs, ys = load_curve_points(Path(row["run_dir"]))
        color = "#D55E00" if int(row["seed"]) == TARGET_SEED else "#9ECAE1"
        linewidth = 2.6 if int(row["seed"]) == TARGET_SEED else 1.6
        ax.plot(xs, ys, linewidth=linewidth, color=color,
                label=f"seed{row['seed']}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized budget")
    ax.set_ylabel("Outer HV")
    ax.set_title("Dense Reward: Bi-CVRP B25 Budget Curves by Seed")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(fig_dir / "dense_reward_bi_cvrp_B25_budget_curves_by_seed.png",
                bbox_inches="tight")
    plt.close(fig)


def write_tables(table_dir: Path, dense_rows: list[dict],
                 setup_rows: list[dict], rank_rows: list[dict]) -> None:
    dense_lines = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Seed & AUBC & Final HV & Valid & Gen. attempts & Invalid proxy \\\\",
        "\\midrule",
    ]
    for row in sorted(dense_rows, key=lambda r: int(r["seed"])):
        dense_lines.append(
            f"{row['seed']} & {fmt(row['aubc'])} & {fmt(row['hv_final'])} & "
            f"{int(row['valid_count'])} & {int(row['generation_attempts'])} & "
            f"{fmt(row['invalid_null_proxy_rate'], 3)} \\\\"
        )
    dense_lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(table_dir / "dense_reward_bi_cvrp_B25_per_seed.tex",
                dense_lines)

    setup_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Setup & AUBC mean & Final-HV mean & Valid mean & Yield mean \\\\",
        "\\midrule",
    ]
    for row in setup_rows:
        setup_lines.append(
            f"{METHOD_LABELS[row['method']]} & "
            f"{fmt(row['aubc_mean'])} & {fmt(row['hv_final_mean'])} & "
            f"{fmt(row['valid_count_mean'])} & "
            f"{fmt(row['valid_yield_per_100_calls_mean'])} \\\\"
        )
    setup_lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(table_dir / "bi_cvrp_B25_setup_means.tex", setup_lines)

    rank_lines = [
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Metric & Setup & Rank & Value \\\\",
        "\\midrule",
    ]
    for row in rank_rows:
        rank_lines.append(
            f"{row['metric']} & {METHOD_LABELS[row['method']]} & "
            f"{row['rank']} & {fmt(row['value'])} \\\\"
        )
    rank_lines += ["\\bottomrule", "\\end{tabular}"]
    write_latex(table_dir / "bi_cvrp_B25_metric_ranks.tex", rank_lines)


def run(results_root: Path, out_dir: Path) -> None:
    configure_matplotlib()
    fig_dir = ensure_dir(out_dir / "figures")
    table_dir = ensure_dir(out_dir / "tables")

    rows = enrich_rows(aggregate(str(results_root)))
    rows = [row for row in rows if row["ablation"] in METHODS]
    dense_rows = [
        row for row in rows
        if row["ablation"] == TARGET_METHOD
        and row["task"] == TARGET_TASK
        and int(row["budget"]) == TARGET_BUDGET
    ]
    target_rows = [r for r in dense_rows if int(r["seed"]) == TARGET_SEED]
    setup_rows = cell_mean_rows(rows)
    rank_rows = rank_cell_means(setup_rows)

    write_csv(out_dir / "dense_reward_bi_cvrp_B25_per_seed.csv", dense_rows)
    write_csv(out_dir / "bi_cvrp_B25_setup_means.csv", setup_rows)
    write_csv(out_dir / "bi_cvrp_B25_metric_ranks.csv", rank_rows)
    write_csv(out_dir / "target_rerun_seed2029.csv", target_rows)
    write_tables(table_dir, dense_rows, setup_rows, rank_rows)

    plot_dense_per_seed_bars(dense_rows, fig_dir)
    plot_valid_samples(dense_rows, fig_dir)
    plot_setup_mean_bars(setup_rows, fig_dir)
    plot_focused_budget_curves(rows, fig_dir)
    plot_dense_seed_curves(dense_rows, fig_dir)

    print(f"[rerun-cell] Wrote tables and figures to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default=str(_PKG_ROOT / "experiments" / "results"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(
            _PKG_ROOT
            / "experiments"
            / "results"
            / "rerun_dense_reward_bi_cvrp_B25_seed2029"
        ),
    )
    args = parser.parse_args(argv)
    run(Path(args.results_root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
