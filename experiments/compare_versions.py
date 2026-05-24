"""Side-by-side comparison of two `experiments/results` trees.

Use this after archiving an old set of runs and re-running the experiments —
e.g. to confirm whether the Page-Hinkley `+ δ` fix changed AUBC at all.

The script aggregates both trees (calling ``aggregate.py`` logic), pairs
each `(ablation, task, budget, seed)` cell that exists in *both* versions,
and reports the change in AUBC and HV-final per cell, plus a per
`(ablation, task, budget)` summary across seeds.

Usage
-----

After ``mpage_bmab/experiments/archive_results.sh my_tag`` and a fresh
sweep::

    mpage_bmab/.venv/bin/python -m mpage_bmab.experiments.compare_versions \\
        --old mpage_bmab/experiments/archive/my_tag \\
        --new mpage_bmab/experiments/results

Output is a Markdown table to stdout plus an optional CSV file.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_PKG_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpage_bmab.experiments.aggregate import aggregate  # noqa: E402


def _load_or_aggregate(results_root: str) -> List[dict]:
    """Aggregate a results tree on the fly, ignoring any stale summary.csv."""
    if not os.path.isdir(results_root):
        print(f"[compare_versions] Missing root: {results_root}",
              file=sys.stderr)
        return []
    rows = aggregate(results_root)
    return rows


def _key(r: dict) -> Tuple[str, str, int, int]:
    return (r['ablation'], r['task'], r['budget'], r['seed'])


def _index(rows: List[dict]) -> Dict[Tuple[str, str, int, int], dict]:
    return {_key(r): r for r in rows}


def per_cell_diffs(old_rows: List[dict], new_rows: List[dict]
                   ) -> List[dict]:
    """Return one record per (ablation, task, budget, seed) present in BOTH."""
    old_idx = _index(old_rows)
    new_idx = _index(new_rows)
    shared = sorted(set(old_idx.keys()) & set(new_idx.keys()))
    diffs = []
    for key in shared:
        o, n = old_idx[key], new_idx[key]
        ablation, task, budget, seed = key
        diffs.append({
            'ablation': ablation,
            'task': task,
            'budget': budget,
            'seed': seed,
            'aubc_old': o.get('aubc', 0.0),
            'aubc_new': n.get('aubc', 0.0),
            'aubc_delta': n.get('aubc', 0.0) - o.get('aubc', 0.0),
            'hv_final_old': o.get('hv_final', 0.0),
            'hv_final_new': n.get('hv_final', 0.0),
            'hv_final_delta': n.get('hv_final', 0.0) - o.get('hv_final', 0.0),
        })
    return diffs


def per_group_summary(diffs: List[dict]
                      ) -> List[dict]:
    """Aggregate per-cell diffs into per (ablation, task, budget) means."""
    bucket: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    for d in diffs:
        bucket[(d['ablation'], d['task'], d['budget'])].append(d)
    out = []
    for (ablation, task, budget), rs in sorted(bucket.items()):
        n = len(rs)

        def _mean(field: str) -> float:
            return sum(r[field] for r in rs) / n if n else 0.0

        def _std(field: str) -> float:
            if n < 2:
                return 0.0
            m = _mean(field)
            return math.sqrt(sum((r[field] - m) ** 2 for r in rs) / n)

        out.append({
            'ablation': ablation,
            'task': task,
            'budget': budget,
            'n_seeds': n,
            'aubc_old_mean': _mean('aubc_old'),
            'aubc_new_mean': _mean('aubc_new'),
            'aubc_delta_mean': _mean('aubc_delta'),
            'aubc_delta_std': _std('aubc_delta'),
            'hv_old_mean': _mean('hv_final_old'),
            'hv_new_mean': _mean('hv_final_new'),
            'hv_delta_mean': _mean('hv_final_delta'),
        })
    return out


# --------------------------------------------------------------------------- #
# Markdown printing
# --------------------------------------------------------------------------- #

def _print_md(summary: List[dict], n_old: int, n_new: int,
              n_shared: int) -> None:
    print(f"# Cross-version comparison\n")
    print(f"* Old version: **{n_old} runs**")
    print(f"* New version: **{n_new} runs**")
    print(f"* Cells present in both: **{n_shared}**\n")
    if not summary:
        print("_No cells matched between the two versions._")
        return

    print("## Per-group means")
    print()
    print("| ablation | task | B | n | AUBC old | AUBC new | ΔAUBC ± std | "
          "HV old | HV new | ΔHV |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summary:
        print(
            f"| {s['ablation']} | {s['task']} | {s['budget']} | {s['n_seeds']} | "
            f"{s['aubc_old_mean']:.2f} | {s['aubc_new_mean']:.2f} | "
            f"{s['aubc_delta_mean']:+.2f} ± {s['aubc_delta_std']:.2f} | "
            f"{s['hv_old_mean']:.2f} | {s['hv_new_mean']:.2f} | "
            f"{s['hv_delta_mean']:+.2f} |"
        )
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Side-by-side comparison of two experiment-result trees.")
    p.add_argument('--old', required=True,
                   help="Path to the *old* experiments/results-like folder "
                        "(usually under experiments/archive/<tag>).")
    p.add_argument('--new', required=True,
                   help="Path to the *new* experiments/results folder.")
    p.add_argument('--out_cells', default=None,
                   help="Optional CSV path for per-cell diffs.")
    p.add_argument('--out_summary', default=None,
                   help="Optional CSV path for the per-group summary.")
    args = p.parse_args(argv)

    old_rows = _load_or_aggregate(args.old)
    new_rows = _load_or_aggregate(args.new)
    diffs = per_cell_diffs(old_rows, new_rows)
    summary = per_group_summary(diffs)
    _print_md(summary, len(old_rows), len(new_rows), len(diffs))

    if args.out_cells and diffs:
        cols = list(diffs[0].keys())
        with open(args.out_cells, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for d in diffs:
                w.writerow(d)
        print(f"[compare_versions] Wrote per-cell CSV: {args.out_cells}",
              file=sys.stderr)
    if args.out_summary and summary:
        cols = list(summary[0].keys())
        with open(args.out_summary, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s in summary:
                w.writerow(s)
        print(f"[compare_versions] Wrote per-group CSV: {args.out_summary}",
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
