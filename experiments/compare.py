"""Statistical comparison of ablations on AUBC.

Runs Wilcoxon signed-rank tests pairing the ``--baseline`` ablation against
every other ablation, **per (task, budget)** cell, using AUBC across seeds
as the paired sample.

Usage::

    python -m mpage_bmab.experiments.compare
    python -m mpage_bmab.experiments.compare --baseline full --metric aubc
    python -m mpage_bmab.experiments.compare --metric hv_final

By default reads ``experiments/results/summary.csv`` (run ``aggregate.py``
first) and writes a Markdown report to stdout plus a CSV alongside the
summary file.
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


try:
    from scipy.stats import wilcoxon  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _read_summary(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    'ablation': row['ablation'],
                    'task': row['task'],
                    'budget': int(row['budget']),
                    'seed': int(row['seed']),
                    'aubc': float(row['aubc']),
                    'hv_final': float(row['hv_final']),
                })
            except (ValueError, KeyError):
                continue
    return rows


def _index(rows: List[dict], metric: str
           ) -> Dict[Tuple[str, str, int], Dict[int, float]]:
    """Index rows by (ablation, task, budget) -> {seed: metric}."""
    out: Dict[Tuple[str, str, int], Dict[int, float]] = defaultdict(dict)
    for r in rows:
        out[(r['ablation'], r['task'], r['budget'])][r['seed']] = r[metric]
    return out


def _wilcoxon(a: List[float], b: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (statistic, p_value). Falls back to None when scipy is missing
    or fewer than 3 paired non-equal observations."""
    if not _HAVE_SCIPY:
        return None, None
    diffs = [x - y for x, y in zip(a, b) if not math.isnan(x - y)]
    diffs = [d for d in diffs if d != 0.0]
    if len(diffs) < 3:
        return None, None
    try:
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def compare(rows: List[dict], baseline: str, metric: str
            ) -> List[dict]:
    """Return a list of comparison records.

    Pairs ``baseline`` against every other ablation it shares
    ``(task, budget, seed)`` cells with."""
    idx = _index(rows, metric)

    cells = sorted({(t, b) for (_, t, b) in idx.keys()})
    others = sorted({a for (a, _, _) in idx.keys() if a != baseline})

    results: List[dict] = []
    for task, budget in cells:
        bkey = (baseline, task, budget)
        if bkey not in idx:
            continue
        bdict = idx[bkey]
        for other in others:
            okey = (other, task, budget)
            if okey not in idx:
                continue
            odict = idx[okey]
            shared = sorted(set(bdict.keys()) & set(odict.keys()))
            if not shared:
                continue
            a = [bdict[s] for s in shared]
            b = [odict[s] for s in shared]
            n = len(shared)
            mean_a = sum(a) / n
            mean_b = sum(b) / n
            stat, pval = _wilcoxon(a, b)
            results.append({
                'task': task,
                'budget': budget,
                'baseline': baseline,
                'other': other,
                'metric': metric,
                'n_seeds': n,
                'baseline_mean': mean_a,
                'other_mean': mean_b,
                'delta': mean_a - mean_b,
                'wilcoxon_stat': stat,
                'p_value': pval,
            })
    return results


def _print_md(results: List[dict], baseline: str, metric: str) -> None:
    print(f"\n# Wilcoxon signed-rank: {baseline} vs others (metric: {metric})\n")
    if not results:
        print("_No comparable cells found. Run aggregate.py first._\n")
        return
    if not _HAVE_SCIPY:
        print("> ⚠ scipy is not available — p-values are blank. "
              "`pip install scipy` to enable significance tests.\n")
    cells = sorted({(r['task'], r['budget']) for r in results})
    for task, budget in cells:
        print(f"## task={task}  B={budget}\n")
        print(f"| other | n | {baseline} mean | other mean | Δ ({baseline}−other) | "
              f"Wilcoxon W | p |")
        print(f"|-------|---|----------------|------------|----------------------|"
              f"------------|---|")
        for r in [x for x in results if x['task'] == task and x['budget'] == budget]:
            stat = "" if r['wilcoxon_stat'] is None else f"{r['wilcoxon_stat']:.2f}"
            pv = "" if r['p_value'] is None else f"{r['p_value']:.4f}"
            print(f"| {r['other']} | {r['n_seeds']} | "
                  f"{r['baseline_mean']:.4f} | {r['other_mean']:.4f} | "
                  f"{r['delta']:+.4f} | {stat} | {pv} |")
        print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Wilcoxon comparisons of AUBC/HV.")
    default_summary = os.path.join(_PKG_ROOT, 'experiments',
                                   'results', 'summary.csv')
    p.add_argument('--summary', default=default_summary,
                   help="CSV produced by aggregate.py.")
    p.add_argument('--out', default=None,
                   help="Where to write the comparisons CSV "
                        "(defaults to comparisons_<metric>.csv next to summary).")
    p.add_argument('--baseline', default='full',
                   help="Ablation to use as the reference (default: full).")
    p.add_argument('--metric', default='aubc', choices=['aubc', 'hv_final'])
    args = p.parse_args(argv)

    if not os.path.isfile(args.summary):
        print(f"[compare] Missing summary CSV: {args.summary}\n"
              f"[compare] Run `python -m mpage_bmab.experiments.aggregate` first.",
              file=sys.stderr)
        return 2

    rows = _read_summary(args.summary)
    if not rows:
        print(f"[compare] Summary {args.summary} is empty.", file=sys.stderr)
        return 1

    results = compare(rows, baseline=args.baseline, metric=args.metric)
    _print_md(results, baseline=args.baseline, metric=args.metric)

    out_path = args.out or os.path.join(
        os.path.dirname(args.summary), f"comparisons_{args.metric}.csv")
    if results:
        cols = list(results[0].keys())
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"[compare] Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
