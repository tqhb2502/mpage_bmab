"""Aggregate per-run JSON artefacts into a single results CSV.

Walks ``<results_root>/<ablation>/<task>/B<budget>/seed<seed>/*/`` and pulls:

* ``aubc.json``        — total/consumed budget + AUBC scalar
* ``budget_curve.json`` — last (consumed_budget, hv, pareto_size) record
* ``budget_history.json`` — number of LLM calls actually issued

Run::

    python -m mpage_bmab.experiments.aggregate
    python -m mpage_bmab.experiments.aggregate --out ./summary.csv

Output columns::

    ablation, task, budget, seed, aubc, hv_final,
    pareto_size, consumed_budget, n_calls, run_dir
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from glob import glob
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)


_RUN_PAT = re.compile(
    r"(?P<ablation>[^/]+)/(?P<task>[^/]+)/B(?P<budget>\d+)/seed(?P<seed>\d+)"
)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _harvest_one(run_dir: str) -> Optional[Dict]:
    """Return a row of metrics, or None when the run is incomplete."""
    aubc_path = os.path.join(run_dir, 'aubc.json')
    curve_path = os.path.join(run_dir, 'budget_curve.json')
    hist_path = os.path.join(run_dir, 'budget_history.json')

    aubc = _read_json(aubc_path)
    if aubc is None:
        return None
    curve = _read_json(curve_path) or []
    hist = _read_json(hist_path) or []

    if curve:
        hv_final = curve[-1].get('hv', 0.0)
        pareto_size = curve[-1].get('pareto_size', 0)
    else:
        hv_final, pareto_size = 0.0, 0

    return {
        'aubc': aubc.get('aubc', 0.0),
        'consumed_budget': aubc.get('consumed_budget', 0.0),
        'total_budget': aubc.get('total_budget', 0.0),
        'hv_final': hv_final,
        'pareto_size': pareto_size,
        'n_calls': len(hist),
        'run_dir': run_dir,
    }


def aggregate(results_root: str) -> List[Dict]:
    rows: List[Dict] = []
    pattern = os.path.join(results_root, '*', '*', 'B*', 'seed*', '*',
                           'aubc.json')
    for aubc_path in sorted(glob(pattern)):
        run_dir = os.path.dirname(aubc_path)
        rel = os.path.relpath(run_dir, results_root)
        # rel = "<ablation>/<task>/B<budget>/seed<seed>/<timestamp_dir>"
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        ablation, task, budget_dir, seed_dir = parts[0], parts[1], parts[2], parts[3]
        try:
            budget = int(budget_dir.lstrip('B'))
            seed = int(seed_dir.lstrip('seed'))
        except ValueError:
            continue
        metrics = _harvest_one(run_dir)
        if metrics is None:
            continue
        rows.append({
            'ablation': ablation,
            'task': task,
            'budget': budget,
            'seed': seed,
            **metrics,
        })
    return rows


def write_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        print(f"[aggregate] No rows found.", file=sys.stderr)
        return
    cols = ['ablation', 'task', 'budget', 'seed',
            'aubc', 'hv_final', 'pareto_size',
            'consumed_budget', 'total_budget', 'n_calls', 'run_dir']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, '') for c in cols})


def _print_summary(rows: List[Dict]) -> None:
    """Mean ± std of AUBC and HV per (ablation, task, budget) — quick eyeball."""
    from collections import defaultdict
    import math

    bucket = defaultdict(list)
    for r in rows:
        bucket[(r['ablation'], r['task'], r['budget'])].append(r)
    print(f"\n{'ablation':<14}{'task':<10}{'B':>5}  "
          f"{'n':>3}  {'AUBC mean':>10}  {'AUBC std':>9}  "
          f"{'HV mean':>9}  {'HV std':>8}")
    print("-" * 80)
    for key in sorted(bucket.keys()):
        rs = bucket[key]
        n = len(rs)
        aubcs = [r['aubc'] for r in rs]
        hvs = [r['hv_final'] for r in rs]
        a_mean = sum(aubcs) / n
        h_mean = sum(hvs) / n
        a_std = math.sqrt(sum((x - a_mean) ** 2 for x in aubcs) / n)
        h_std = math.sqrt(sum((x - h_mean) ** 2 for x in hvs) / n)
        ablation, task, budget = key
        print(f"{ablation:<14}{task:<10}{budget:>5}  {n:>3}  "
              f"{a_mean:>10.4f}  {a_std:>9.4f}  "
              f"{h_mean:>9.4f}  {h_std:>8.4f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aggregate experiment results")
    p.add_argument('--results_root',
                   default=os.path.join(_PKG_ROOT, 'experiments', 'results'))
    p.add_argument('--out',
                   default=os.path.join(_PKG_ROOT, 'experiments',
                                        'results', 'summary.csv'))
    p.add_argument('--no_print', action='store_true')
    args = p.parse_args(argv)

    rows = aggregate(args.results_root)
    print(f"[aggregate] Found {len(rows)} runs under {args.results_root}.")
    write_csv(rows, args.out)
    print(f"[aggregate] Wrote {args.out}")
    if not args.no_print:
        _print_summary(rows)
    return 0


if __name__ == '__main__':
    sys.exit(main())
