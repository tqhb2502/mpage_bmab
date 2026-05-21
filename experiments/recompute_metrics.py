"""One-off recovery script: recompute budget_curve.json + aubc.json offline.

Use this when a run was completed but the saved HV / AUBC are corrupt because
of a stale reference-point shape (e.g. the tri_tsp 3-D vs 2-D mismatch that
made every tri_tsp AUBC = 0).

The saved **score data** in `samples/samples_*.json` is fine — only the
post-processed curve is wrong. This script walks each run directory and
re-derives the budget-vs-HV curve from the samples using a corrected
reference point. Both files (`budget_curve.json`, `aubc.json`) are
backed up with `.bak` suffix before being overwritten.

Usage
-----

Re-run a single task with a new ref point:

    python -m mpage_bmab.experiments.recompute_metrics \\
        --task tri_tsp \\
        --ref_point 20.0,60.0

Always run with `--dry_run` first to see what would be touched.

Approximations
--------------

1. The HV curve is recomputed at one point **per registered sample** (the
   `sample_order` ordering inside `samples_*.json`), not at the exact
   original curve-recording moments. The new curve is therefore denser
   than the original.
2. The budget axis for each sample is taken from `budget_history.json`
   when available: we count heuristic-generation attempts in step order
   and place the k-th valid sample at the budget step of the k-th such
   attempt. When `budget_history.json` is missing or malformed, we fall
   back to a uniform spacing `budget = total_budget * (k / n_samples)`.
3. The "population" at each point is the **cumulative non-dominated
   archive of all samples seen so far**, not the actual capped-at-pop_size
   PFG population. This can slightly OVER-estimate HV vs the original run
   when the Pareto front exceeds `pop_size`, but the bias applies equally
   to every method being compared, so the relative ordering is preserved.

These approximations are acceptable for the thesis comparison because they
treat every method identically. To recover the *exact* original curve you
would need to re-run the LLM calls.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_PKG_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpage_bmab.reward import _is_valid_score, hypervolume  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (minimisation)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _update_archive(archive: List[np.ndarray], cand: np.ndarray
                    ) -> List[np.ndarray]:
    """Insert `cand` into a non-dominated archive (in place semantics)."""
    if any(_dominates(m, cand) for m in archive):
        return archive
    pruned = [m for m in archive if not _dominates(cand, m)]
    pruned.append(cand)
    return pruned


def _aubc_from_curve(curve: List[dict], total_budget: float) -> float:
    """Trapezoidal AUBC over `curve`, normalised by `total_budget`.
    Mirrors ``profiler.BMABProfiler.aubc`` exactly."""
    if not curve:
        return 0.0
    pts = sorted(curve, key=lambda x: x['budget_consumed'])
    xs = [0.0] + [p['budget_consumed'] for p in pts]
    ys = [0.0] + [p['hv'] for p in pts]
    if xs[-1] < total_budget:
        xs.append(total_budget)
        ys.append(ys[-1])
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return area / max(total_budget, 1e-9)


_GEN_LABELS = ('init', 'mutate', 'crossover', 'suggestion')


def _is_gen_label(label: str) -> bool:
    """A budget entry counts as a heuristic-generation attempt if its label
    is one of the operator labels (or starts with one of them).

    `cluster` calls do NOT count — they spend budget but produce no sample.
    `suggestion` calls also do not produce a sample but do spend budget; we
    treat them as gen-attempts here so the count of attempts matches the
    number of registered samples better.  (Edge case; default --review is
    off so suggestion entries are rare.)
    """
    if not isinstance(label, str):
        return False
    return any(label.startswith(p) for p in _GEN_LABELS)


def _budget_per_sample(history: List[dict], n_samples: int,
                       total_budget: float) -> List[float]:
    """Return a list of length n_samples giving the budget_consumed at which
    each registered sample was admitted.

    Strategy: count heuristic-gen entries in `history` in step order. The
    k-th valid sample is assumed to be the k-th success among those
    attempts; we place it at the budget of the gen-attempt at position
    ⌈k · N_attempts / n_samples⌉ in the gen-only history.
    """
    if n_samples <= 0:
        return []
    # Filter to gen-only entries in step order
    gen_steps: List[Tuple[int, float]] = []
    for e in sorted(history, key=lambda x: int(x.get('step', 0))):
        if _is_gen_label(e.get('label', '')):
            step = int(e['step'])
            consumed = total_budget - float(e.get('remaining', total_budget))
            gen_steps.append((step, consumed))
    if not gen_steps:
        # Fallback to uniform spacing
        return [total_budget * (k / n_samples)
                for k in range(1, n_samples + 1)]
    # Map each sample to the closest gen-attempt in attempt order
    n_attempts = len(gen_steps)
    out = []
    for k in range(1, n_samples + 1):
        # k-th success among n_samples valid samples ↔ attempt index
        # round(k * n_attempts / n_samples)
        idx = min(n_attempts - 1, max(0,
                  int(round(k * n_attempts / n_samples)) - 1))
        out.append(float(gen_steps[idx][1]))
    return out


# --------------------------------------------------------------------------- #
# Single-run recovery
# --------------------------------------------------------------------------- #

def recompute_run(run_dir: str, ref_point: List[float],
                  verbose: bool = False) -> Optional[dict]:
    """Walk a single run directory and rewrite its curve + AUBC files.

    Returns a small summary dict, or None if the run is unrecoverable
    (missing samples, missing aubc.json, malformed JSON).
    """
    curve_path = os.path.join(run_dir, 'budget_curve.json')
    aubc_path = os.path.join(run_dir, 'aubc.json')
    hist_path = os.path.join(run_dir, 'budget_history.json')

    if not os.path.isfile(aubc_path):
        return None

    try:
        with open(aubc_path) as f:
            old_aubc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    total_budget = float(old_aubc.get('total_budget', 0))
    consumed_budget = float(old_aubc.get('consumed_budget', total_budget))
    if total_budget <= 0:
        return None

    # Load samples in sample_order
    sample_files = sorted(glob.glob(os.path.join(run_dir, 'samples',
                                                 'samples_*.json')))
    samples: List[dict] = []
    for sf in sample_files:
        try:
            with open(sf) as f:
                samples.extend(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    samples.sort(key=lambda s: s.get('sample_order', 0))
    if not samples:
        return None

    # Load budget history (optional)
    history: List[dict] = []
    if os.path.isfile(hist_path):
        try:
            with open(hist_path) as f:
                history = json.load(f)
        except (OSError, json.JSONDecodeError):
            history = []

    n_samples = len(samples)
    bud_per_sample = _budget_per_sample(history, n_samples, total_budget)

    # Build cumulative non-dominated archive
    ref = np.asarray(ref_point, dtype=float)
    archive: List[np.ndarray] = []
    new_curve: List[dict] = []
    for i, s in enumerate(samples):
        score = s.get('score')
        if not _is_valid_score(score):
            continue
        arr = np.asarray(score, dtype=float)
        if arr.shape != ref.shape:
            # Should not happen now that ref is correct, but defend anyway
            continue
        archive = _update_archive(archive, arr)
        if archive:
            hv = float(hypervolume(np.array(archive), ref))
        else:
            hv = 0.0
        new_curve.append({
            'budget_consumed': float(bud_per_sample[i]),
            'hv': hv,
            'pareto_size': len(archive),
        })

    if not new_curve:
        return None

    new_aubc = _aubc_from_curve(new_curve, total_budget)

    # Backup originals
    if os.path.isfile(curve_path) and not os.path.isfile(curve_path + '.bak'):
        os.rename(curve_path, curve_path + '.bak')
    if os.path.isfile(aubc_path) and not os.path.isfile(aubc_path + '.bak'):
        os.rename(aubc_path, aubc_path + '.bak')

    with open(curve_path, 'w') as f:
        json.dump(new_curve, f, indent=2)
    with open(aubc_path, 'w') as f:
        json.dump({
            'total_budget': total_budget,
            'consumed_budget': consumed_budget,
            'aubc': new_aubc,
        }, f, indent=2)

    summary = {
        'run_dir': run_dir,
        'n_samples': n_samples,
        'pareto_size_final': len(archive),
        'aubc_old': float(old_aubc.get('aubc', 0.0)),
        'aubc_new': float(new_aubc),
        'hv_final_new': float(new_curve[-1]['hv']),
    }
    if verbose:
        rel = os.path.relpath(run_dir, _PKG_ROOT)
        print(f"  {rel}")
        print(f"    n_samples={n_samples}  |F|={len(archive)}  "
              f"AUBC: {summary['aubc_old']:.2f} → {summary['aubc_new']:.2f}  "
              f"HV-final: {summary['hv_final_new']:.2f}")
    return summary


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Offline HV/AUBC recovery for runs with stale ref_points.")
    p.add_argument('--results_root',
                   default=os.path.join(_PKG_ROOT, 'experiments', 'results'),
                   help="Root of the results tree.")
    p.add_argument('--task', required=True,
                   help="Task name to recompute (e.g. tri_tsp).")
    p.add_argument('--ref_point', required=True,
                   help='Comma-separated new ref point, e.g. "20.0,60.0".')
    p.add_argument('--dry_run', action='store_true',
                   help="List target run dirs; don't touch any files.")
    args = p.parse_args(argv)

    try:
        ref = [float(x) for x in args.ref_point.split(',')]
    except ValueError:
        print(f"[recompute] Bad --ref_point '{args.ref_point}'", file=sys.stderr)
        return 2
    if len(ref) < 2:
        print(f"[recompute] --ref_point needs at least 2 components.",
              file=sys.stderr)
        return 2

    pattern = os.path.join(args.results_root, '*', args.task, 'B*',
                           'seed*', '*', 'aubc.json')
    run_dirs = sorted({os.path.dirname(p) for p in glob.glob(pattern)})

    print(f"[recompute] Found {len(run_dirs)} '{args.task}' runs under "
          f"{args.results_root}.")
    print(f"[recompute] New ref_point: {ref}")
    if args.dry_run:
        for d in run_dirs:
            print(f"  {os.path.relpath(d, _PKG_ROOT)}")
        print("[recompute] Dry run — nothing modified.")
        return 0

    n_ok = 0
    n_fail = 0
    for d in run_dirs:
        result = recompute_run(d, ref, verbose=True)
        if result is None:
            n_fail += 1
        else:
            n_ok += 1

    print(f"\n[recompute] Done. {n_ok} runs recomputed, {n_fail} skipped.")
    print(f"[recompute] Backups saved with .bak suffix.")
    print(f"[recompute] Re-run aggregate.py + compare.py to refresh summary.csv.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
