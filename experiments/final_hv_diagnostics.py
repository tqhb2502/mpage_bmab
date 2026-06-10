"""Diagnostics for terminal heuristic-population HV.

This script compares the recorded final HV against a counterfactual HV obtained
from all saved valid samples in a run. It is intentionally dependency-light and
does not import ``mpage_bmab`` so it can run even on machines where optional
benchmark dependencies are not installed.

Example:

    python -m mpage_bmab.experiments.final_hv_diagnostics \
        --summary mpage_bmab/experiments/results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence


REF_POINTS = {
    'bi_tsp': (20.0, 60.0),
    'tri_tsp': (20.0, 60.0),
    'bi_cvrp': (40.0, 60.0),
    'bi_kp': (0.0, 60.0),
}


def _valid_score(score, ref) -> bool:
    if score is None or len(score) != len(ref):
        return False
    try:
        return all(math.isfinite(float(x)) for x in score)
    except Exception:
        return False


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _hv_2d(points: Iterable[Sequence[float]], ref: Sequence[float]) -> float:
    pts = [
        (float(p[0]), float(p[1])) for p in points
        if len(p) == 2 and p[0] < ref[0] and p[1] < ref[1]
    ]
    nd = [
        p for p in pts
        if not any(_dominates(q, p) for q in pts if q is not p)
    ]
    nd.sort(key=lambda p: p[0])
    hv = 0.0
    prev_y = ref[1]
    for x, y in nd:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return max(0.0, hv)


def _fronts(points: List[Sequence[float]]) -> List[List[int]]:
    n = len(points)
    dominates_list = [[] for _ in range(n)]
    dominated_count = [0 for _ in range(n)]
    fronts = [[]]
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            if _dominates(p, q):
                dominates_list[i].append(j)
            elif _dominates(q, p):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)
    idx = 0
    while idx < len(fronts):
        nxt = []
        for i in fronts[idx]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        idx += 1
    return fronts


def _crowding(points: List[Sequence[float]], front: List[int]) -> dict:
    distances = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        return {idx: float('inf') for idx in front}
    for obj in range(len(points[0])):
        ordered = sorted(front, key=lambda idx: points[idx][obj])
        distances[ordered[0]] = float('inf')
        distances[ordered[-1]] = float('inf')
        lo = points[ordered[0]][obj]
        hi = points[ordered[-1]][obj]
        if hi == lo:
            continue
        for pos in range(1, len(ordered) - 1):
            distances[ordered[pos]] += (
                points[ordered[pos + 1]][obj]
                - points[ordered[pos - 1]][obj]
            ) / (hi - lo)
    return distances


def _managed(points: List[Sequence[float]], pop_size: int) -> List[Sequence[float]]:
    if len(points) <= pop_size:
        return points
    selected = []
    for front in _fronts(points):
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
            continue
        remaining = pop_size - len(selected)
        distances = _crowding(points, front)
        selected.extend(sorted(front, key=lambda idx: -distances[idx])[:remaining])
        break
    return [points[idx] for idx in selected]


def _pop_size_for(task: str, budget: int) -> int:
    if budget < 50:
        base = 4
    elif budget < 100:
        base = 6
    elif budget < 200:
        base = 8
    else:
        base = 10
    return base + (2 if task == 'tri_tsp' else 0)


def _sample_scores(run_dir: Path, ref) -> List[List[float]]:
    scores: List[List[float]] = []
    for sample_file in (run_dir / 'samples').glob('*.json'):
        try:
            data = json.loads(sample_file.read_text())
        except Exception:
            continue
        for item in data:
            score = item.get('score')
            if _valid_score(score, ref):
                scores.append([float(x) for x in score])
    return scores


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Final-HV archive diagnostics")
    p.add_argument('--summary',
                   default='mpage_bmab/experiments/results/summary.csv')
    p.add_argument('--out', default=None,
                   help="Output CSV path. Defaults beside --summary.")
    p.add_argument('--ablation', default='full',
                   help="Ablation/method to diagnose, or 'all'.")
    args = p.parse_args(argv)

    summary_path = Path(args.summary)
    out_path = Path(args.out) if args.out else summary_path.with_name(
        'final_hv_diagnostics.csv')

    rows = []
    with summary_path.open() as f:
        for row in csv.DictReader(f):
            if args.ablation != 'all' and row['ablation'] != args.ablation:
                continue
            task = row['task']
            ref = REF_POINTS.get(task)
            if ref is None:
                continue
            budget = int(float(row['budget']))
            official = float(row['hv_final'])
            run_dir = Path(row['run_dir'])
            scores = _sample_scores(run_dir, ref)
            archive_hv = _hv_2d(scores, ref) if scores else 0.0
            managed_hv = _hv_2d(_managed(scores, _pop_size_for(task, budget)),
                                ref) if scores else 0.0
            rows.append({
                **row,
                'valid_saved_samples': len(scores),
                'managed_archive_hv': managed_hv,
                'all_archive_hv': archive_hv,
                'managed_minus_recorded': managed_hv - official,
                'all_minus_recorded': archive_hv - official,
            })

    if not rows:
        print("[final_hv_diagnostics] No rows matched.")
        return 1
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gaps = [float(r['managed_minus_recorded']) for r in rows]
    positive = sum(g > 1e-9 for g in gaps)
    print(f"[final_hv_diagnostics] wrote {out_path}")
    print(f"[final_hv_diagnostics] rows={len(rows)} "
          f"positive_managed_gaps={positive}/{len(rows)} "
          f"mean_gap={sum(gaps)/len(gaps):.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
