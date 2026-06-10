# MPaGE vs mpage_bmab Experimental Analysis

This report analyzes the runs under `mpage_bmab/experiments/results`. The comparison is between the original MPaGE workflow, represented by the `mpage_orig` wrapper, and the historical `full` mpage_bmab method that produced this result set. In this report, `full` is called **mpage_bmab**.

**Important version note.** These results were generated before the later final-HV-oriented implementation fixes. In the current code, `full` means the fixed method with `reward_mode=final_hv`; `dense_reward` means the fixed method with the legacy immediate-HVI reward. Therefore, the numbers in this report should be cited as the historical/pre-fix `full` result set, not as evidence for the new fixed `full`, `dense_reward`, or `hybrid_reward` methods.

## Experimental Setup

The result directory contains **160 runs**: 2 methods x 4 tasks x 4 budgets x 5 seeds.

- **Methods:** `mpage_orig` and `full` (`mpage_bmab`).
- **Tasks:** Bi-TSP, Tri-TSP, Bi-CVRP, and Bi-KP.
- **Budgets:** B = 25, 50, 100, and 200 LLM calls.
- **Seeds:** 2025, 2026, 2027, 2028, and 2029.
- **Primary metric:** AUBC, the area under the budget-vs-hypervolume curve. Higher AUBC means the method obtains useful heuristic-population hypervolume earlier under the same LLM-call budget.
- **Secondary metric:** final heuristic-population HV at the end of the budget. Higher is better.

The HV values here are **algorithm-level heuristic-population hypervolumes**, not the normalized solution-level HV/IGD values reported in the MPaGE paper tables. They are therefore appropriate for comparing these runs against each other, but not for direct numeric comparison with the paper's benchmark tables.

All runs consumed their configured total budget. The `mpage_orig` runs store budget history as one aggregate entry, so `n_calls` in `summary.csv` is not directly comparable to the per-call histories saved by mpage_bmab.

## Generated Figures

- [AUBC mean by budget](images/aubc_mean_by_budget.png)
- [Final HV mean by budget](images/hv_final_mean_by_budget.png)
- [AUBC relative-improvement heatmap](images/aubc_delta_percent_heatmap.png)
- [Final-HV relative-improvement heatmap](images/hv_final_delta_percent_heatmap.png)
- [Per-seed AUBC delta boxplot](images/paired_aubc_delta_boxplot.png)
- [Budget curves at B=50](images/budget_curves_B50.png)
- [AUBC/final-HV ratio](images/aubc_to_final_hv_ratio.png)
- [Valid heuristic yield by budget](images/valid_yield_by_budget.png)
- [Invalid/null-score rate proxies](images/invalid_null_rate_by_budget.png)
- [Valid-yield relative-improvement heatmap](images/valid_yield_delta_percent_heatmap.png)

## Key Findings

### 1. mpage_bmab is clearly stronger on budget efficiency

mpage_bmab has higher mean AUBC in **15/16** task-budget cells, with an average relative AUBC improvement of **+26.1%** over MPaGE-orig. At the paired run level, mpage_bmab wins **71/80** AUBC comparisons.

This is visible in the AUBC line chart and the AUBC heatmap. The gains are especially large on Bi-CVRP and Tri-TSP, where the adaptive method gets substantially more value from the same LLM-call budget. The B=50 budget curves also show that mpage_bmab often reaches a stronger heuristic-population HV earlier in the run, which directly explains the AUBC advantage.

| Task | AUBC positive cells | Mean AUBC delta % | HV-positive cells | Mean final-HV delta % |
|---|---:|---:|---:|---:|
| Bi-TSP | 4/4 | +13.1% | 2/4 | -0.4% |
| Tri-TSP | 4/4 | +30.1% | 2/4 | -0.0% |
| Bi-CVRP | 4/4 | +45.9% | 3/4 | +5.1% |
| Bi-KP | 3/4 | +15.3% | 3/4 | -1.2% |

### 2. Final HV is mixed, so the main contribution is not consistently better terminal quality

Final HV improves in only **10/16** cells, and the average final-HV relative change is **+0.9%**. At the paired run level, mpage_bmab wins **38/80** final-HV comparisons.

The final-HV line chart and final-HV heatmap show a much less consistent picture than AUBC. This means the strongest evidence for mpage_bmab is budget efficiency: it tends to find useful heuristics earlier, while the final population after the full budget is often close to MPaGE-orig.

### 3. Null-score and yield metrics add useful diagnostic information

The number of null-score heuristics is worth tracking because it measures wasted heuristic-generation effort under a fixed LLM-call budget. The artifacts show a strong difference in logging behavior:

- MPaGE-orig saved **1,625** generated heuristic samples: **986** valid and **639** with `score: null`, a **39.3% recorded null-score rate**.
- mpage_bmab saved **1,091** scored samples and **0** saved `score: null` samples.
- However, mpage_bmab only persists valid scored samples. Its invalid or failed generation attempts must be inferred from `budget_history.json`, not counted from `samples/*.json`.

For this reason, the report uses two related diagnostics:

- **Valid heuristic yield per budget:** valid scored heuristics divided by total LLM-call budget. This is comparable across methods.
- **Invalid/null proxy rate:** for MPaGE-orig, saved `score: null` entries divided by saved generated samples; for mpage_bmab, charged generation calls that did not produce a saved valid sample divided by charged generation calls. This second metric is broader for mpage_bmab, so it should be interpreted as a failure/attrition proxy, not as an exact null-score rate.

Overall valid yield is **14.5 valid heuristics per 100 calls** for mpage_bmab versus **13.1 per 100 calls** for MPaGE-orig. mpage_bmab has higher valid yield in **10/16** task-budget cells. The valid-yield figures show that this advantage is modest and task-dependent, so it cannot by itself explain the much larger AUBC gains.

The strongest counterexample is Bi-CVRP: mpage_bmab has lower valid yield than MPaGE-orig at every budget, yet it has much higher AUBC. This means the AUBC gain is not simply because mpage_bmab produces more valid heuristics; the valid heuristics it does obtain are better timed, better selected, or more useful for the heuristic-population front.

### 4. Statistical evidence is suggestive but limited by only five seeds

No cell reaches conventional two-sided Wilcoxon significance at p < 0.05. This is expected with n=5 paired seeds: when all five seeds favor one method, the two-sided Wilcoxon p-value bottoms out at 0.0625 in this setup.

For AUBC, many cells are directionally strong and practically meaningful: Bi-TSP B=25 (p=0.0625), Bi-TSP B=50 (p=0.0625), Bi-TSP B=100 (p=0.0625), Bi-TSP B=200 (p=0.0625), Tri-TSP B=25 (p=0.0625), Tri-TSP B=50 (p=0.1250), Tri-TSP B=100 (p=0.0625), Tri-TSP B=200 (p=0.0625), Bi-CVRP B=25 (p=0.0625), Bi-CVRP B=50 (p=0.0625), Bi-CVRP B=100 (p=0.0625), Bi-CVRP B=200 (p=0.1250). These cells should be described as **consistent and practically meaningful**, not formally significant at p < 0.05.

### 5. The strongest gains occur when operator/cluster adaptivity can exploit early feedback

The result pattern is consistent with the project design. MPaGE-orig follows the paper's fixed evolutionary schedule, while mpage_bmab adapts operator and cluster choices using reward feedback under a hard budget. That helps most when early choices matter: small and medium budgets, and tasks where poor generated heuristics waste substantial budget.

Bi-CVRP shows the clearest practical gain: mpage_bmab improves AUBC by **+59.5%** at B=25 and **+77.9%** at B=50. Tri-TSP also benefits substantially, with AUBC gains from **+16.2%** to **+65.0%** across budgets. Bi-TSP gains are consistent but smaller as budget increases, suggesting that both methods converge to similar final heuristic quality on the easier/smaller TSP setting.

### 6. Bi-KP is the least stable task

Bi-KP is the only task-budget cell where mpage_bmab has lower mean AUBC: B=25, with **-1.3%** relative AUBC. It then becomes positive at larger budgets, but with high variance. The per-seed AUBC boxplot shows Bi-KP has the widest spread among the bi-objective tasks.

A likely explanation is that Bi-KP generated heuristics are brittle under very small budgets: a few poor or invalid early heuristics can dominate AUBC before the bandit has enough observations to learn useful operator/cluster preferences. This is consistent with the small-B setting, where warm starts and priors have more influence than observed rewards.

### 7. The previous Tri-TSP B=25 zero-run issue is resolved

After rerunning mpage_bmab Tri-TSP B=25 seed=2027, the current row is no longer a zero-result failure. The rerun produced **AUBC = 144,905.6** and **final HV = 207,008.1** for that seed. Current zero-row check: No current row has zero AUBC or zero final HV.

This materially changes the Tri-TSP B=25 interpretation: the cell now shows **+65.0%** mean AUBC improvement over MPaGE-orig, with **5/5** paired AUBC seed wins and Wilcoxon p = **0.0625**. Final HV remains essentially tied for the cell, with a small mean difference of **-294.3**.

## Detailed AUBC Results

| Task | B | mpage_bmab mean | MPaGE-orig mean | Delta | Delta % | Seed wins | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bi-TSP | 25 | 12,750.8 | 9,841.9 | 2,908.9 | +29.6% | 5/5 | 0.0625 |
| Bi-TSP | 50 | 12,890.6 | 11,807.6 | 1,083.0 | +9.2% | 5/5 | 0.0625 |
| Bi-TSP | 100 | 13,748.1 | 12,956.3 | 791.8 | +6.1% | 5/5 | 0.0625 |
| Bi-TSP | 200 | 14,471.8 | 13,462.6 | 1,009.2 | +7.5% | 5/5 | 0.0625 |
| Tri-TSP | 25 | 144,482.9 | 87,564.7 | 56,918.2 | +65.0% | 5/5 | 0.0625 |
| Tri-TSP | 50 | 159,646.0 | 137,434.6 | 22,211.4 | +16.2% | 4/5 | 0.1250 |
| Tri-TSP | 100 | 184,626.5 | 152,646.7 | 31,979.7 | +21.0% | 5/5 | 0.0625 |
| Tri-TSP | 200 | 195,125.3 | 165,126.2 | 29,999.1 | +18.2% | 5/5 | 0.0625 |
| Bi-CVRP | 25 | 11,679.8 | 7,322.7 | 4,357.1 | +59.5% | 5/5 | 0.0625 |
| Bi-CVRP | 50 | 13,318.3 | 7,486.2 | 5,832.1 | +77.9% | 5/5 | 0.0625 |
| Bi-CVRP | 100 | 15,059.8 | 12,211.7 | 2,848.1 | +23.3% | 5/5 | 0.0625 |
| Bi-CVRP | 200 | 16,372.1 | 13,339.4 | 3,032.8 | +22.7% | 4/5 | 0.1250 |
| Bi-KP | 25 | 16,007.4 | 16,217.4 | -210.0 | -1.3% | 3/5 | 1.0000 |
| Bi-KP | 50 | 26,573.3 | 21,096.1 | 5,477.2 | +26.0% | 3/5 | 0.3125 |
| Bi-KP | 100 | 27,489.8 | 23,046.1 | 4,443.7 | +19.3% | 4/5 | 0.6250 |
| Bi-KP | 200 | 29,643.5 | 25,270.8 | 4,372.7 | +17.3% | 3/5 | 0.4375 |

## Detailed Final-HV Results

| Task | B | mpage_bmab mean | MPaGE-orig mean | Delta | Delta % | Seed wins | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bi-TSP | 25 | 14,764.5 | 14,750.8 | 13.7 | +0.1% | 3/5 | 0.8125 |
| Bi-TSP | 50 | 14,558.5 | 14,911.4 | -353.0 | -2.4% | 0/5 | 0.0625 |
| Bi-TSP | 100 | 15,061.3 | 15,096.6 | -35.3 | -0.2% | 2/5 | 0.4375 |
| Bi-TSP | 200 | 15,174.9 | 15,016.3 | 158.6 | +1.1% | 3/5 | 0.4375 |
| Tri-TSP | 25 | 206,382.3 | 206,676.6 | -294.3 | -0.1% | 3/5 | 1.0000 |
| Tri-TSP | 50 | 207,330.8 | 206,978.3 | 352.4 | +0.2% | 3/5 | 0.6250 |
| Tri-TSP | 100 | 206,978.0 | 207,362.1 | -384.1 | -0.2% | 2/5 | 0.3125 |
| Tri-TSP | 200 | 207,247.8 | 207,182.3 | 65.5 | +0.0% | 3/5 | 1.0000 |
| Bi-CVRP | 25 | 16,264.4 | 14,386.8 | 1,877.6 | +13.1% | 1/5 | 0.6250 |
| Bi-CVRP | 50 | 17,393.6 | 16,246.9 | 1,146.7 | +7.1% | 3/5 | 0.3125 |
| Bi-CVRP | 100 | 17,205.0 | 18,153.3 | -948.3 | -5.2% | 1/5 | 0.4375 |
| Bi-CVRP | 200 | 18,164.2 | 17,210.4 | 953.8 | +5.5% | 3/5 | 0.8125 |
| Bi-KP | 25 | 18,721.8 | 24,123.1 | -5,401.3 | -22.4% | 3/5 | 0.8125 |
| Bi-KP | 50 | 30,255.6 | 29,638.7 | 616.9 | +2.1% | 2/5 | 1.0000 |
| Bi-KP | 100 | 29,927.3 | 27,239.9 | 2,687.4 | +9.9% | 4/5 | 0.6250 |
| Bi-KP | 200 | 31,724.1 | 29,974.8 | 1,749.2 | +5.8% | 2/5 | 1.0000 |

## Detailed Validity and Yield Results

The following table reports valid yield as valid scored heuristics per 100 LLM calls. The last two columns use the invalid/null proxy definition described above.

| Task | B | mpage_bmab valid / 100 calls | MPaGE-orig valid / 100 calls | Yield delta % | mpage_bmab invalid proxy | MPaGE-orig null-score rate |
|---|---:|---:|---:|---:|---:|---:|
| Bi-TSP | 25 | 28.0 | 32.8 | -14.6% | 66.7% | 18.7% |
| Bi-TSP | 50 | 25.2 | 22.8 | +10.5% | 71.4% | 27.5% |
| Bi-TSP | 100 | 20.6 | 19.8 | +4.0% | 77.2% | 21.9% |
| Bi-TSP | 200 | 18.6 | 16.0 | +16.2% | 79.7% | 30.1% |
| Tri-TSP | 25 | 28.8 | 24.0 | +20.0% | 68.9% | 28.7% |
| Tri-TSP | 50 | 18.4 | 18.4 | +0.0% | 80.1% | 33.4% |
| Tri-TSP | 100 | 13.2 | 11.8 | +11.9% | 85.7% | 43.4% |
| Tri-TSP | 200 | 9.0 | 7.8 | +15.4% | 90.3% | 53.9% |
| Bi-CVRP | 25 | 17.6 | 20.8 | -15.4% | 80.3% | 32.5% |
| Bi-CVRP | 50 | 14.4 | 15.2 | -5.3% | 84.3% | 50.7% |
| Bi-CVRP | 100 | 10.6 | 11.4 | -7.0% | 88.3% | 52.5% |
| Bi-CVRP | 200 | 8.3 | 8.6 | -3.5% | 91.0% | 48.3% |
| Bi-KP | 25 | 20.8 | 19.2 | +8.3% | 75.4% | 39.4% |
| Bi-KP | 50 | 16.0 | 15.6 | +2.6% | 81.9% | 33.5% |
| Bi-KP | 100 | 14.2 | 11.6 | +22.4% | 84.2% | 29.8% |
| Bi-KP | 200 | 13.5 | 8.8 | +53.4% | 85.3% | 43.6% |

## Interpretation

The original MPaGE paper optimizes for high-quality heuristic discovery through PFG-guided selection, semantic clustering, and fixed mutation/crossover scheduling. That design is strong when the run has enough budget to explore several generations. The mpage_bmab project changes the objective: instead of only asking what the final population looks like, it asks how much useful progress is achieved throughout a limited LLM-call budget.

The experimental results support that shift. mpage_bmab is not reliably better at final HV, but it is reliably better at AUBC. This means the bandit mechanism is mainly improving **when** good heuristics appear, not necessarily the best final heuristic discovered by the end of the run.

The B=50 budget-curve figure is useful here: because AUBC integrates the whole curve, early separation between methods matters even if final HV later becomes similar. This is exactly the regime where a budgeted method should be evaluated.

The validity/yield results refine this interpretation. mpage_bmab has a small overall advantage in valid heuristic yield, but the advantage is not universal. Therefore, the main AUBC improvement should be attributed to **more effective use of valid heuristics over the budget curve**, not simply to producing many more valid heuristics.

## Limitations

- The result folder only contains `full` mpage_bmab and `mpage_orig`. It does not contain the `no_ph`, `no_diversity`, `op_only`, or `cluster_only` ablations needed to isolate which component drives the gains.
- There are only five seeds per cell. The trends are practically strong, but formal statistical power is limited.
- The metrics are algorithm-level heuristic-population HV and AUBC, not the original paper's normalized solution-level HV/IGD. Absolute values should not be compared across tasks or against the paper tables.
- The `mpage_orig` wrapper counts budget comparably for total budget, but its `budget_history.json` is aggregate rather than per-call, so call-count diagnostics are less detailed.
- Null-score accounting differs between methods: MPaGE-orig stores `score: null` sample entries, while mpage_bmab stores valid samples and requires failure attempts to be inferred from `budget_history.json`. The invalid/null proxy figure should be used diagnostically, not as a perfectly symmetric measurement.
- Final-HV comparisons are noisy, especially on Bi-KP and Tri-TSP, so claims should emphasize budget efficiency rather than terminal dominance.

## Conclusion

The most important takeaway is that **mpage_bmab substantially improves budget efficiency over the original MPaGE workflow under equal LLM-call budgets**. It wins AUBC in 15 of 16 task-budget settings and shows especially strong gains on Bi-CVRP and Tri-TSP.

However, **mpage_bmab does not consistently dominate MPaGE-orig in final HV**. The evidence supports the thesis that adaptive bandit-guided generation makes better use of limited LLM calls, not that it always finds a better final heuristic population after the same total budget.

For a graduation-project narrative, the strongest claim is therefore: mpage_bmab reframes MPaGE for budget-constrained heuristic design and improves the area-under-budget-curve, while retaining broadly comparable final quality.
