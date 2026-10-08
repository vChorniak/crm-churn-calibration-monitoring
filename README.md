# Calibration-aware monitoring of churn prediction models under distribution shift

Code and results for the paper

> Chorniak, V.; Tymchenko, O. *Which Monitoring Signals Detect Harmful Degradation? Calibration-Aware Monitoring
> of Churn Prediction Models under Distribution Shift.* Submitted to *Applied Sciences* (MDPI), 2026.

The study compares eighteen label-based and label-free monitoring signals (Brier score, Murphy reliability, ECE,
calibration slope, PR-AUC, ROC-AUC, lift, PSI, KS, mean score, conformal ambiguity, …) under identical alarm rules,
in a controlled simulation with five drift regimes and 50 replications, and on two public datasets
(KKBox subscription churn and UCI Online Retail II). It also evaluates threshold-based intervention policies,
a retraining loop with a model-promotion rule, and the effect of label noise.

## Repository layout

```
src/
  simulation.py                 synthetic CRM generator, scoring pipeline (L1-LR + isotonic + Mondrian conformal), 
                                monitoring signals; 50 seeds x 6 drift scenarios -> results/slices*.csv
  detection_analysis.py         alarm rules (3-sigma, 2-of-3), detection delays, false-alarm rates -> results/detection*.csv
  retraining.py                 retraining triggers and promotion rules (strict vs non-inferiority) -> results/retrain.csv
  label_noise.py                L1 / L2 / no penalty under symmetric label noise -> results/noise.csv
  real_data.py                  rolling-origin study on KKBox and Online Retail II -> results/real_pairs.csv
  baselines.py                  gradient boosting and ablations (synthetic and real) -> results/baselines_*.csv
  summarise_retraining_real.py  summaries of the retraining and real-data results -> results/retrain_summary.csv
  collect_numbers.py            every number quoted in the paper -> results/numbers.json
  make_figures.py               Figures 1-6 -> figures/
  data_prep/                    conversion and monthly labelling of the two public datasets
results/                        all per-slice and summary results used in the paper
figures/                        the figures of the paper
data/README.md                  where to download the public datasets
run_all.sh                      reproduces everything end to end
```

## Requirements

Python 3.10 or newer and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The results in `results/` were produced with Python 3.11, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0,
pandas 3.0.2, pyarrow 25.0.1, matplotlib 3.10.9 and DuckDB 1.5.6.

## Data

The synthetic experiments need no external data. The real-data analyses use two public datasets that are **not**
redistributed here; see [`data/README.md`](data/README.md) for download links and the expected file locations.

KKBox churn labels are produced for every month (February 2015 – February 2017) by re-implementing the
organisers' published labelling procedure (`WSDMChurnLabeller.scala`); for February 2017 the labels coincide
with the official `train.csv` labels for all users present in both sets (the script prints this check).

## Reproducing the results

```
bash run_all.sh
```

or step by step (from the repository root):

| Step | Command | Output | Time* |
|---|---|---|---|
| Data preparation | `src/data_prep/*.py` (see `run_all.sh`) | `data/processed/` | ~5 min |
| Simulation | `python src/simulation.py` and `python src/simulation.py extra` | `results/slices.csv`, `results/slices_extra.csv` | ~17 min |
| Detection analysis | `python src/detection_analysis.py` | `results/detection*.csv` | <1 min |
| Retraining | `python src/retraining.py` | `results/retrain.csv` | ~40 min |
| Label noise | `python src/label_noise.py` | `results/noise.csv` | ~6 min |
| Real data | `python src/real_data.py` | `results/real_pairs.csv` | ~2 min |
| Baselines | `python src/baselines.py` | `results/baselines_*.csv` | ~12 min |
| Summaries | `python src/summarise_retraining_real.py`, `python src/collect_numbers.py` | `results/retrain_summary.csv`, `results/numbers.json` | <1 min |
| Figures | `python src/make_figures.py` | `figures/*.png` | <1 min |

\*On a 2-core machine. `python src/simulation.py 5` runs a quick check with 5 seeds instead of 50.

All random seeds, including the solver seed, are fixed, so re-running the scripts reproduces the numbers in
`results/`.

## Mapping to the paper

| Paper | Source |
|---|---|
| Table 1 (effect of drift scenarios) | `results/slices.csv`, `results/slices_extra.csv` (`numbers.json` → `harm`) |
| Table 2 (stable-period baselines) | `results/baselines_synth.csv` |
| Table 3, Figure 2 (detection delays, false alarms) | `results/detection_summary.csv` |
| Figure 3 (signal trajectories), Figure 4 (policies) | `results/slices.csv`, `results/slices_extra.csv` |
| Table 4, Figure 5 (retraining, promotion rule) | `results/retrain_summary.csv` (figure file `fig6_promotion.png`) |
| Table 5 (label noise) | `results/noise.csv` |
| Table 6 (real-data quality) | `results/baselines_real.csv` |
| Table 7, Figure 6 (signals vs. degradation on real data) | `results/real_pairs.csv` (figure file `fig5_real.png`) |
| Figure 1 (framework) | `figures/fig1_framework.png` |

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.

## Citation

If you use this code, please cite the paper above (the full reference and DOI will be added after publication).
