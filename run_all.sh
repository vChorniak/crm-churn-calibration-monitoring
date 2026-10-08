#!/usr/bin/env bash
# Reproduces all results, tables (via results/numbers.json) and figures of the paper.
# Run from the repository root. Approximate run times on a 2-core machine are given in brackets.
set -euo pipefail
export DATA_DIR="${DATA_DIR:-data/processed}"
mkdir -p results figures data/work "$DATA_DIR"

# 1. Real data preparation (needs the raw files described in data/README.md)          [~5 min]
python src/data_prep/kkbox_01_to_parquet.py data/raw/kkbox data/work/kkbox
python src/data_prep/kkbox_02_label_features.py data/work/kkbox "$DATA_DIR" 100000
python src/data_prep/retail_01_xlsx_to_parquet.py data/raw/online_retail_II.xlsx data/work/retail
python src/data_prep/retail_02_label_features.py data/work/retail "$DATA_DIR"

# 2. Controlled simulation: 5 drift scenarios + the prior-fixed scenario, 50 seeds   [~17 min]
python src/simulation.py
python src/simulation.py extra
python src/detection_analysis.py | tee results/detection_report.txt

# 3. Retraining / promotion rule, label noise                                        [~45 min]
python src/retraining.py
python src/label_noise.py

# 4. Real data (rolling origin) and baselines / ablation                             [~15 min]
python src/real_data.py
python src/baselines.py
python src/summarise_retraining_real.py | tee results/retraining_real_report.txt

# 5. Numbers quoted in the paper and figures                                         [<1 min]
python src/collect_numbers.py
python src/make_figures.py
