#!/bin/bash
# Mac verification run of the CMS-LEIE arm (RUN_ON_MAC.md steps 2-6). Stops at the first mismatch with the Cowork run.
# Log: mac_run.log. Comparisons: mac_compare_*.md. Resumable: every step skips work already done.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
CORES=$(sysctl -n hw.ncpu)
log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "step 2: download CMS by-Provider 2013-2024"
python3 download.py leie 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
python3 compare_cowork.py downloads | tee mac_compare_downloads.md

log "step 3: labels, features, folds"
python3 build_labels.py
python3 build_features.py
python3 add_sensitivity_stats.py
python3 make_folds.py
python3 compare_cowork.py reports | tee mac_compare_reports.md

log "step 4: SMOTE-ENN per fold (3 in parallel)"
printf '%s\n' 1 2 3 4 5 | xargs -P 3 -I{} python3 resample_fold.py {}
log "step 4: train 8 models x 5 folds ($((CORES - 2)) jobs in parallel)"
for m in gb rf mlp nb rf200 gb200 svm svc200; do for k in 1 2 3 4 5; do echo "$m $k"; done; done \
  | xargs -P $((CORES - 2)) -L 1 sh -c 'python3 train_model.py $0 $1 --budget 1000000 > "logs/train_${0}_${1}.log" 2>&1 || echo "FAILED $0 $1"'

log "step 5: summarise primary"
python3 summarise.py
python3 compare_cowork.py results primary_is_fraud | tee mac_compare_primary.md

log "step 6: sensitivities (GB + RF), five chains in parallel"
python3 make_folds.py 50
./run_sensitivity.sh primary is_fraud_mandatory      > logs/sens_mandatory.log 2>&1 &
./run_sensitivity.sh primary is_fraud_5yr            > logs/sens_5yr.log 2>&1 &
./run_sensitivity.sh sensitivity_cc_2017plus is_fraud > logs/sens_cc2017.log 2>&1 &
./run_sensitivity.sh primary_plus_year is_fraud      > logs/sens_year.log 2>&1 &
./run_sensitivity.sh primary is_fraud 50             > logs/sens_r50.log 2>&1 &
wait
python3 compare_cowork.py reports | tee mac_compare_reports_after_r50.md || true
log "DONE"
