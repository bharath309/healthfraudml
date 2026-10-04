#!/bin/bash
# Step 6, fold-parallel: same commands as run_sensitivity.sh, but every (sensitivity, fold) is its own job so all cores
# are busy (run_sensitivity.sh runs folds one after another). Longest jobs (50:1) first. Resumable.
set -uo pipefail
cd "$(dirname "$0")"; export PATH="$PWD/.venv/bin:$PATH"
echo "[$(date -u +%FT%TZ)] step 6 (fold-parallel): 25 jobs on $(( $(sysctl -n hw.ncpu) - 1 )) cores"
{ for k in 1 2 3 4 5; do echo "primary is_fraud 50 $k"; done
  for s in "primary is_fraud_mandatory" "primary is_fraud_5yr" "sensitivity_cc_2017plus is_fraud" "primary_plus_year is_fraud"; do
    for k in 1 2 3 4 5; do echo "$s 20 $k"; done; done; } \
| xargs -P $(( $(sysctl -n hw.ncpu) - 1 )) -L 1 sh -c '
  log="logs/sens_$0_$1_r$2_fold$3.log"
  { python3 resample_fold.py $3 $0 $1 $2 &&
    python3 train_model.py gb $3 --set $0 --label $1 --ratio $2 --budget 1000000 &&
    python3 train_model.py rf $3 --set $0 --label $1 --ratio $2 --budget 1000000; } > "$log" 2>&1 || echo "FAILED $0 $1 r$2 fold$3"'
for s in "primary is_fraud_mandatory 20" "primary is_fraud_5yr 20" "sensitivity_cc_2017plus is_fraud 20" "primary_plus_year is_fraud 20" "primary is_fraud 50"; do
  set -- $s; python3 summarise.py --set $1 --label $2 --ratio $3 > "logs/summarise_$1_$2_r$3.log" 2>&1 || echo "FAILED summarise $s"
done
python3 compare_cowork.py reports > mac_compare_reports_after_r50.md
echo "[$(date -u +%FT%TZ)] DONE"
