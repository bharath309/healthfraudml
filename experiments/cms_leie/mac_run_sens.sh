#!/bin/bash
# Step 6 only (sensitivities, GB + RF), run after the primary comparison showed cross-platform float differences
# (see MAC_VS_COWORK.md). Same commands as the tail of mac_run.sh.
set -uo pipefail
cd "$(dirname "$0")"; export PATH="$PWD/.venv/bin:$PATH"
echo "[$(date -u +%FT%TZ)] step 6: sensitivities, five chains in parallel"
python3 make_folds.py 50
./run_sensitivity.sh primary is_fraud_mandatory       > logs/sens_mandatory.log 2>&1 &
./run_sensitivity.sh primary is_fraud_5yr             > logs/sens_5yr.log 2>&1 &
./run_sensitivity.sh sensitivity_cc_2017plus is_fraud > logs/sens_cc2017.log 2>&1 &
./run_sensitivity.sh primary_plus_year is_fraud       > logs/sens_year.log 2>&1 &
./run_sensitivity.sh primary is_fraud 50              > logs/sens_r50.log 2>&1 &
wait
python3 compare_cowork.py reports > mac_compare_reports_after_r50.md
echo "[$(date -u +%FT%TZ)] DONE"
