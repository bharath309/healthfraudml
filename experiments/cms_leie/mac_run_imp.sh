#!/bin/bash
# v32 brief phase 3: sixth sensitivity (suppressed demographic counts = 5.5). Build (resumable) then 5 folds in parallel.
set -uo pipefail
cd "$(dirname "$0")"; export PATH="$PWD/.venv/bin:$PATH"
S0=$(md5 -q out/scaler_stats.json)
python3 build_features.py --variant imp > logs/imp_build.log 2>&1 || { echo "FAILED build"; exit 1; }
[ "$(md5 -q out/scaler_stats.json)" = "$S0" ] || { echo "FAILED: frozen scaler_stats.json changed"; exit 1; }
echo "[$(date -u +%FT%TZ)] build done"
printf '%s\n' 1 2 3 4 5 | xargs -P 5 -n 1 ./imp_fold.sh
python3 summarise.py --set primary_imp --label is_fraud > logs/imp_summarise.log 2>&1 || echo "FAILED summarise"
echo "[$(date -u +%FT%TZ)] IMP DONE"
