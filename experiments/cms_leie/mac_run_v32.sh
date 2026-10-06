#!/bin/bash
# Brief ClaudeCode_Brief_v32_Numbers_Prospective_Oct5.md, phases 1-2 on the Mac. Phase 2 (train_full -> score) and
# phase 1 run side by side. Resumable. Log: logs/v32_*.log
set -uo pipefail
cd "$(dirname "$0")"; export PATH="$PWD/.venv/bin:$PATH"; mkdir -p logs
( until python3 train_full.py --budget 100000 > logs/v32_train_full.log 2>&1; do grep -q "budget" logs/v32_train_full.log || exit 1; done
  python3 score_cy2024.py > logs/v32_score.log 2>&1 && echo "[$(date -u +%FT%TZ)] PHASE2 DONE" || echo "PHASE2 FAILED" ) &
( python3 capacity_table.py > logs/v32_capacity.log 2>&1 || echo "FAILED capacity"
  python3 bootstrap_ci.py --B 1000 --budget 100000 > logs/v32_bootstrap.log 2>&1 || echo "FAILED bootstrap"
  python3 explore_leadtime_dollars.py > logs/v32_explore.log 2>&1 || echo "FAILED explore"
  python3 label_completeness.py > logs/v32_labelcompl.log 2>&1 || echo "FAILED label_completeness"
  echo "[$(date -u +%FT%TZ)] PHASE1 DONE" ) &
wait; echo "[$(date -u +%FT%TZ)] ALL DONE"
