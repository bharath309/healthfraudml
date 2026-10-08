#!/bin/bash
# Protocol T driver (cloud primary run, 8 Oct 2026): labels -> builds -> fits (resumable) ; metrics run separately after the OOF files exist.
set -u; cd "$(dirname "$0")"; export ASOF_ARM_DIR=/home/claude/cms_leie; export PYTHONUNBUFFERED=1
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
log "protocol sha256 $(sha256sum protocol_T_frozen.md | cut -c1-64)"
[ -f ../data/asof_known.parquet ] || { log "labels"; python3 asof_labels.py > ../logs/asof_labels.log 2>&1 || { log "labels FAILED"; exit 1; }; }
for t in 2018 2019 2020 2021 2022; do
  for V in A B; do
    if [ ! -f ../data/asof_resampled_${t}_${V}.npz ] || [ ! -f ../data/asof_test_${t}_${V}.npz ]; then log "build $t $V"; python3 asof_build.py $t $V > ../logs/asof_build_${t}_${V}.log 2>&1 || { log "build $t $V FAILED"; exit 1; }; fi
    for m in gb rf; do
      until [ -f ../results/asof/fit_${t}_${V}_${m}.json ]; do log "train $t $V $m"; python3 asof_train.py $t $V $m --budget 170 >> ../logs/asof_train_${t}_${V}_${m}.log 2>&1 || { log "train $t $V $m FAILED"; exit 1; }; done
      log "done $t $V $m"
    done
  done
done
log "ALL FITS DONE"
