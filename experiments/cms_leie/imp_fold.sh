#!/bin/bash
# One fold of the sixth sensitivity: SMOTE-ENN, then GB and RF (resumable).
cd "$(dirname "$0")"; export PATH="$PWD/.venv/bin:$PATH"; k=$1
{ python3 resample_fold.py $k primary_imp is_fraud 20 &&
  python3 train_model.py gb $k --set primary_imp --label is_fraud --ratio 20 --budget 1000000 &&
  python3 train_model.py rf $k --set primary_imp --label is_fraud --ratio 20 --budget 1000000; } > logs/imp_fold$k.log 2>&1 || echo "FAILED fold $k"
