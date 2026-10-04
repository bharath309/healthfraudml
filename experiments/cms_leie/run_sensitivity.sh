#!/bin/bash
# Resumable sensitivity chain: run_sensitivity.sh <set> <label> [ratio]   (set: primary | sensitivity_cc_2017plus; label: is_fraud | is_fraud_mandatory | is_fraud_5yr)
# For each fold: SMOTE-ENN on the pre-sampled training set (cached), then Gradient Boosting and Random Forest (warm-start, resumable).
SET=$1; LABEL=$2; RATIO=${3:-20}
for k in 1 2 3 4 5; do
  python3 resample_fold.py $k $SET $LABEL $RATIO
  for m in gb rf; do python3 train_model.py $m $k --set $SET --label $LABEL --ratio $RATIO --budget 100000; done
done
python3 summarise.py --set $SET --label $LABEL --ratio $RATIO
