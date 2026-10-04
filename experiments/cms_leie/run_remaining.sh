#!/bin/bash
# Queue for the remaining sensitivities (runs after the two chains now in progress have exited): 5-year window; year as a feature; 50:1 negatives.
cd "$(dirname "$0")"
while pgrep -f "run_sensitivity.sh primary is_fraud_mandatory" > /dev/null || pgrep -f "run_sensitivity.sh sensitivity_cc_2017plus" > /dev/null; do sleep 60; done
./run_sensitivity.sh primary is_fraud_5yr 20
./run_sensitivity.sh primary_plus_year is_fraud 20
./run_sensitivity.sh primary is_fraud 50
