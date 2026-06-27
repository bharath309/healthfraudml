#!/bin/bash
cd "$(dirname "$0")"
echo "=== Installing dependencies ==="
pip3 install -r requirements.txt 2>&1
echo ""
echo "=== Running tests ==="
python3 -m pytest tests/ -v 2>&1
echo ""
echo "=== Done ==="
