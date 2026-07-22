#!/bin/bash
# HealthFraudML — one-click setup (Mac / Linux)
# Double-click this file (Mac) or run: bash setup_billaudit.command
# It will: find Python -> install the tool -> create a BillAudit folder
# with the runner + sample -> run a verification audit. ~2 minutes.
set -u

echo "============================================="
echo "  HealthFraudML - Bill Audit setup"
echo "============================================="

# --- 1. find python ---------------------------------------------------------
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo ""
  echo "Python isn't installed yet (one-time, ~5 min):"
  echo "  1. Opening python.org/downloads in your browser..."
  echo "  2. Download and run the installer, then double-click this file again."
  command -v open >/dev/null 2>&1 && open "https://www.python.org/downloads/"
  read -r -p "Press Enter to close..." _; exit 1
fi
echo "[1/4] Found Python: $($PY --version 2>&1)"

# --- 2. install the tool ----------------------------------------------------
echo "[2/4] Installing HealthFraudML (may take a minute)..."
$PY -m pip install --quiet --upgrade healthfraudml 2>/dev/null \
  || $PY -m pip install --quiet --user --upgrade healthfraudml \
  || { echo "!! pip install failed - email bharath.p90@gmail.com"; read -r -p "Press Enter to close..." _; exit 1; }

# --- 3. set up the BillAudit folder ----------------------------------------
FOLDER="$HOME/BillAudit"
mkdir -p "$FOLDER"
BASE="https://raw.githubusercontent.com/bharath309/healthfraudml/main/examples"
echo "[3/4] Creating $FOLDER ..."
curl -fsSL "$BASE/pilot_audit.py"            -o "$FOLDER/pilot_audit.py"            || { echo "!! download failed (internet?)"; read -r -p "Press Enter..." _; exit 1; }
curl -fsSL "$BASE/sample_claims_pilot.csv"   -o "$FOLDER/sample_claims_pilot.csv"   || { echo "!! download failed"; read -r -p "Press Enter..." _; exit 1; }

# the everyday runner: audits every CSV in the folder, writes <name>_report.md
cat > "$FOLDER/run_audit.command" <<RUNNER
#!/bin/bash
cd "\$(dirname "\$0")"
PY=\$(command -v python3 || command -v python)
found=0
for f in *.csv; do
  [ -e "\$f" ] || continue
  found=1
  name="\${f%.csv}"
  echo "Auditing \$f ..."
  "\$PY" pilot_audit.py "\$f" --provider "\$name" --out "\${name}_report"
  echo "  -> \${name}_report.md"
done
[ "\$found" = "0" ] && echo "No CSV files here yet - drop a bill CSV into this folder first."
command -v open >/dev/null 2>&1 && open .
read -r -p "Done. Press Enter to close..." _
RUNNER
chmod +x "$FOLDER/run_audit.command"

# --- 4. verification run ----------------------------------------------------
echo "[4/4] Verification: auditing the sample bill..."
cd "$FOLDER"
if $PY pilot_audit.py sample_claims_pilot.csv --provider "Setup Test" --out setup_test >/dev/null 2>&1 \
   && [ -f setup_test.md ]; then
  echo ""
  echo "=============================================="
  echo "  SUCCESS - everything works."
  echo "  Your folder:  $FOLDER"
  echo "  Daily use:    save a bill as CSV into that"
  echo "  folder, then double-click run_audit.command"
  echo "=============================================="
  command -v open >/dev/null 2>&1 && open "$FOLDER"
else
  echo "!! Verification run failed - email bharath.p90@gmail.com"
fi
read -r -p "Press Enter to close..." _
