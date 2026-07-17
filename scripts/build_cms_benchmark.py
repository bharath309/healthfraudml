#!/usr/bin/env python3
"""
Build the CMS Physician Fee Schedule price benchmark for HealthFraudML.

Reads the official CMS PFS Relative Value File (PPRRVU) and emits a
CSV of CPT/HCPCS **code numbers + national Medicare payment amounts only**.
No AMA CPT descriptions are read into or written out of the benchmark —
the auditor uses the partner's own `description` column for display.

National payment = Total RVU x Conversion Factor (facility and non-facility
computed separately). This reproduces CMS's national, unadjusted amounts;
locality (GPCI) adjustments are intentionally not applied — a benchmark, not
a bill re-pricer.

`fair_max` is a DISCLOSED HEURISTIC dispute ceiling = medicare_max * FAIR_MULT,
not an official figure. It exists only to flag charges far above Medicare for
human review. Tune FAIR_MULT to taste.

Source (2025): https://www.cms.gov/files/zip/rvu25a.zip  (PPRRVU25_JAN.csv)
Conversion factor 2025: 32.3465

Usage:
    python scripts/build_cms_benchmark.py PPRRVU25_JAN.csv \
        --cf 32.3465 --fair-mult 5.0 \
        --out healthfraudml/auditor/data/cms_pfs_benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# PPRRVU column indices (0-based), stable across recent CMS releases.
COL_HCPCS = 0
COL_MOD = 1
COL_STATUS = 3
COL_NONFAC_TOTAL_RVU = 11
COL_FAC_TOTAL_RVU = 12

# Status codes that represent a separately payable service.
PAYABLE_STATUS = {"A", "R", "T"}


def _f(x: str):
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def build(src: Path, cf: float, fair_mult: float, out: Path) -> int:
    rows = list(csv.reader(open(src, encoding="latin-1")))
    seen = set()
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cpt_code", "medicare_min", "medicare_max", "fair_min", "fair_max"])
        for r in rows:
            if len(r) <= COL_FAC_TOTAL_RVU:
                continue
            code = (r[COL_HCPCS] or "").strip()
            mod = (r[COL_MOD] or "").strip()
            # base code only (no modifier rows), 5-char HCPCS, payable status
            if mod or len(code) != 5 or r[COL_STATUS].strip() not in PAYABLE_STATUS:
                continue
            if code in seen:
                continue
            nf_rvu = _f(r[COL_NONFAC_TOTAL_RVU])
            fac_rvu = _f(r[COL_FAC_TOTAL_RVU])
            prices = [round(v * cf, 2) for v in (nf_rvu, fac_rvu) if v and v > 0]
            if not prices:
                continue
            mmin, mmax = min(prices), max(prices)
            fair_max = round(mmax * fair_mult, 2)
            w.writerow([code, mmin, mmax, mmax, fair_max])
            seen.add(code)
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pprrvu_csv", type=Path, help="Path to CMS PPRRVU CSV (e.g. PPRRVU25_JAN.csv)")
    ap.add_argument("--cf", type=float, default=32.3465, help="Medicare conversion factor (2025 = 32.3465)")
    ap.add_argument("--fair-mult", type=float, default=5.0, help="fair_max = medicare_max * this (disclosed heuristic)")
    ap.add_argument("--out", type=Path,
                    default=Path("healthfraudml/auditor/data/cms_pfs_benchmark.csv"))
    args = ap.parse_args()
    n = build(args.pprrvu_csv, args.cf, args.fair_mult, args.out)
    print(f"Wrote {n} codes to {args.out} (CF={args.cf}, fair_mult={args.fair_mult}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
