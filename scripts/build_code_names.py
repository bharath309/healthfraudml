#!/usr/bin/env python3
"""
Build `code_names.csv` — plain-language names for billing codes.

Two lawful sources, kept strictly separate (see the IP notes below):

1. ``cms_hcpcs_l2`` — HCPCS **Level II** long descriptions, ingested from the
   public CMS quarterly alpha-numeric HCPCS file. Level II descriptors are
   maintained by CMS and are in the public domain.
2. ``authored`` — short plain-language names written by this project for
   numeric CPT (Level I) codes. These are paraphrases, NOT AMA descriptors,
   and are surfaced in reports labelled "(unofficial name)".

## IP exclusions enforced here (do not remove)

* **Numeric (Level I / CPT) rows are skipped entirely.** CMS ships CPT long and
  short descriptions inside the same HCPCS file, but those descriptors are
  copyrighted by the AMA. We never read them into our data.
* **D-series rows are skipped.** Per CMS's own record layout, Level II D-codes
  carry ADA CDT copyright — the same class of restriction as CPT, despite being
  letter-prefixed.

Everything that survives is a non-D, letter-prefixed Level II code.

Fixed-width field positions (1-based, from HCPC*_recordlayout.txt):
    code              1-5
    long description  12-91
    short description 92-119

Usage:
    python scripts/build_code_names.py HCPC2025_JUL_ANWEB_v3.txt \
        --authored-csv path/to/authored_cpt_names.csv \
        --out healthfraudml/auditor/data/code_names.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

CODE_SLICE = slice(0, 5)
LONG_DESC_SLICE = slice(11, 91)
SHORT_DESC_SLICE = slice(91, 119)

# Letter prefixes whose descriptors are NOT public domain.
EXCLUDED_PREFIXES = {"D"}  # ADA CDT copyright


def parse_hcpcs_level2(path: Path) -> dict[str, str]:
    """Return {code: long_description} for public-domain Level II codes only."""
    names: dict[str, str] = {}
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            code = line[CODE_SLICE].strip().upper()
            if len(code) != 5:
                continue
            first = code[0]
            # Level I (numeric CPT) -> AMA copyright -> skip.
            if not first.isalpha():
                continue
            # D-series -> ADA CDT copyright -> skip.
            if first in EXCLUDED_PREFIXES:
                continue
            desc = line[LONG_DESC_SLICE].strip() or line[SHORT_DESC_SLICE].strip()
            if not desc:
                continue
            # First occurrence wins (later rows are modifier/sequence variants).
            names.setdefault(code, " ".join(desc.split()))
    return names


def load_authored(path: Path | None) -> dict[str, str]:
    """Load project-authored plain-language names (columns: code, plain_name)."""
    authored: dict[str, str] = {}
    if not path or not path.is_file():
        return authored
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            code = row.get("code") or row.get("cpt_code")
            name = row.get("plain_name") or row.get("name")
            if code and name:
                authored[code.upper()] = name
    return authored


def build(hcpcs: Path, authored_csv: Path | None, out: Path,
          restrict_to: Path | None = None) -> tuple[int, int]:
    level2 = parse_hcpcs_level2(hcpcs)
    if restrict_to and restrict_to.is_file():
        keep = {r["cpt_code"] for r in csv.DictReader(open(restrict_to, encoding="utf-8"))}
        level2 = {c: d for c, d in level2.items() if c in keep}
    authored = load_authored(authored_csv)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "plain_name", "source"])
        for code in sorted(level2):
            w.writerow([code, level2[code], "cms_hcpcs_l2"])
        for code in sorted(authored):
            # authored entries win if a code somehow appears in both
            if code in level2:
                continue
            w.writerow([code, authored[code], "authored"])
    return len(level2), len(authored)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hcpcs_file", type=Path, help="CMS alpha-numeric HCPCS fixed-width .txt")
    ap.add_argument("--authored-csv", type=Path, default=None,
                    help="CSV of project-authored CPT names (code,plain_name)")
    ap.add_argument("--restrict-to-benchmark", type=Path, default=None,
                    help="Optional cms_pfs_benchmark.csv to limit L2 codes to priced ones")
    ap.add_argument("--out", type=Path,
                    default=Path("healthfraudml/auditor/data/code_names.csv"))
    args = ap.parse_args()
    n_l2, n_auth = build(args.hcpcs_file, args.authored_csv, args.out,
                         args.restrict_to_benchmark)
    print(f"Wrote {args.out}: {n_l2} HCPCS Level II (non-D) + {n_auth} authored names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
