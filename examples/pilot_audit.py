#!/usr/bin/env python3
"""
HealthFraudML — Pilot Quickstart Script
=======================================
Audits a CSV of billed claim lines and produces a flagged-claims report
plus a draft dispute letter. Runs entirely in YOUR environment — no data
leaves your machine, no API keys required.

Usage:
    pip install healthfraudml            # or: pip install git+https://github.com/bharath309/healthfraudml.git
    python pilot_audit.py your_claims.csv --provider "Your Facility Name"

Input CSV format (header row required):
    cpt_code,amount,description
    99285,6672.00,ED Visit Level 5
    56420,709.00,Bartholin Cyst I&D
    99214,350.00,Office Visit Established Level 4

Notes for pilot evaluators:
  * v0.1.0 ships with price benchmarks for 10 common CPT codes (ED E/M levels
    99281-99285, office visits 99214/99215, wound repairs 12001/12002, 56420).
    Lines with other codes are still checked for structural issues (upcoding
    patterns, unbundling) but skip price benchmarking and are marked as such
    in the output — they are never silently dropped.
  * Expanded benchmark coverage (CMS Physician Fee Schedule) is on the
    roadmap for pilot partners — tell us which code families you need.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_claims(csv_path: Path):
    items = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"cpt_code", "amount"}
        if not required.issubset({(c or "").strip().lower() for c in reader.fieldnames or []}):
            sys.exit(
                f"ERROR: CSV must have columns: cpt_code, amount[, description]. Found: {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            if not row.get("cpt_code") and not row.get("description"):
                continue
            try:
                amount = float(row["amount"].replace("$", "").replace(",", ""))
            except (ValueError, KeyError):
                print(f"  ! Skipping line {i}: unparseable amount {row.get('amount')!r}")
                continue
            items.append(
                {
                    "cpt_code": row.get("cpt_code", ""),
                    "amount": amount,
                    "description": row.get("description", ""),
                }
            )
    return items


def main():
    ap = argparse.ArgumentParser(description="Audit a CSV of claim lines with HealthFraudML.")
    ap.add_argument("csv_file", type=Path, help="CSV with columns: cpt_code, amount, description")
    ap.add_argument("--provider", default="Pilot Provider", help="Provider/facility name for the report")
    ap.add_argument("--out", type=Path, default=Path("audit_report"), help="Output basename (writes .json and .md)")
    args = ap.parse_args()

    try:
        from healthfraudml import BillingAuditor
    except ImportError:
        sys.exit("ERROR: healthfraudml not installed. Run: pip install healthfraudml")

    items = load_claims(args.csv_file)
    if not items:
        sys.exit("ERROR: no valid claim lines found in the CSV.")
    print(f"Loaded {len(items)} claim lines from {args.csv_file}")

    auditor = BillingAuditor(provider_name=args.provider)
    report = auditor.audit_bill(items)

    # --- console summary ---
    print("=" * 62)
    print(f"  Provider:            {report['provider_name']}")
    print(f"  Total billed:        ${report['total_billed']:,.2f}")
    print(f"  Risk level:          {report['risk_level']}")
    print(f"  Suggested savings:   ${report['suggested_savings']:,.2f}")
    print(f"  Findings:            {len(report.get('findings', []))}")
    print("=" * 62)
    for finding in report.get("findings", []):
        if isinstance(finding, dict):
            print(f"  • [{finding.get('severity','?')}] {finding.get('type','Finding')}: {finding.get('message','')}")
        else:
            print(f"  • {finding}")

    # --- write artifacts ---
    json_path = args.out.with_suffix(".json")
    md_path = args.out.with_suffix(".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    lines = [
        f"# Billing Audit Report — {report['provider_name']}",
        "",
        f"**Total billed:** ${report['total_billed']:,.2f}  ",
        f"**Risk level:** {report['risk_level']}  ",
        f"**Suggested savings:** ${report['suggested_savings']:,.2f}",
        "",
        "## Findings",
        "",
    ]
    findings = report.get("findings", [])
    if not findings:
        lines.append("- (no findings)")
    for finding in findings:
        if isinstance(finding, dict):
            lines.append(f"- **[{finding.get('severity','?')}] {finding.get('type','Finding')}** — {finding.get('message','')}")
        else:
            lines.append(f"- {finding}")
    lines += ["", "## Line-item detail", ""]
    for it in report.get("audited_items", []):
        code = it.get("cpt_code", "?")
        amt = it.get("billed_amount", it.get("amount", 0)) or 0
        status = it.get("status", "")
        note = it.get("notes", "")
        lines.append(f"- `{code}` ${amt:,.2f} — **{status}**{(' — ' + note) if note else ''}")
    lines += ["", "## Draft dispute letter", "", "```", str(report.get("dispute_letter", "")).strip(), "```", ""]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
