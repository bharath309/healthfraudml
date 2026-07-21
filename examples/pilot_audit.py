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
  * v0.2.0 ships price benchmarks for ~7,300 CPT/HCPCS codes derived from the
    CMS Physician Fee Schedule (2025 national payment amounts). A curated set
    of common codes (ED E/M 99281-99285, office visits 99214/99215, wound
    repairs 12001/12002, 56420) additionally carries severity metadata used for
    upcoding detection.
  * Lines with codes outside the benchmark are still checked for structural
    issues (unbundling) but skip price benchmarking and are marked as such in
    the output — they are never silently dropped.
  * The benchmark holds code numbers + CMS payment ranges only (no procedure
    descriptions); the report uses your CSV's own `description` column.
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
    # Windows consoles default to a legacy code page (cp1252) that cannot encode
    # characters such as arrows. Without this, printing a report can raise
    # UnicodeEncodeError part-way through and lose the run. Degrade to
    # replacement characters instead of crashing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description="Audit a CSV of claim lines with HealthFraudML.")
    ap.add_argument("csv_file", type=Path, help="CSV with columns: cpt_code, amount, description")
    ap.add_argument("--provider", default="Pilot Provider", help="Provider/facility name for the report")
    ap.add_argument("--out", type=Path, default=Path("audit_report"), help="Output basename (writes .json and .md)")
    ap.add_argument("--no-coding-audit", action="store_true",
                    help="Skip the coding audit (code-vs-description check)")
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

    # --- coding audit: is the code right for the described service? ---
    coding_rows = []
    coding_limits = ""
    if not args.no_coding_audit:
        try:
            from healthfraudml.auditor.coding_audit import (
                CodingAuditor, CODING_AUDIT_LIMITS, plain_verdict, coverage_summary,
                status_label, plain_detail,
            )
            coding = CodingAuditor()
            coding_rows = coding.audit_bill(items)
            coding_limits = CODING_AUDIT_LIMITS
            if not coding.available:
                print("\nCoding audit not available — install healthfraudml[rag] to enable it.")
        except Exception as exc:  # never let the coding audit break the price audit
            print(f"\nCoding audit skipped ({exc.__class__.__name__}: {exc}).")
    report["coding_audit"] = coding_rows

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

    if coding_rows:
        print("\nDOES THE CODE MATCH THE SERVICE?")
        print("=" * 62)
        print(f"  {'CODE':<8}{'STATUS':<15}SERVICE")
        print(f"  {'-' * 7} {'-' * 14} {'-' * 36}")
        for row in coding_rows:
            # Never truncate the service name: on an E/M line the part that
            # would be cut ("level 5 (high severity)") is the part that matters.
            name = row.get("billed_name") or "(no description on file)"
            print(f"  {row['cpt_code']:<8}{status_label(row):<15}{name}")
            print(f"  {'':<8}{'':<15}{plain_detail(row)}")
        print()
        print(f"  {coverage_summary(coding_rows)}")
        if coding_limits:
            print(f"  {coding_limits}")

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
    if coding_rows:
        lines += ["", "## Does the code match the service?", "",
                  "| Code | Status | Service | What we found |",
                  "|---|---|---|---|"]
        for row in coding_rows:
            name = row.get("billed_name") or "_no description on file_"
            lines.append(
                f"| `{row['cpt_code']}` | **{status_label(row)}** | {name} | {plain_detail(row)} |"
            )
        lines += ["", coverage_summary(coding_rows), "", f"> {coding_limits}", ""]

    lines += ["", "## Draft dispute letter", "", "```", str(report.get("dispute_letter", "")).strip(), "```", ""]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
