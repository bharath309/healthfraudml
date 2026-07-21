# Auditor reference data

This directory holds two generated data files: `cms_pfs_benchmark.csv` (prices)
and `code_names.csv` (plain-language names). Neither contains AMA CPT descriptors.

---

# code_names.csv — plain-language code names

Columns: `code, plain_name, source` where `source` ∈ `cms_hcpcs_l2 | authored`.

## Sources and the IP boundary

| source | what it is | provenance |
|---|---|---|
| `cms_hcpcs_l2` | HCPCS **Level II** long descriptions | Public CMS quarterly alpha-numeric HCPCS file (2025 July release, `HCPC2025_JUL_ANWEB_v3.txt`). Maintained by CMS, public domain. |
| `authored` | Short plain-language paraphrases for numeric CPT codes | Written by this project. **Not** AMA descriptors. Always displayed with an `(unofficial name)` suffix. |

**Two exclusions are enforced in `scripts/build_code_names.py` and must not be removed:**

1. **Numeric (Level I / CPT) rows are skipped.** CMS ships CPT long/short
   descriptions inside the same file, but those descriptors are AMA-copyrighted.
2. **D-series rows are skipped.** Per CMS's own record layout, Level II D-codes
   carry ADA CDT copyright — the same class of restriction as CPT, despite being
   letter-prefixed.

A regression test asserts every `cms_hcpcs_l2` row is letter-prefixed and non-D.

## Regenerating

```bash
# download the quarterly alpha-numeric HCPCS zip from CMS, unzip, then:
python scripts/build_code_names.py HCPC2025_JUL_ANWEB_v3.txt \
    --authored-csv scripts/authored_cpt_names.csv \
    --out healthfraudml/auditor/data/code_names.csv
```

Growing the authored CPT list is a matter of adding rows to
`scripts/authored_cpt_names.csv` and re-running the script. Every code that
gains a name also becomes resolvable by the coding audit.

---

# CMS PFS price benchmark

`cms_pfs_benchmark.csv` holds **CPT/HCPCS code numbers + national Medicare
payment amounts only**. It contains **no AMA CPT descriptions** — the auditor
uses the partner's own `description` column for display.

## Provenance

- **Source:** CMS Physician Fee Schedule Relative Value File (PPRRVU),
  2025 release — <https://www.cms.gov/files/zip/rvu25a.zip> (`PPRRVU25_JAN.csv`).
- **National payment** = Total RVU × conversion factor (2025 CF = **32.3465**),
  computed separately for facility and non-facility settings. No locality
  (GPCI) adjustment is applied — this is a national benchmark, not a bill
  re-pricer.
- **Coverage:** ~7,300 separately payable codes (status A / R / T with a
  positive total RVU). Codes outside this set are still checked for structural
  issues (unbundling) but skip price benchmarking and are marked as such in the
  report — never silently dropped.

## Columns

| column | meaning |
|---|---|
| `cpt_code` | 5-char CPT/HCPCS code |
| `medicare_min` | lower of facility / non-facility national payment |
| `medicare_max` | higher of facility / non-facility national payment |
| `fair_min` | = `medicare_max` (floor of the "fair" band) |
| `fair_max` | **disclosed heuristic** dispute ceiling = `medicare_max` × 5.0 |

`fair_max` is **not an official figure**. It is a heuristic threshold that flags
charges far above Medicare for human review; a charge above it is a prompt to
look, not proof of fraud. Tune the multiplier in `scripts/build_cms_benchmark.py`.

## Regenerating

```bash
# download rvu25a.zip from CMS, unzip, then:
python scripts/build_cms_benchmark.py PPRRVU25_JAN.csv \
    --cf 32.3465 --fair-mult 5.0 \
    --out healthfraudml/auditor/data/cms_pfs_benchmark.csv
```
