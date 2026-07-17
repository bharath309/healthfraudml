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
