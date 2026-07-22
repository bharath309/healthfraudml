# Code vocabulary: what the auditor can name, and what it can match

This document records a constraint that is easy to rediscover the hard way: the
coding audit's coverage is bounded by **code-description licensing**, not by the
quality of the matcher.

## Two different jobs

A code name is used for two separate purposes, and they have different rules.

| Job | What it needs | Where it comes from |
|---|---|---|
| **Display** — show a human what a code means | any name we may lawfully print | shipped public-domain data, or a name this project wrote |
| **Match** — resolve "what does this description sound like?" | an *official* description | official public-domain descriptions only |

Project-authored paraphrases are **display-only**. Matching a bill against
wording this project invented would let an unofficial name drive a coding
suggestion, so authored entries are excluded from the search index
(`CodingAuditor.indexable_codes()`).

## What ships today

`healthfraudml/auditor/data/code_names.csv` — 8,397 entries:

- **8,387** HCPCS Level II descriptions from the public CMS quarterly
  alpha-numeric HCPCS file. Public domain. **Indexed and matchable.**
- **10** project-authored paraphrases for numeric CPT codes, always rendered
  with `(unofficial name)`. **Displayed, never matched.**

Every displayed name carries provenance (`description_source` /
`billed_name_source`), so a reader can tell whether a description is their own
wording, an official CMS one, or one we wrote.

## The licence landscape

| System | Licence | Appears on | Status here |
|---|---|---|---|
| HCPCS Level II | public domain (CMS) | physician/outpatient — drugs, supplies, vaccines, G-codes | **shipped + indexed** |
| ICD-10-PCS | public domain (CMS) | inpatient hospital procedures | not shipped |
| ICD-10-CM | public domain (CDC/NCHS) | diagnoses, not services | not shipped |
| MS-DRG | public domain (CMS) | inpatient payment groups | not shipped |
| NDC | public domain (FDA) | drugs | not shipped |
| **CPT (HCPCS Level I)** | **AMA licensed** | **physician/outpatient — most of our target bills** | **excluded** |
| CDT (D-codes) | ADA licensed | dental | **excluded at ingest** |
| Revenue codes | NUBC licensed | facility bills | not shipped |
| LOINC | free, licence terms + attribution | labs | not shipped |
| SNOMED CT | free in US via UMLS licence | clinical terms | not shipped |

## The structural cap

The codes we cannot name are exactly the codes that dominate the bills we audit.
Of 7,359 priced codes in the CMS PFS benchmark, **7,128 have a price but no
name** — almost all numeric CPT. No public-domain vocabulary fixes this, because
CMS publishes CPT *rates* freely while AMA controls CPT *descriptions*.

So: **the coding audit is capped at HCPCS Level II lines.** The price audit has
no such cap — it covers all 7,359 codes. State this plainly to evaluators rather
than letting them infer the tool is broken when a CPT line reports "not checked".

Adding the other public-domain systems does **not** lift the cap: ICD-10-PCS is
inpatient-only, ICD-10-CM is diagnoses rather than services, and neither appears
on the outpatient bills the pilot targets.

## Planned: bring-your-own vocabulary

Licensing is a decision for the person running the tool, not for this project.
The local-first model makes that workable: **we ship the mechanism, the user
supplies the vocabulary.** Nothing licensed enters this repo or the PyPI package.

Intended design:

- Vocabulary directory: `HEALTHFRAUDML_VOCAB_DIR`, default
  `~/.healthfraudml/vocabularies/`
- Drop-in CSVs with columns `code, plain_name, system`
- The loader merges shipped public-domain data with user-supplied files
- User-supplied entries are labelled `user_supplied` in provenance output
- Adding a file changes the vocabulary, which triggers the existing index
  rebuild — no stale-cache trap

### Non-negotiable: segment the index by code system

Matching must be restricted to the billed code's own system. A single embedding
space holding diagnoses *and* procedures *and* drugs lets a CPT line resolve to
an ICD-10-CM diagnosis or an NDC drug — a category error that manufactures the
same false-positive class already fixed twice in this module. Segmentation is a
requirement of the feature, not an optimisation.

### Licence points to surface in the docs

These belong to the user, but the documentation should raise them rather than
let them be discovered later:

1. Internal use and redistribution are usually licensed, and priced,
   differently.
2. **Our output is designed to be sent to a third party.** A dispute letter or
   report containing licensed descriptors, mailed to a provider, is plausibly
   distribution. Anyone loading a licensed vocabulary should understand that
   before generating letters from it.

## Verifying the shipped vocabulary

The shipped file is checked against an independent re-parse of the CMS source:
row count and split, no duplicates or empty names, **no numeric (AMA CPT) codes**,
**no D-series (ADA CDT) codes**, every code present in the CMS source, every name
matching CMS verbatim, no eligible code dropped, ASCII-only, and index vector
count equal to the indexable vocabulary.

Regenerate with:

```bash
python scripts/build_code_names.py HCPC2025_JUL_ANWEB_v3.txt \
    --authored-csv scripts/authored_cpt_names.csv \
    --out healthfraudml/auditor/data/code_names.csv
```
