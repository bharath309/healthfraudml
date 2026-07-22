# Medicare benchmark: what we compare against, and what is wrong with it today

Status: **design note — describes defects in the shipped benchmark and the fix.
Not yet implemented.**

## The problem in one line

For a hospital emergency-department bill we compare the **whole hospital charge**
against the **physician's professional fee only**, which inflates the reported
multiple by roughly 4.6x.

```
  bill charges $6,672.00 for CPT 99285
    vs professional only :  39.5x   <- what the tool reports today
    vs Medicare total    :   8.5x   <- the defensible number
```

`39.5x` will not survive contact with a billing department. `8.5x` is real,
checkable, and still a strong argument. Overstating in the direction that
flatters us is the worst possible direction for credibility.

## Three defects, increasing in severity

### 1. Hand-written prices override verified CMS data

`_BUILTIN_CPT_REFERENCE` carries hard-coded `medicare_min/max` and `fair_max`
values written before the CMS ingest existed. Because the built-ins are overlaid
*on top of* the CMS benchmark, they win for all 10 curated codes.

| code | CMS 2025 derived | hand-written | tool reports |
|---|---|---|---|
| 99285 | **168.85** | 250.00 | 250.00 |
| 99283 | 68.25 | 120.00 | 120.00 |
| 12001 | 91.22 | 130.00 | 130.00 |
| 99214 | 125.18 | 140.00 | 140.00 |

The generated letter states these as fact and invites verification:

> "Medicare's national payment for this service is $250.00 ... Medicare rates are
> published by CMS in the Physician Fee Schedule and can be verified independently."

CMS says $168.85. The sentence added to make the letter credible is the sentence
that discredits it.

The ceilings are incoherent too: 99285 uses 3,500 where 5x CMS is 844.25, while
99214 uses 450 where 5x CMS is 625.90 — the overrides run in both directions, so
no rule explains them.

**Fix:** delete the hand-written prices. Built-ins keep only what CMS cannot
supply — `description` and `severity`. All pricing comes from the CMS benchmark.

### 2. The benchmark has no facility component

`cms_pfs_benchmark.csv` is built from the Physician Fee Schedule, which prices
the **professional** service only. For a hospital ED visit CMS pays twice:

```
  CPT 99285 RVU components (CMS 2025)
    work RVU              4.00  ->  $129.39   the physician's own work
    practice expense RVU  0.79  ->  $ 25.55   facility setting
    malpractice RVU       0.43  ->  $ 13.91
    TOTAL                 5.22  ->  $168.85   <- professional fee
```

Practice expense is only 0.79 RVU because the hospital supplies the room, staff
and equipment and is paid for them **separately under OPPS**. Independent
sources put the Medicare facility payment for a Level 5 ED visit at roughly
**$612**, giving a Medicare total near **$781** — none of which is in our data.

Note also that 99285's facility and non-facility PE RVUs are identical (0.79),
i.e. there is no office-setting version of an ED visit at all.

**What we do not have:** `OPPSCAP_JAN.csv` ships in the same CMS zip and looks
relevant, but it is a locality-specific payment *cap* file for technical-component
services — `99285` does not appear in it. Facility rates come from **OPPS
Addendum B** (APC payment rates), a separate CMS download, plus a HCPCS -> APC
mapping.

### 3. The comparison is invalid for facility-based lines

Even with correct numbers, dividing a facility charge by a professional rate
compares different things. This affects every ED and hospital outpatient line —
exactly the bills the pilot targets.

## The hard part: we cannot tell which component a line represents

From a patient CSV (`cpt_code, amount, description`) there is no way to know
whether a line is the professional charge, the facility charge, or a combined
one. Signals that may be available:

- a place-of-service column, when the partner's export has one
- the provider name (hospital vs clinic) — weak, but present
- the same code appearing twice with different amounts (professional + facility)
- the code itself: ED codes 99281-99285 essentially always involve a facility
- facility vs non-facility PE RVU differing materially in the PFS file, which
  tells us the code *has* two settings

None of these is reliable alone. The design must degrade honestly when the
setting is unknown rather than guess.

## Options

| option | what it does | cost |
|---|---|---|
| A. Ingest OPPS Addendum B | gives a facility rate per HCPCS via APC | new data pipeline + APC mapping |
| B. Ask the user | one flag: hospital bill or office bill | pushes work to the partner, but they know the answer |
| C. Show both anchors, no single multiple | "professional $168.85; facility ~$612; total ~$781" | honest, no inference needed |
| D. Suppress the multiple where setting is unknown | never states an indefensible number | loses the headline figure |

**Recommendation: C now, A+B next.** Reporting both anchors and a total requires
no inference about setting, kills the 39.5x overstatement immediately, and is
strictly more informative. Ingesting OPPS (A) and accepting a
setting hint (B) then sharpen it into a single defensible multiple.

## Consequences to plan for

Fixing defect 1 alone changes every headline figure already in circulation:
the `$5,472` savings on `demo_bill_1` appears in `PILOT_ROADMAP.md`, the PR
description, the case-study plan and the test suite. Re-baselining is required,
and any number already shown to a third party should be corrected rather than
quietly changed.

## Verification trail for the figures above

- CY 2025 PFS conversion factor **32.3465** — CMS fact sheet states $32.35,
  a 2.83% reduction from 2024's $33.2875. The value is *not* present in the RVU
  file itself, so it must be sourced and cited, not assumed.
- 99285 RVUs (work 4.00 / facility PE 0.79 / MP 0.43, total 5.22) read directly
  from `PPRRVU25_JAN.csv`.
- $168.85 = 5.22 x 32.3465, recomputed from the raw CMS file.
- The ~$174 professional figure quoted by third-party sources is the same number
  on the 2024 conversion factor (5.22 x 33.2875 = $173.76); it is not a conflict.
- The ~$612 facility figure and ~$2,209 average *list price* come from
  third-party summaries and are **not yet verified against OPPS Addendum B**.
  Treat as indicative until ingested.
