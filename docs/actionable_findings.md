# What counts as a finding, and what the visit notes would unlock

Status: **design note — not implemented.** Changes what every bill reports, so
it needs agreement before code.

## The problem

Across the nine demo bills the tool produces 19 findings:

| type | count | share |
|---|---|---|
| Overpricing | 14 | **74%** |
| Unbundling | 3 | 16% |
| Upcoding | 2 | 11% |

and rates **8 of 9 bills "High risk"**.

Almost all of that is the tool objecting to a price the hospital is entitled to
charge. A chargemaster amount is set by the hospital, is not governed by any
rule we can cite, and is not evidence of an error. Reporting it as a
High-severity finding in a product called a fraud detector is a category error:
**we built a price comparator and labelled it a fraud finding.**

It is also the critique the pilot partner already made — that a 5x-Medicare
ceiling has no standing in a dispute. The v0.3.0 rename ("review ceiling", not
"fair market value") made the *label* honest while leaving the *classification*
wrong.

## The line

| a rule was broken - **finding** | just expensive - **context** |
|---|---|
| code does not match the service | chargemaster is high |
| duplicate charges | markup above Medicare |
| unbundling (violates coding rules) | above any multiple of a benchmark |
| charge with no code disclosed | |
| **billed above the plan's allowed amount** | |
| service not documented as rendered | |

Everything on the left is an error or a violation, and a patient can demand it
be corrected. Everything on the right is a price, where the remedy is
negotiation, financial assistance or the cash price - not a letter alleging
something is wrong.

## Proposed changes

1. **Price stops being a finding.** It becomes context on the line:
   *"Medicare pays $777 for this; this hospital charges $6,672."* Useful
   leverage, no accusation, nothing to defend.
2. **Findings reserved for errors and violations**, per the table above.
3. **Risk level stops keying on price.** A correctly coded but expensive bill
   should read *low risk, high cost* - those are different facts and a reader
   needs both.
4. **Billed vs allowed becomes the headline check.** It is the one place a real
   rule can be broken (in-network balance billing) and the only number that
   changes what the patient actually owes. Already scoped as the v0.3.1
   fast-track: partner supplies the EOB amount, pure arithmetic, no statutory
   claim.

## Second input: the visit summary / notes

Today the coding audit prints *"visit-level codes depend on the doctor's notes,
not the bill"* and stops. The notes are obtainable - patients can download an
After Visit Summary from their portal, and have a right of access to their
records - so the honest next step is to accept them as a second input.

What that unlocks, in descending order of confidence:

| check | confidence | why |
|---|---|---|
| **Was the billed service documented at all?** | high | a charge for a procedure the note never mentions is the strongest question on any bill |
| Do the dates and provider match the charge? | high | simple comparison |
| Does the described service match the billed code? | medium | this is today's coding audit, with a much better description than a bill line |
| Is the E/M level supported? | **low - do not assert** | level rests on medical decision making or total time; judging it is a coding determination, which this tool does not make |

The fourth row is the tempting one and the one to refuse. We can surface
*whether MDM or time is documented at all* - "the note records no total time and
no assessment of problem complexity, so the basis for a level 5 charge is not
visible" - which is a question, not a determination.

### Constraints

- **Notes are the most sensitive document a patient holds.** Local-first
  processing is not optional here; nothing may leave the machine.
- Extraction must be **quoting, not summarising**: point at the line in the note
  that does or does not support the charge, so the patient can verify it.
- Absence of documentation is **not** proof a service was not rendered. Frame as
  "not documented in the records provided", never "not performed".

## Sequencing

1. Reclassify findings vs context (this note, section above) - no new data needed
2. EOB allowed-amount check - one number from the patient, highest value
3. Visit-summary ingest, starting with "was this service documented"
4. OPPS facility rates - still worth having as context, but it makes a weak
   comparison less wrong rather than making it right

Note the reordering: OPPS ingest was previously the next engineering task. Under
this framing it drops to last, because it improves a number that should no
longer be a finding at all.
