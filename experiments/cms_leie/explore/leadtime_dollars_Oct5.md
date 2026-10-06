# EXPLORATORY: lead time, dollar capture, score vs years-to-exclusion (cloud OOF, GB and RF) — 5 Oct 2026
Post-hoc analysis of saved out-of-fold scores; not a frozen-protocol result. Capacity is applied within each calendar year.

Fraud rows 11169 from 2131 excluded NPIs; their Medicare payments total $2,339.3 M.

| model | capacity | flagged rows | fraud rows caught | recall (rows) | precision | lift | excluded NPIs caught (share) | fraud $ caught (M) | $-weighted recall | all flagged $ (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| GB | top 0.5% | 67,633 | 1,233 | 0.110 | 0.0182 | 22.1× | 405 (0.190) | 606.8 | 0.259 | 24,085.3 |
| GB | top 1% | 135,270 | 1,848 | 0.165 | 0.0137 | 16.5× | 551 (0.259) | 819.4 | 0.350 | 39,187.2 |
| GB | top 2% | 270,550 | 2,586 | 0.232 | 0.0096 | 11.6× | 731 (0.343) | 997.3 | 0.426 | 61,674.4 |
| GB | top 5% | 676,385 | 3,835 | 0.343 | 0.0057 | 6.9× | 1,046 (0.491) | 1,233.5 | 0.527 | 111,870.8 |
| GB | top 10% | 1,352,775 | 5,160 | 0.462 | 0.0038 | 4.6× | 1,336 (0.627) | 1,439.1 | 0.615 | 184,066.9 |
| GB | 1,000/yr | 12,000 | 374 | 0.033 | 0.0312 | 37.7× | 157 (0.074) | 214.3 | 0.092 | — |
| RF | top 0.5% | 67,633 | 1,066 | 0.095 | 0.0158 | 19.1× | 360 (0.169) | 478.1 | 0.204 | 16,239.9 |
| RF | top 1% | 135,270 | 1,638 | 0.147 | 0.0121 | 14.7× | 514 (0.241) | 657.7 | 0.281 | 29,111.7 |
| RF | top 2% | 270,550 | 2,390 | 0.214 | 0.0088 | 10.7× | 687 (0.322) | 839.1 | 0.359 | 53,390.7 |
| RF | top 5% | 676,385 | 3,716 | 0.333 | 0.0055 | 6.7× | 997 (0.468) | 1,175.4 | 0.502 | 111,760.7 |
| RF | top 10% | 1,352,775 | 5,026 | 0.450 | 0.0037 | 4.5× | 1,302 (0.611) | 1,403.4 | 0.600 | 190,877.2 |
| RF | 1,000/yr | 12,000 | 298 | 0.027 | 0.0248 | 30.1× | 142 (0.067) | 109.4 | 0.047 | — |

## Triage rules at the same capacity (GB): probability vs expected payments at risk (p × payment) vs payment alone
| rule | capacity | fraud rows caught | recall (rows) | excluded NPIs caught | fraud $ caught (M) | $-weighted recall |
|---|---|---|---|---|---|---|
| p_gb | top 0.5% | 1,233 | 0.110 | 405 | 606.8 | 0.259 |
| p_gb | top 1% | 1,848 | 0.165 | 551 | 819.4 | 0.350 |
| p_gb | top 2% | 2,586 | 0.232 | 731 | 997.3 | 0.426 |
| p_gb | top 5% | 3,835 | 0.343 | 1,046 | 1,233.5 | 0.527 |
| p_gb | top 10% | 5,160 | 0.462 | 1,336 | 1,439.1 | 0.615 |
| p_gb_x_payment | top 0.5% | 787 | 0.070 | 261 | 1,240.7 | 0.530 |
| p_gb_x_payment | top 1% | 1,211 | 0.108 | 379 | 1,452.8 | 0.621 |
| p_gb_x_payment | top 2% | 1,896 | 0.170 | 538 | 1,674.6 | 0.716 |
| p_gb_x_payment | top 5% | 3,137 | 0.281 | 804 | 1,906.1 | 0.815 |
| p_gb_x_payment | top 10% | 4,351 | 0.390 | 1,039 | 2,060.7 | 0.881 |
| payment_only | top 0.5% | 181 | 0.016 | 66 | 942.9 | 0.403 |
| payment_only | top 1% | 342 | 0.031 | 116 | 1,133.4 | 0.485 |
| payment_only | top 2% | 718 | 0.064 | 227 | 1,394.3 | 0.596 |
| payment_only | top 5% | 1,435 | 0.128 | 400 | 1,677.2 | 0.717 |
| payment_only | top 10% | 2,373 | 0.212 | 617 | 1,892.0 | 0.809 |

Payments to later-excluded providers after the first year they entered the GB top 1%: 551 NPIs; $913.9 M in fraud-labelled years strictly after the flag year ($1,099.8 M from the flag year on); 500 of them billed in later years, median 4.0 further years.
Within-provider trend of the GB percentile (escalation check): excluded NPIs with ≥3 rows 1604, share whose rank improves toward exclusion 0.497 (median slope 0.0000/yr) vs non-fraud sample 0.556 (-0.0030/yr) — no escalation signal in the current score.
Equal-share-per-specialty allocation (top k% within each year × provider type) vs global ranking, fraud rows caught: 1% 1,335 vs 1,848; 5% 3,048 vs 3,835.

## Lead time to exclusion (first year the provider entered the top-k of its year)
| model | capacity | excluded NPIs flagged ever | share | median lead (yr) | flagged ≥2 yr before exclusion (share of all excluded) | ≥3 yr |
|---|---|---|---|---|---|---|
| GB | top 1% | 551 | 0.259 | 7.0 | 0.254 | 0.240 |
| GB | top 5% | 1,046 | 0.491 | 7.0 | 0.481 | 0.449 |
| GB | top 10% | 1,336 | 0.627 | 7.0 | 0.611 | 0.576 |
| RF | top 1% | 514 | 0.241 | 7.0 | 0.240 | 0.229 |
| RF | top 5% | 997 | 0.468 | 7.0 | 0.459 | 0.435 |
| RF | top 10% | 1,302 | 0.611 | 7.0 | 0.599 | 0.567 |

## Score vs years to exclusion (fraud rows; GB)
| years to exclusion | rows | mean within-year percentile (GB) | share in top 1% (GB) | share in top 5% (GB) | mean percentile (RF) |
|---|---|---|---|---|---|
| 1 | 310 | 0.180 | 0.135 | 0.387 | 0.173 |
| 2 | 808 | 0.201 | 0.156 | 0.368 | 0.195 |
| 3 | 1,151 | 0.211 | 0.159 | 0.341 | 0.204 |
| 4 | 1,303 | 0.217 | 0.181 | 0.339 | 0.215 |
| 5 | 1,284 | 0.221 | 0.183 | 0.362 | 0.221 |
| 6 | 1,208 | 0.212 | 0.186 | 0.358 | 0.217 |
| 7 | 1,104 | 0.219 | 0.178 | 0.360 | 0.224 |
| 8 | 1,048 | 0.225 | 0.171 | 0.349 | 0.230 |
| 9 | 938 | 0.234 | 0.155 | 0.340 | 0.233 |
| 10 | 785 | 0.237 | 0.155 | 0.313 | 0.240 |
| 11 | 592 | 0.245 | 0.133 | 0.321 | 0.249 |
| 12 | 396 | 0.257 | 0.129 | 0.278 | 0.261 |
| 13 | 195 | 0.274 | 0.128 | 0.241 | 0.271 |
| 14 | 47 | 0.266 | 0.085 | 0.234 | 0.283 |

Billing years observed before exclusion: median first-observed-to-exclusion 8.0 yr; median last-observed-to-exclusion 3.0 yr; distribution of last-to-exclusion: {1: 310, 2: 541, 3: 452, 4: 302, 5: 197, 6: 106, 7: 80, 8: 63, 9: 40, 10: 22, 11: 13, 12: 4, 13: 1}.

## As-of label completeness (positives among rows with year < t that are already excluded at t)
| t | eventual positive rows (year < t) | known at t | share known |
|---|---|---|---|
| 2015 | 3,340 | 152 | 0.046 |
| 2016 | 4,820 | 358 | 0.074 |
| 2017 | 6,159 | 803 | 0.130 |
| 2018 | 7,329 | 1,329 | 0.181 |
| 2019 | 8,332 | 1,948 | 0.234 |
| 2020 | 9,177 | 2,431 | 0.265 |
| 2021 | 9,860 | 3,043 | 0.309 |
| 2022 | 10,402 | 3,874 | 0.372 |
| 2023 | 10,786 | 5,271 | 0.489 |
| 2024 | 11,029 | 6,816 | 0.618 |
| 2025 | 11,169 | 8,787 | 0.787 |

## Where the top-1% flags fall (aggregate provider types; precision at top 1%, types with ≥2,000 flagged rows)
| provider type | flagged | fraud | precision |
|---|---|---|---|
| Anesthesiology | 3,193 | 84 | 0.0263 |
| Physical Medicine and Rehabilitation | 5,468 | 121 | 0.0221 |
| Pain Management | 5,181 | 112 | 0.0216 |
| General Practice | 5,601 | 112 | 0.0200 |
| Neurology | 2,483 | 49 | 0.0197 |
| Clinical Laboratory | 3,525 | 62 | 0.0176 |
| Family Practice | 22,656 | 394 | 0.0174 |
| Internal Medicine | 17,595 | 247 | 0.0140 |
| Podiatry | 4,067 | 50 | 0.0123 |
| Interventional Pain Management | 5,675 | 57 | 0.0100 |
| Nurse Practitioner | 9,885 | 89 | 0.0090 |
| Licensed Clinical Social Worker | 2,773 | 20 | 0.0072 |
| Psychiatry | 24,522 | 163 | 0.0066 |
