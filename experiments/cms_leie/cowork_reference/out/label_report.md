# CMS–LEIE arm — gate 1 label report (built 3 Oct 2026 in the Cowork cloud workspace)

Every number below is read from `out/label_report.json`, `out/download_manifest.json` and `out/folds_report.json`, produced by `download.py`, `build_labels.py` and `make_folds.py` (this folder). No model has been trained.

## Files downloaded (SHA-256 recorded; CSVs converted to parquet with every column read as text)

| key | file | bytes | SHA-256 | rows | cols | HTTP Last-Modified |
|---|---|---|---|---|---|---|
| leie | LEIE_UPDATED.csv | 15,608,468 | `175352046e438bc477fe0f8985c645c3eb492114b449e13a5700b30e00d27417` | 84,001 | 18 | Thu, 10 Sep 2026 12:13:06 GMT |
| 2013 | MUP_PHY_R25_P05_V20_D13_Prov.csv | 325,926,227 | `d388c76a251c434d06c4c20991190065e1ca9e68e66113cce9be8e0612c9fdac` | 956,251 | 56 | Fri, 07 Nov 2025 13:34:43 GMT |
| 2014 | MUP_PHY_R25_P05_V20_D14_Prov.csv | 336,174,411 | `85f9e377795dd348350a8fee61cbf7ba608ca4276579e0566cdd239cb5fa7994` | 986,657 | 56 | Fri, 07 Nov 2025 15:26:18 GMT |
| 2015 | MUP_PHY_R25_P05_V20_D15_Prov.csv | 347,504,913 | `cb6bda1d7b00cd476264b399e78bce15c19143b11c45a41904c8030c9bbf4214` | 1,019,377 | 56 | Fri, 07 Nov 2025 15:34:06 GMT |
| 2016 | MUP_PHY_R25_P05_V20_D16_Prov.csv | 362,426,740 | `6ab55e3255d1ea101107ac32bcd40b9db5bfb94246b379c0665bf157af08cdd2` | 1,053,958 | 56 | Fri, 07 Nov 2025 15:41:27 GMT |
| 2017 | MUP_PHY_R25_P07_V20_D17_Prov.csv | 433,626,921 | `38a3227ea26f0297d4a5441e90805a3d35330bd5a26c3561acd0c36d14ef9baa` | 1,088,687 | 81 | Fri, 07 Nov 2025 16:19:36 GMT |
| 2018 | MUP_PHY_R25_P07_V20_D18_Prov.csv | 446,044,426 | `04a10009a91cc0e26f75d6dfcdac77b0119f26566e3002ade3f48b67370a518f` | 1,121,462 | 81 | Fri, 07 Nov 2025 16:30:54 GMT |
| 2019 | MUP_PHY_R25_P07_V20_D19_Prov.csv | 459,418,436 | `bd789342af70524356be846953a5c1943a041e423760c4c481f6dd64178f79b4` | 1,155,870 | 81 | Fri, 07 Nov 2025 16:49:24 GMT |
| 2020 | MUP_PHY_R25_P07_V20_D20_Prov.csv | 458,027,890 | `6de6d53eaa8fd681a3513bbf691b410bcfcd040354dcd814f98317b87d71593a` | 1,161,542 | 81 | Fri, 07 Nov 2025 17:14:47 GMT |
| 2021 | MUP_PHY_R25_P07_V20_D21_Prov.csv | 473,462,899 | `6b0a6281038371cc8bbf1a310c12d403dba4e24db6da17a5750be5af4eaa8bc5` | 1,198,754 | 81 | Fri, 07 Nov 2025 17:46:55 GMT |
| 2022 | MUP_PHY_R25_P07_V20_D22_Prov.csv | 484,670,744 | `5c34a49258ea9c09933592667c802c24669e22b077b761a20b65dfa47bc62914` | 1,230,293 | 81 | Fri, 07 Nov 2025 18:00:07 GMT |
| 2023 | MUP_PHY_R25_P05_V20_D23_Prov.csv | 495,304,913 | `33dac951769eab332ad366f347c090a233de4404de2b3016a2a5e59b684de8cb` | 1,259,343 | 81 | Tue, 08 Apr 2025 14:15:45 GMT |
| 2024 | MUP_PHY_R26_P05_V10_D24_Prov.csv | 508,819,864 | `e9e779645b5dd8fdd39e8533df525899b488a66863bb51ea8644ec59861feefc` | 1,296,739 | 81 | Mon, 11 May 2026 15:34:54 GMT |

Source URLs: CMS DCAT catalog `https://data.cms.gov/data.json` (saved as `out/by_provider_sources.json`); LEIE `https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv` (file dated 10 Sep 2026). **CY2024 is published** (catalog modified 2026-05-21) and is included: twelve calendar years 2013–2024.

Column sets: 2013–2016 files have 56 columns (no chronic-condition block); 2017–2024 files have 81 (the 25-field `Bene_CC_BH_*`/`Bene_CC_PH_*` block of the re-released R25/R26 files). No file carries a provider-gender field. One row per NPI per year (asserted unique).

## LEIE

- leie_rows: 84,001
- leie_rows_with_npi: 8,881
- leie_rows_primary_set: 74,198
- leie_rows_primary_set_with_npi: 7,335
- leie_rows_mandatory_set_with_npi: 4,478
- leie_rows_with_reindate: 0
- leie_rows_with_waiverdate: 2
- leie_rows_bad_excldate: 0
- distinct_npis_primary: 7,231
- distinct_npis_mandatory: 4,436
- distinct_npis_any_authority: 8,700
- Distinct `EXCLTYPE` values (count): `1128b4` 33,393; `1128a1` 26,047; `1128a2` 8,118; `1128a3` 5,884; `1128a4` 3,593; `1128b14` 2,166; `1128b8` 1,497; `1128b1` 942; `1128b5` 805; `1128b7` 756; `1128b3` 330; `1128Aa` 172; `1128b6` 77; `1128b2` 66; `1156` 53; `1128b15` 37; `BRCH SA` 28; `BRCH CIA` 13; `1128b11` 13; `1160` 8; `1128b16` 2; `1128b12` 1
- Normalised matching of the primary authorities: 1128a1 → ['1128a1'], 1128a2 → ['1128a2'], 1128a3 → ['1128a3'], 1128b4 → ['1128b4'], 1128b7 → ['1128b7'], 1128c3gi → None, 1128c3gii → None. **The two 1128(c)(3)(G) codes do not occur in this LEIE snapshot**, so the primary set is effectively 1128(a)(1), (a)(2), (a)(3), (b)(4), (b)(7); the mandatory-only set is (a)(1), (a)(2), (a)(3).
- Exclusion year (rounded per refs [11]/[16]: year of min(EXCLDATE, WAIVERDATE), +1 when the month is July–December) for primary-set rows with an NPI: 1989: 2, 1993: 1, 1995: 1, 1996: 1, 1997: 1, 2000: 3, 2001: 1, 2002: 3, 2003: 3, 2004: 3, 2005: 3, 2006: 2, 2007: 4, 2008: 10, 2009: 113, 2010: 226, 2011: 174, 2012: 286, 2013: 298, 2014: 388, 2015: 411, 2016: 463, 2017: 452, 2018: 400, 2019: 362, 2020: 292, 2021: 343, 2022: 377, 2023: 550, 2024: 597, 2025: 752, 2026: 626, 2027: 187

## Provider-years and labels (US rows; rows of matched NPIs in or after the exclusion year dropped)

| year | US rows | kept | dropped post-exclusion | fraud (primary) | rate % | fraud (mandatory) | fraud (5-yr window) | entity I / O | fraud I / O |
|---|---|---|---|---|---|---|---|---|---|
| 2013 | 956,177 | 956,177 | 0 | 1,733 | 0.181 | 1,021 | 535 | 898,179 / 57,998 | 1,714 / 19 |
| 2014 | 986,584 | 986,581 | 3 | 1,607 | 0.163 | 955 | 526 | 925,257 / 61,324 | 1,585 / 22 |
| 2015 | 1,019,301 | 1,019,299 | 2 | 1,480 | 0.145 | 894 | 464 | 956,725 / 62,574 | 1,457 / 23 |
| 2016 | 1,053,883 | 1,053,880 | 3 | 1,339 | 0.127 | 800 | 412 | 991,284 / 62,596 | 1,317 / 22 |
| 2017 | 1,088,599 | 1,088,594 | 5 | 1,170 | 0.107 | 704 | 376 | 1,026,731 / 61,863 | 1,148 / 22 |
| 2018 | 1,121,205 | 1,121,203 | 2 | 1,003 | 0.089 | 576 | 396 | 1,060,427 / 60,776 | 984 / 19 |
| 2019 | 1,155,787 | 1,155,785 | 2 | 845 | 0.073 | 463 | 429 | 1,095,856 / 59,929 | 827 / 18 |
| 2020 | 1,161,468 | 1,161,465 | 3 | 683 | 0.059 | 354 | 473 | 1,100,723 / 60,742 | 671 / 12 |
| 2021 | 1,198,687 | 1,198,684 | 3 | 542 | 0.045 | 261 | 478 | 1,133,848 / 64,836 | 532 / 10 |
| 2022 | 1,230,217 | 1,230,215 | 2 | 384 | 0.031 | 176 | 384 | 1,165,571 / 64,644 | 377 / 7 |
| 2023 | 1,259,266 | 1,259,264 | 2 | 243 | 0.019 | 102 | 243 | 1,196,456 / 62,808 | 239 / 4 |
| 2024 | 1,296,656 | 1,296,654 | 2 | 140 | 0.011 | 55 | 140 | 1,235,673 / 60,981 | 139 / 1 |
| **all** | 13,527,830 | 13,527,801 | 29 | **11,169** | **0.0826** | 6,361 | 4,856 | | |

- Matched NPIs (primary set) present in the CMS files: 2,131; NPIs under any authority present: 2,646.
- Fraud rows by entity type: I: 10,990, O: 179.
- Most frequent provider types among fraud rows (counts only): Family Practice 1,954; Internal Medicine 1,619; Nurse Practitioner 793; Psychiatry 552; Chiropractic 481; Emergency Medicine 412; Podiatry 378; Anesthesiology 369; General Practice 280; Physical Medicine and Rehabilitation 272; Physician Assistant 252; Obstetrics & Gynecology 206; Orthopedic Surgery 193; General Surgery 191; Pain Management 183
- The fraud rate falls from 0.181% (2013) to 0.011% (2024): exclusions lag the billing they punish and the LEIE snapshot cannot yet contain exclusions that will follow 2023–2024 billing. **Year is therefore not a feature** (brief §3.1). Only 29 provider-years of matched NPIs fall in or after the exclusion year — excluded parties do stop billing, which is a sanity check on the join.
- Sanity range (brief §1c): fraud rate 0.083% vs 0.075% (ref [16], 2012–2015, finer grain) and ~0.03% (ref [11]); rows per year 0.96–1.30 M. In range.

## Decision rule applied (fixed in the brief before any count was seen)
- P = 11,169 fraud provider-years ≥ 2,000 → **negatives pre-sampled at 20 per positive** in each training fold.

## Folds (StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by NPI; provider-disjointness asserted)

| fold | test rows | test positives | test providers | training positives | sampled negatives | training rows | training providers |
|---|---|---|---|---|---|---|---|
| 1 | 2,705,561 | 2,234 | 384,721 | 8,935 | 178,700 | 187,635 | 168,575 |
| 2 | 2,705,560 | 2,234 | 384,721 | 8,935 | 178,700 | 187,635 | 168,514 |
| 3 | 2,705,560 | 2,234 | 384,721 | 8,935 | 178,700 | 187,635 | 168,509 |
| 4 | 2,705,560 | 2,233 | 384,721 | 8,936 | 178,720 | 187,656 | 168,613 |
| 5 | 2,705,560 | 2,234 | 384,722 | 8,935 | 178,700 | 187,635 | 168,480 |

Rows 13,527,801; positives 11,169; providers 1,923,606.
