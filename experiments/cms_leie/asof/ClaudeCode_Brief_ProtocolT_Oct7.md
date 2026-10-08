# Claude Code brief — Protocol T (as-of evaluation): commit the protocol, then run it on the Mac (7 Oct 2026)

Everything runs in `GitHub_Framework/experiments/cms_leie/` on the Mac, on the arm's existing data (nothing re-downloaded; no LEIE
file newer than 10 Sep 2026). Bharath runs every command and commit. Order matters: **the protocol is committed with its hash
before `asof_labels.py` is run, and no model is trained before that commit.** Pipeline smoke-tested in the cloud on synthetic data
(`asof/asof_smoke.py`, 7 Oct); not yet run on real data anywhere.

## 0. Files to put in `experiments/cms_leie/asof/` (from `Paper2_ML_Comparison/cms_leie_arm/asof/`; sha256 on the Mac must match)
| file | sha256 |
|---|---|
| `protocol_T_frozen.md` | 4bdbad3d02d1e41331b062a93d645d441345f6693ea8237e101b03f8f804c061 |
| `asof_common.py` | a16ddf62fb07ca8be747d388ac82c54d23d6426a632d551aefa334cd5a902513 |
| `asof_labels.py` | 55130d5e456e709362c69ecce511a451df6c879bfcff8ec51ea141a920d7386a |
| `asof_build.py` | de1350b3121376071e859cbf08cf2334eefb22aafd240e775e567f328b009515 |
| `asof_train.py` | ae33cb920967f42541d6758dc11549a5b5517572a9e4580a6e34e2a5db3a383b |
| `asof_metrics.py` | f143ddefb6cbc60dc7ef6d7a1f7dceabfde96b45a66addbe9c049c997854b11d |
| `asof_smoke.py` | 0711df8354a2a7a1b051b749c2daefeb6168f5d22a089215614a439a50d33683 |
(`PriorArt_Log_NewMethods.md` goes to `experiments/cms_leie/asof/` too; it is appended, never edited.)

## 1. Freeze
1. Branch: continue on `cms-leie-arm` (after the post-submission merge, on `main` or a new branch `protocol-t` — Bharath's choice).
2. `shasum -a 256 asof/protocol_T_frozen.md` → must print 4bdbad3d…; write `asof/PROTOCOL_HASH.txt` with that hash, the UTC time
   and the git commit hash after the commit (two-step: commit the protocol + scripts; then append the commit hash to
   `PROTOCOL_HASH.txt` and commit again). Commit message: "Protocol T frozen: as-of evaluation of CMS-LEIE flags (no model trained)".
3. Only then continue.

## 2. Smoke test on the Mac (synthetic; ~3 min)
`cd experiments/cms_leie/asof && REAL_ARM=.. python3 asof_smoke.py /tmp/asof_smoke` → must end with `SMOKE OK`. (It copies cmsarm.py
and the scripts into a throw-away synthetic arm; it never touches the real data.) If the imbalanced-learn version differs from the
arm's, record it in the report.

## 3. Real run (order of execution = protocol §10)
```
cd experiments/cms_leie/asof
python3 asof_labels.py                       # writes data/asof_known.parquet, results/asof/labels_report.json (label-only; seconds)
for t in 2018 2019 2020 2021 2022; do for V in A B; do python3 asof_build.py $t $V; done; done      # scaler stats on <= t-1; SMOTE-ENN; minutes each
```
Then the 20 fits; five as-of years in parallel (one process per t) as Phase 3 was run, GB first:
```
for t in 2018 2019 2020 2021 2022; do (python3 asof_train.py $t A gb --budget 600 && python3 asof_train.py $t A rf --budget 600 \
   && python3 asof_train.py $t B gb --budget 600 && python3 asof_train.py $t B rf --budget 600) > logs/asof_$t.log 2>&1 & done; wait
```
`asof_train.py` is resumable (state/asof_<t>_<V>_<model>.pkl); re-run the same command after any interruption. Expect the training
sets to be smaller than the arm's folds (only exclusions known at t are positives), so GB fits should take well under the arm's
2 h each.
```
ASOF_BOOT=1000 python3 asof_metrics.py       # results/asof/summary.json + summary.md; bootstrap over providers, B = 1,000
```
4. Check `results/asof/labels_report.json` first: for each t, `share_known` should rise with t (≈0.13 at 2017 start in the memo's
   table; here 2018–2022); `excl_year_mismatches_vs_arm` must be 0 (asserted).
5. Privacy: `results/asof/scores_*.parquet` carry NPIs — keep them out of the repository (add `results/asof/scores_*.parquet` to
   `.gitignore`); commit `labels_report.json`, `fit_*.json`, `summary.json`, `summary.md`, `out/asof_scaler_*.json` only.

## 4. Report back (one message, numbers only, no provider identifiers)
- `PROTOCOL_HASH.txt` content; the git commit hashes (freeze commit, results commit).
- `labels_report.json` per-t block; the summary.md table; the four decision rules as printed (MET / NOT MET) — reported as they are.
- Fit seconds per model and any resume events; platform/library versions (`fit_*.json`).
- SHA-256 of `data/asof_known.parquet`, of each `asof_resampled_*.npz` and of each `scores_*.parquet` (for the cloud replication).

## 5. Not in this brief
No HCPCS-mix features (M6), no new architectures, no change to the arm's data pipeline other than the as-of label column, no LEIE
download. Deviations from the protocol, if any become necessary, are appended to `protocol_T_frozen.md` with date and reason before
the affected step is run, and reported.
