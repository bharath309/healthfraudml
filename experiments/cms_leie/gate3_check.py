#!/usr/bin/env python3
"""Independent gate-3 checks (Verification_Checklist_CMS_LEIE_Arm_Oct3.md) on the Mac run, written separately from
summarise.py: it reads only saved outputs (data/folds.parquet, data/test_*_y_*.npy, results/oof/*.npy, results/*/fold*.json,
results/*/summary.json, out/*.json) and the paper's saved arms (../../results/<arm>/experiment_results.json).
Also profiles the chronic-condition missingness behind `cc_missing`. Writes gate3_report.json and prints markdown.
Usage: python3 gate3_check.py"""
import os, json, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_auc_score
from scipy.stats import kendalltau, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); OOF = os.path.join(HERE, 'results', 'oof')
PAPER = os.path.join(HERE, '..', '..', 'results')
NAMES = {'rf': 'Random Forest', 'svm': 'SVM (RBF)', 'mlp': 'Neural Network (MLP)', 'gb': 'Gradient Boosting', 'nb': 'Naive Bayes', 'ens': 'AdaBoost-Voting Ensemble'}
MET = ['f1', 'precision', 'recall', 'mcc', 'auc_pr', 'auc_roc']
TAGS = {'primary_is_fraud': ('primary', 'is_fraud'), 'primary_is_fraud_mandatory': ('primary', 'is_fraud_mandatory'),
        'primary_is_fraud_5yr': ('primary', 'is_fraud_5yr'), 'sensitivity_cc_2017plus_is_fraud': ('sensitivity_cc_2017plus', 'is_fraud'),
        'primary_plus_year_is_fraud': ('primary_plus_year', 'is_fraud'), 'primary_is_fraud_r50': ('primary', 'is_fraud')}
rep = {}; L = []
J = lambda p: json.load(open(p))

# 1. folds: provider-disjoint; sizes and positives as reported
f = pd.read_parquet(os.path.join(DATA, 'folds.parquet'))
npi_folds = f.groupby('npi')['fold'].nunique()
fr = J(os.path.join(HERE, 'out', 'folds_report.json'))
rep['folds'] = {'providers_in_more_than_one_fold': int((npi_folds > 1).sum()), 'rows_per_fold': f['fold'].value_counts().sort_index().astype(int).to_dict()}
L.append(f"## 1. Folds\n- Providers in more than one fold: **{rep['folds']['providers_in_more_than_one_fold']}**\n- Rows per fold: {rep['folds']['rows_per_fold']}")

def ytest(setname, label, k):
    p = os.path.join(DATA, f'test_{setname}_{k}_y_{label}.npy')
    return np.load(p) if os.path.exists(p) else np.load(os.path.join(DATA, f'test_primary_{k}_y_{label}.npy'))

# 2 + 3. per run: fold metrics recomputed from stored probabilities; summary means/SDs recomputed from fold files
rep['runs'] = {}
L.append("\n## 2–3. Fold metrics from stored probabilities, and summary means/SDs from fold files\n| run | model | max |Δ| fold JSON vs recomputed | max |Δ| summary vs fold files |\n|---|---|---|---|")
for tag, (setname, label) in TAGS.items():
    s = J(os.path.join(HERE, 'results', tag, 'summary.json')); rr = {}
    for m, mv in s['models'].items():
        d1 = 0.0
        for k in range(1, 6):
            pp = os.path.join(OOF, f'{tag}_{m}_fold{k}.npy')
            if not os.path.exists(pp): continue
            y = ytest(setname, label, k); p = np.load(pp)
            if m == 'svm': yh = (np.load(os.path.join(OOF, f'{tag}_svm_fold{k}_decision.npy')) > 0).astype(int)
            else: yh = (p > 0.5).astype(int)
            mine = {'f1': f1_score(y, yh), 'precision': precision_score(y, yh, zero_division=0), 'recall': recall_score(y, yh),
                    'mcc': matthews_corrcoef(y, yh), 'auc_pr': average_precision_score(y, p), 'auc_roc': roc_auc_score(y, p)}
            fold = mv['folds'][str(k)]
            d1 = max(d1, max(abs(mine[x] - fold[x]) for x in MET))
        d2 = max(max(abs(np.mean([fv[x] for fv in mv['folds'].values()]) - mv[x]['mean']),
                     abs(np.std([fv[x] for fv in mv['folds'].values()]) - mv[x]['std'])) for x in MET)
        rr[m] = {'recompute_max_abs_diff': float(d1), 'summary_max_abs_diff': float(d2)}
        L.append(f"| {tag} | {m} | {d1:.1e} | {d2:.1e} |")
    rep['runs'][tag] = rr

# 4. threshold identity on every grid row + nested selection, independently implemented (vectorised)
def sweep(y, p, ts):
    pos = y.sum(); neg = len(y) - pos; out = []
    for t in ts:
        yh = p > t; tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / pos; FPR = fp / neg; F = 2 * P * R / (P + R) if P + R else 0.0
        out.append((t, P, R, FPR, F))
    return out
grid = np.round(np.arange(0.01, 1.0, 0.01), 2); rep['threshold'] = {}
L.append("\n## 4. Gradient Boosting threshold study (all 99 grid thresholds)\n| run | max |precision − πR/(πR+(1−π)FPR)| | grid optimum (mine / summary) | nested F1 (mine / summary) |\n|---|---|---|---|")
for tag, (setname, label) in TAGS.items():
    ys = {k: ytest(setname, label, k) for k in range(1, 6)}; ps = {k: np.load(os.path.join(OOF, f'{tag}_gb_fold{k}.npy')) for k in range(1, 6)}
    y = np.concatenate([ys[k] for k in range(1, 6)]); p = np.concatenate([ps[k] for k in range(1, 6)]); pi = y.mean()
    rows = sweep(y, p, grid)
    ident = max(abs(P - (pi * R) / (pi * R + (1 - pi) * FPR)) for t, P, R, FPR, F in rows if pi * R + (1 - pi) * FPR > 0)
    best = max(rows, key=lambda r: r[4])
    nest = []
    for k in range(1, 6):
        yt = np.concatenate([ys[j] for j in range(1, 6) if j != k]); pt = np.concatenate([ps[j] for j in range(1, 6) if j != k])
        tk = max(sweep(yt, pt, grid), key=lambda r: r[4])[0]; nest.append(sweep(ys[k], ps[k], [tk])[0][4])
    g = J(os.path.join(HERE, 'results', tag, 'summary.json'))['gb_threshold_study']
    rep['threshold'][tag] = {'max_identity_abs_diff': float(ident), 'grid_optimum': float(best[0]), 'grid_optimum_f1': float(best[4]),
                             'nested_f1_mean': float(np.mean(nest)), 'summary_grid_optimum': g['grid_optimum']['threshold'], 'summary_nested_f1_mean': g['nested']['f1_mean']}
    L.append(f"| {tag} | {ident:.1e} | {best[0]:.2f} / {g['grid_optimum']['threshold']:.2f} | {np.mean(nest):.4f} / {g['nested']['f1_mean']:.4f} |")

# 5. rank orders vs the paper's Tables 1-2 (saved arms), Kendall/Spearman, and the pre-registered decision rule
def paper(arm):
    d = J(os.path.join(PAPER, arm, 'experiment_results.json')); r = d.get('results', d)
    return {m: {'f1': v['f1']['mean'], 'auc_pr': v['auc_pr']['mean']} for m, v in r.items() if isinstance(v, dict) and 'f1' in v}
s = J(os.path.join(HERE, 'results', 'primary_is_fraud', 'summary.json')); models = s['complete_models']; rep['ranks'] = {}
L.append("\n## 5. Rank orders vs the paper (primary run, six models)")
for metric in ['f1', 'auc_pr']:
    ours = {NAMES[m]: s['models'][m][metric]['mean'] for m in models}; order = sorted(ours, key=lambda n: -ours[n])
    L.append(f"\n**{metric.upper().replace('_', '-')}** — CMS–LEIE: {' > '.join(order)}")
    for arm in ['synthetic', 'kaggle_medicare']:
        pp = paper(arm); names = list(ours); a = [ours[n] for n in names]; b = [pp[n][metric] for n in names]
        tau = kendalltau(a, b); rho = spearmanr(a, b); porder = sorted(names, key=lambda n: -pp[n][metric])
        top2 = set(order[:2]) == set(porder[:2]); rule = top2 and tau.statistic >= 0.6
        rep['ranks'][f'{metric}_{arm}'] = {'paper_order': porder, 'kendall_tau': float(tau.statistic), 'kendall_p': float(tau.pvalue),
                                          'spearman_rho': float(rho.statistic), 'top2_identical': top2, 'rule_met': rule}
        L.append(f"- vs {arm}: {' > '.join(porder)} — τ = {tau.statistic:.2f} (p = {tau.pvalue:.2f}), ρ = {rho.statistic:.2f}; "
                 f"top-2 identical: **{'yes' if top2 else 'no'}**; rule (top-2 identical and τ ≥ 0.6): **{'met' if rule else 'not met'}**")
    rep['ranks'][f'cms_order_{metric}'] = order

# 6. sensitivities: present, plus effective training ratio and test prevalence per label (protocol 13:20 UTC note)
L.append("\n## 6. Sensitivities — effective class balance\n| run | train rows | train positives | effective neg:pos | test prevalence |\n|---|---|---|---|---|")
rep['sens'] = {}
for tag, (setname, label) in TAGS.items():
    rs = [J(p) for p in sorted(glob.glob(os.path.join(HERE, 'out', f'resample_{setname}_{label}_[1-5]{"_r50" if tag.endswith("r50") else ""}.json')))]
    rows = sum(r['rows_before'] for r in rs); pos = sum(r['pos_before'] for r in rs)
    y = np.concatenate([ytest(setname, label, k) for k in range(1, 6)])
    rep['sens'][tag] = {'folds': len(rs), 'train_rows': rows, 'train_pos': pos, 'neg_per_pos': (rows - pos) / pos, 'test_prevalence': float(y.mean())}
    L.append(f"| {tag} | {rows:,} | {pos:,} | {(rows - pos) / pos:.1f}:1 | {y.mean() * 100:.4f}% |")

# 7. chronic-condition missingness behind cc_missing (2017-2024, kept rows)
CC = [c for c in pq.read_schema(os.path.join(DATA, 'by_provider_2017.parquet')).names if c.startswith('Bene_CC')]
L.append("\n## 7. Chronic-condition fields: why `cc_missing` is almost always 1\n| year | rows | any CC blank | all 25 blank | median blank fields | rows with Tot_Benes < 11 |\n|---|---|---|---|---|---|")
rep['cc'] = {}; per_field = None
for yr in range(2017, 2025):
    t = pq.read_table(os.path.join(DATA, f'by_provider_{yr}.parquet'), columns=['Rndrng_NPI', 'Tot_Benes'] + CC).to_pandas()
    blank = t[CC].apply(lambda c: pd.to_numeric(c, errors='coerce')).isna()
    nb = blank.sum(axis=1); benes = pd.to_numeric(t['Tot_Benes'], errors='coerce')
    rep['cc'][yr] = {'rows': len(t), 'any_blank': float((nb > 0).mean()), 'all_blank': float((nb == len(CC)).mean()), 'median_blank': float(nb.median()), 'benes_lt11': float((benes < 11).mean())}
    pf = blank.mean(); per_field = pf if per_field is None else per_field + pf
    L.append(f"| {yr} | {len(t):,} | {(nb > 0).mean() * 100:.1f}% | {(nb == len(CC)).mean() * 100:.1f}% | {nb.median():.0f} | {(benes < 11).mean() * 100:.1f}% |")
per_field = (per_field / 8).sort_values(ascending=False)
rep['cc_field_blank_share'] = per_field.round(4).to_dict()
L.append("\nShare blank per field (mean over 2017–2024), most to least: " + '; '.join(f"{c.replace('Bene_CC_', '').replace('_Pct', '')} {v * 100:.0f}%" for c, v in per_field.items()))
json.dump(rep, open(os.path.join(HERE, 'gate3_report.json'), 'w'), indent=1, default=float)
print('\n'.join(L))
