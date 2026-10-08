#!/usr/bin/env python3
"""Paper T — Results tables generated from results/asof/summary.json and labels_report.json (Rule 0d: every number traces to the run
output; nothing is typed by hand). Usage: paperT_results_from_summary.py RES_DIR OUT.md"""
import os, sys, json
RES=sys.argv[1]; out=sys.argv[2]; S=json.load(open(os.path.join(RES,'summary.json'))); L=json.load(open(os.path.join(RES,'labels_report.json')))
T=[t for t in ['2018','2019','2020','2021','2022'] if t in S['years']]; B=S['B']
def f(x,d=3): return '—' if x is None else (f'{x:.{d}f}' if isinstance(x,(int,float)) else str(x))
M=['# Paper T — Results (generated from summary.json; protocol sha256 '+S['protocol_sha256']+f'; bootstrap B = {B})','']
M+=['## Table 1. As-of training sets and label completeness','','| t | training rows (year ≤ t−1) | eventual positives among them | known at 1 Jan t | completeness | test rows (year t) | positives (eventual) | positives within 3 years |','|---|---|---|---|---|---|---|---|']
for t in T:
    r=L[t]; M.append(f"| {t} | {r['train_rows']:,} | {r['train_eventual_positives']:,} | {r['train_known_positives']:,} | {r['share_known']:.3f} | {r['test_rows']:,} | {r['test_positives']:,} | {r['test_positives_3yr']:,} |")
M+=['',f"Rows kept: {L['rows_kept']:,}; eventual positives: {L['positives']:,}; NPIs with an LEIE end date: {L['npis_with_end_date']:,}; excl_year mismatches vs the companion arm: {L['excl_year_mismatches_vs_arm']}.",'']
M+=['## Table 2. As-of performance by year (budgets within year t; primary label = eventual exclusion)','','| t | variant/model | recall@0.5% | recall@1% | precision@1% | lift@1% | $-recall@1% | recall@2% | recall@5% | AUC-PR | AUC-ROC |','|---|---|---|---|---|---|---|---|---|---|---|']
for t in T:
    Y=S['years'][t]
    for v in ['gb_A','rf_A','gb_B','rf_B']:
        if v not in Y['asof']: continue
        a=Y['asof'][v]; M.append(f"| {t} | {v} | {f(a['0.005']['recall'])} | {f(a['0.01']['recall'])} | {f(a['0.01']['precision'],4)} | {f(a['0.01']['lift'],1)} | {f(a['0.01']['dollar_recall'])} | {f(a['0.02']['recall'])} | {f(a['0.05']['recall'])} | {f(a['auc_pr'],4)} | {f(a['auc_roc'])} |")
M+=['','### Table 2b. Three-year label (exclusion recorded by t+3): recall@1 %, AUC-PR, AUC-ROC','','| t | gb_A | rf_A | gb_B | rf_B |','|---|---|---|---|---|']
for t in T:
    Y=S['years'][t]; M.append('| '+t+' | '+' | '.join(f"{f(Y['asof_3yr'][v]['0.01']['recall'])} / {f(Y['asof_3yr'][v]['auc_pr'],4)} / {f(Y['asof_3yr'][v]['auc_roc'])}" if v in Y['asof_3yr'] else '—' for v in ['gb_A','rf_A','gb_B','rf_B'])+' |')
M+=['','## Figure 1 data. Hindsight (cross-validated, labels current at the study date) vs as-of lift@1 % and the hindsight-optimism ratio','','| t | completeness | as-of GB-A lift@1% | hindsight GB lift@1% | ratio (GB) | as-of RF-A lift@1% | hindsight RF lift@1% | as-of GB-A AUC-ROC | hindsight GB AUC-ROC |','|---|---|---|---|---|---|---|---|---|']
for t in T:
    Y=S['years'][t]; h=Y.get('hindsight',{}); g=Y['asof']['gb_A']; r=Y['asof'].get('rf_A',{})
    M.append(f"| {t} | {Y['share_of_training_positives_known_at_t']:.3f} | {f(g['0.01']['lift'],1)} | {f(h.get('gb',{}).get('0.01',{}).get('lift'),1)} | {f(Y.get('hindsight_optimism_ratio_gb'),2)} | {f(r.get('0.01',{}).get('lift'),1)} | {f(h.get('rf',{}).get('0.01',{}).get('lift'),1)} | {f(g['auc_roc'])} | {f(h.get('gb',{}).get('auc_roc'))} |")
M+=['','## Table 3. Operating decisions at a 1 % budget (as-of GB-A scores)','','| t | rule | rows caught | later-excluded NPIs | precision | $-recall |','|---|---|---|---|---|---|']
for t in T:
    tr=S['years'][t].get('triage_1pct',{})
    for name in ['p','p_x_payment','payment_only']:
        if name in tr: x=tr[name]; M.append(f"| {t} | {name} | {x['tp']:,} | {x['distinct_npis']:,} | {f(x['precision'],4)} | {f(x['dollar_recall'])} |")
M+=['','| t | global recall@1% | yield-aware recall@1% | difference |','|---|---|---|---|']
for t in T:
    al=S['years'][t].get('allocation_1pct',{})
    if al: M.append(f"| {t} | {f(al['global_recall'])} | {f(al['yield_aware_recall'])} | {al['yield_aware_recall']-al['global_recall']:+.3f} |")
M+=['','## Table 4. Pre-specified decision rules (reported as printed by asof_metrics.py)','','| rule | criterion | years met | 5-year mean difference | 95 % bootstrap CI | result |','|---|---|---|---|---|---|']
for k,v in S['decision_rules'].items():
    M.append(f"| {k} | {v['rule']} | {v['years_met']}/{len(T)} | {f(v.get('mean_diff'),4)} | {('{:.4f} to {:.4f}'.format(*v['ci']) if 'ci' in v else '—')} | **{'MET' if v['met'] else 'NOT MET'}** |")
if 'T1' in S['decision_rules']: M.append(f"\nT1 ratios by year: {', '.join(f(x,2) for x in S['decision_rules']['T1']['ratios'])}.")
if 'T3' in S['decision_rules']: M.append(f"T3 dollar-recall differences (p × payment − p) by year: {', '.join(f(x) for x in S['decision_rules']['T3']['diffs'])}.")
open(out,'w').write('\n'.join(M)+'\n'); print('\n'.join(M))
