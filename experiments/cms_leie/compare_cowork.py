#!/usr/bin/env python3
"""Compare this Mac run with the Cowork cloud run kept in cowork_reference/.
Usage: compare_cowork.py downloads | reports | results <tag>
Exit code 1 on any mismatch. Metric tolerance: four-decimal agreement (|diff| < 5e-5); confusion counts must be equal."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); REF = os.path.join(HERE, 'cowork_reference')
def load(p):
    strip = lambda o: {k: strip(v) for k, v in o.items() if k != 'note'} if isinstance(o, dict) else o   # Cowork's free-text notes
    return strip(json.load(open(p)))
bad = []

def downloads():
    a, b = load(os.path.join(HERE, 'out', 'download_manifest.json')), load(os.path.join(REF, 'out', 'download_manifest.json'))
    print('| file | Mac SHA-256 (12) | Cowork SHA-256 (12) | bytes equal | match |\n|---|---|---|---|---|')
    for k in sorted(b):
        if k not in a: bad.append(k); print(f'| {k} | missing | {b[k]["sha256"][:12]} | – | NO |'); continue
        ok = a[k]['sha256'] == b[k]['sha256']
        if not ok: bad.append(k)
        print(f'| {k} | {a[k]["sha256"][:12]} | {b[k]["sha256"][:12]} | {a[k]["bytes"] == b[k]["bytes"]} | {"yes" if ok else "NO"} |')

def reports():
    for f in ['label_report.json', 'scaler_stats.json', 'folds_report.json', 'feature_list.json', 'category_vocab.json']:
        p = os.path.join(HERE, 'out', f)
        if not os.path.exists(p): bad.append(f); print(f'{f}: MISSING'); continue
        ok = load(p) == load(os.path.join(REF, 'out', f))
        if not ok: bad.append(f)
        print(f'{f}: {"identical" if ok else "DIFFERS"}')

def results(tag):
    a, b = load(os.path.join(HERE, 'results', tag, 'summary.json')), load(os.path.join(REF, 'results', tag, 'summary.json'))
    mets = ['f1', 'precision', 'recall', 'mcc', 'auc_pr', 'auc_roc']; cnt = ['tp', 'fp', 'fn', 'tn']
    print('| model | fold | max |metric diff| | counts equal | match |\n|---|---|---|---|---|')
    for m, mb in b['models'].items():
        for k, fb in mb.get('folds', {}).items():
            fa = a['models'].get(m, {}).get('folds', {}).get(k)
            if fa is None: bad.append(f'{m}/{k}'); print(f'| {m} | {k} | missing | – | NO |'); continue
            d = max(abs(fa[x] - fb[x]) for x in mets if x in fb); ce = all(fa.get(x) == fb.get(x) for x in cnt if x in fb)
            ok = d < 5e-5 and ce
            if not ok: bad.append(f'{m}/{k}')
            print(f'| {m} | {k} | {d:.2e} | {ce} | {"yes" if ok else "NO"} |')

{'downloads': downloads, 'reports': reports, 'results': lambda: results(sys.argv[2])}[sys.argv[1]]()
print('RESULT:', 'ALL MATCH' if not bad else f'MISMATCH {bad}')
sys.exit(1 if bad else 0)
