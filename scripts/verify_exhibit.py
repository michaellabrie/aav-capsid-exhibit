#!/usr/bin/env python3
"""Dependency-free verification of the shipped test records and exported MLP.
Run: python3 scripts/verify_exhibit.py
"""
import csv
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
model=json.loads((ROOT/'model/network.json').read_text())
tests=json.loads((ROOT/'data/heldout_test.json').read_text())
metrics=model['metrics']
assert len(tests)==metrics['test_count']==900
assert metrics['train_count']+metrics['validation_count']+metrics['test_count']==metrics['sample_count']==6000
rows=list(csv.DictReader((ROOT/'data/synthetic_aav_assembly.csv').open(newline='')))
assert len(rows)==6000
assert {k:sum(r['split']==k for r in rows) for k in ('train','validation','test')}=={'train':4200,'validation':900,'test':900}
assert set(r['sample_id'] for r in rows if r['split']=='test')==set(r['id'] for r in tests)

def forward(r):
    x=[r['aap'],r['contact'],r['stability'],r['ph'],r['salt'],r['concentration'],*[int(r['serotype']==s) for s in ['AAV2','AAV5','AAV8','AAV9']]]
    x=[(v-model['mean'][i])/model['scale'][i] for i,v in enumerate(x)]
    for layer in model['layers']:
        z=[bias+sum(x[i]*layer['weights'][i][j] for i in range(len(x))) for j,bias in enumerate(layer['bias'])]
        x=[max(0,v) for v in z] if layer['activation']=='relu' else [1/(1+math.exp(-v)) for v in z]
    return x[0]

diff=max(abs(forward(r)-r['prob']) for r in tests)
assert diff < 0.00015, f'Browser weights disagree with case predictions by {diff}'

threshold=metrics['threshold']
tp=tn=fp=fn=0
for r in tests:
    prediction=r['prob']>=threshold
    if r['actual']==1:
        if prediction:tp+=1
        else:fn+=1
    else:
        if prediction:fp+=1
        else:tn+=1
assert [[tn,fp],[fn,tp]]==metrics['confusion_matrix']
assert math.isclose((tn+tp)/len(tests), metrics['accuracy'])
assert math.isclose(tp/(tp+fp),metrics['precision'])
assert math.isclose(tp/(tp+fn),metrics['recall'])
print('PASS: all 6,000 records, 4,200/900/900 splits, 900 exported cases, browser weights, held-out accuracy and confusion matrix match.')
print('Maximum probability difference after CSV/test-record rounding:',round(diff,8))
