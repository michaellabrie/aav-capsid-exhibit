#!/usr/bin/env python3
"""Reproducibly generate a synthetic AAV teaching dataset, train/test a small NN,
and export weights for exact browser-side inference. NOT biological observations.
"""
from pathlib import Path
import json
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 20261119
rng = np.random.default_rng(RNG_SEED)
N = 6000
serotypes = np.array(['AAV2','AAV5','AAV8','AAV9'])
serotype = rng.choice(serotypes, N, p=[.27,.24,.24,.25])
aap = rng.integers(0,2,N)
contact = rng.uniform(.12,.98,N)
stability = rng.uniform(.16,.98,N)
ph = rng.uniform(5.8,8.0,N)
salt = rng.uniform(75,225,N)
concentration = rng.uniform(.15,.98,N)
# Generating equation creates educational labels and is *not* a mechanistic model.
# Nonlinear terms make a small neural network an appropriate illustrative learner.
z = (
    0.25 + 2.8*(contact-.55) + 2.55*(stability-.55)
    + 1.75*(concentration-.5) - .65*np.abs(ph-7.15)
    - .32*(np.abs(salt-145)/90)
    + np.where(serotype=='AAV2', -1.75*(1-aap) + .42*aap, 0.0)
    + np.where(serotype=='AAV5', .48 + .14*aap, 0.0)
    + np.where(serotype=='AAV8', -.35*(1-aap)+.18*aap, 0.0)
    + np.where(serotype=='AAV9', -.52*(1-aap)+.24*aap, 0.0)
    + .72*(contact>.7)*(stability>.68)
    + .30*np.sin((ph-6.2)*3)
)
prob = 1/(1+np.exp(-1.55*z))
y = rng.binomial(1,prob)
FEATURES=['aap_present','interface_score','vp_stability','ph','ionic_strength_mm','vp_concentration','is_aav2','is_aav5','is_aav8','is_aav9']
X = np.column_stack([aap,contact,stability,ph,salt,concentration,*[(serotype==s).astype(int) for s in serotypes]]).astype(float)
all_indices=np.arange(N)
train_val,test=train_test_split(all_indices,test_size=.15,random_state=17,stratify=y)
train,val=train_test_split(train_val,test_size=900,random_state=17,stratify=y[train_val])
scaler=StandardScaler()
train_s=scaler.fit_transform(X[train]); val_s=scaler.transform(X[val]); test_s=scaler.transform(X[test])
model=MLPClassifier(hidden_layer_sizes=(14,8), activation='relu',solver='adam',learning_rate_init=.002, alpha=.0005, max_iter=500, random_state=17, early_stopping=False,n_iter_no_change=35)
model.fit(train_s,y[train])
val_prob=model.predict_proba(val_s)[:,1]
thresholds=np.linspace(.2,.8,121)
threshold=float(sorted([(float(f1_score(y[val],val_prob>=th)),float(th)) for th in thresholds],key=lambda p:(p[0],-abs(p[1]-.5)),reverse=True)[0][1])
test_prob=model.predict_proba(test_s)[:,1]
yp=(test_prob>=threshold).astype(int)
base=int(np.mean(y[train])>=.5)
metrics={
 'sample_count':N,'train_count':len(train),'validation_count':len(val),'test_count':len(test),
 'seed':RNG_SEED, 'model':'MLPClassifier (10 → 14 → 8 → 1, ReLU)',
 'threshold':round(threshold,3),
 'accuracy':float(accuracy_score(y[test],yp)),
 'baseline_accuracy':float(accuracy_score(y[test],np.full_like(y[test],base))),
 'precision':float(precision_score(y[test],yp,zero_division=0)),
 'recall':float(recall_score(y[test],yp,zero_division=0)),
 'f1':float(f1_score(y[test],yp,zero_division=0)),
 'roc_auc':float(roc_auc_score(y[test],test_prob)),
 'confusion_matrix':confusion_matrix(y[test],yp,labels=[0,1]).tolist(),
 'positive_rate_test':float(np.mean(y[test])),
 'majority_class':base,
 'training_iterations':model.n_iter_,
}
print(json.dumps(metrics,indent=2))
# Export full learned MLP: the browser evaluates EXACT trained layers, never a mock probability.
model_json={
 'kind':'sklearn_MLPClassifier', 'features':FEATURES,
 'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
 'layers':[{'weights':coef.tolist(),'bias':bias.tolist(),'activation':'relu' if i<len(model.coefs_)-1 else 'sigmoid'} for i,(coef,bias) in enumerate(zip(model.coefs_,model.intercepts_))],
 'metrics':metrics,'source':'synthetic_educational_data','seed':RNG_SEED,
}
(ROOT/'model'/'network.json').write_text(json.dumps(model_json,separators=(',',':')), encoding='utf-8')
(ROOT/'model'/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
with (ROOT/'data'/'synthetic_aav_assembly.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['sample_id','split','serotype','aap_present','interface_score','vp_stability','ph','ionic_strength_mm','vp_concentration','assembled'])
 split={int(i):'train' for i in train};split.update({int(i):'validation' for i in val});split.update({int(i):'test' for i in test})
 for i in range(N):
  w.writerow([f'SYN-{i+1:04}',split[i],serotype[i],int(aap[i]),f'{contact[i]:.5f}',f'{stability[i]:.5f}',f'{ph[i]:.4f}',f'{salt[i]:.3f}',f'{concentration[i]:.5f}',int(y[i])])
# Only heldout test examples go to the web exhibit
records=[]
for i,pred in zip(test,test_prob):
 row={'id':f'SYN-{i+1:04}', 'serotype':str(serotype[i]),'aap':int(aap[i]),'contact':round(float(contact[i]),5),'stability':round(float(stability[i]),5),'ph':round(float(ph[i]),4),'salt':round(float(salt[i]),3),'concentration':round(float(concentration[i]),5),'actual':int(y[i]),'prob':round(float(pred),7)}
 records.append(row)
(ROOT/'data'/'heldout_test.json').write_text(json.dumps(records,separators=(',',':')),encoding='utf-8')
# Validate forward pass mathematically with numpy
z_check=(X[test][:10]-scaler.mean_)/scaler.scale_
for i,(w,b) in enumerate(zip(model.coefs_,model.intercepts_)):
 z_check=z_check@w+b
 if i<len(model.coefs_)-1:z_check=np.maximum(0,z_check)
 else:z_check=1/(1+np.exp(-z_check))
assert np.max(abs(z_check.ravel()-test_prob[:10]))<1e-10
print(f'Browser forward pass validated on first 10 held-out predictions (max error {np.max(abs(z_check.ravel()-test_prob[:10])):.3g})')
