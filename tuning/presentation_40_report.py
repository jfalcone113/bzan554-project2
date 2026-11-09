#!/usr/bin/env python3
"""Export the slide-focused 40-cap follow-up without altering historical study logs."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'tuning/presentation_40'
def load(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 protocol=load(P/'protocol.json');rows=[]
 for kind,name in protocol['scope']:
  r=load(P/'runs'/name/'result.json');assert r['status']=='complete'
  history=load(P/'runs'/name/'epochs.json')
  if kind=='restore':
   original=next(x for x in load(ROOT/'presentation/results_20_epoch.json' if (ROOT/'presentation/results_20_epoch.json').exists() else ROOT/'presentation/results.json') if x['name']==name)['validation']['r2']
  else:original=load(ROOT/'tuning/runs'/f'{name}_seed42/epochs.json')[19]['validation']['r2']
  assert abs(history[19]['validation']['r2']-original)<1e-5,(name,'reproduction mismatch')
  rows.append(r)
 reused=[]
 for alias,name in [('09_checkpoint','e_baseline'),('b_rmsprop001','e_finalist2'),('d_l2_low','e_finalist1')]:
  r=load(ROOT/'tuning/runs'/f'{name}_seed42/result.json');r['source_run_name']=r['name'];r['name']=alias;rows.append(r);reused.append(alias)
 dump(P/'results.json',rows)
 fields=['name','seed','reused','selected_epoch','best_observed_epoch','epochs_run','parameters','fit_seconds','train_r2','train_rmse','train_mse','train_mae','validation_r2','validation_rmse','validation_mse','validation_mae','config_json','checkpoint']
 with (P/'all_runs.csv').open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
  for r in rows:
   row={k:r[k] for k in fields if k in r};row['reused']=r['name'] in reused;row['config_json']=json.dumps(r['signature']['config'],sort_keys=True)
   for part in ['train','validation']:
    row.update({f'{part}_{m}':r[part][m] for m in ['r2','rmse','mse','mae']})
   w.writerow(row)
 text='# Presentation trials: 40-epoch-cap follow-up\n\nTen new seed-42 fits and three reused seed-42 confirmation fits. Same source data and split. Only epoch cap changes. Early restoration steps still evaluate final-epoch weights; expanded trials and steps 6–8 restore best validation checkpoints. No new test evaluation or selected-model replacement.\n\nAll new runs must reproduce their original epoch-20 validation R² within 0.00001 before acceptance.\n\n| Trial | Validation R² | RMSE | MSE | MAE | Selected / completed epoch | Reused |\n|---|---:|---:|---:|---:|---:|---|\n'
 for r in rows:text+=f"| {r['name']} | {r['validation']['r2']:.6f} | {r['validation']['rmse']:.4f} | {r['validation']['mse']:.4f} | {r['validation']['mae']:.4f} | {r['selected_epoch']} / {r['epochs_run']} | {r['name'] in reused} |\n"
 (P/'RESULTS.md').write_text(text)
 by={r['name']:r for r in rows};names=['00_basic','01_log','02_sqrt','03_hidden_dropout','04_sku_dropout','05_l2','09_checkpoint','b_rmsprop001','d_l2_low']
 # Keep frozen 20-cap provenance separate from the new presentation inputs.
 old=ROOT/'presentation/results_20_epoch.json'
 if not old.exists():old.write_bytes((ROOT/'presentation/results.json').read_bytes())
 dump(ROOT/'presentation/results_40_epoch.json',[by[n] for n in names])
 original=load(ROOT/'presentation/non_improving_trials.json')
 out={'comparison':'All four displayed branches use seed42 and cap40, matched to RMSprop L2 .0001 confirmation.','rows':[]}
 parent=by['b_rmsprop001']['validation']['r2']
 for row in original['rows']:
  if row['parent']!='b_rmsprop001':continue
  r=by[row['trial']];out['rows'].append({**row,'epoch_cap':40,'baseline_r2':parent,'result_r2':r['validation']['r2'],'delta_r2':r['validation']['r2']-parent})
 dump(ROOT/'presentation/non_improving_trials_40.json',out)
 print(text)
if __name__=='__main__':main()
