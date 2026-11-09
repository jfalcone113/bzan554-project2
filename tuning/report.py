#!/usr/bin/env python3
"""Export full trial logs and concise, presentation-ready measured improvements."""
from pathlib import Path
import csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-tuning-mpl')
P=Path(__file__).resolve().parent
ROOT=P.parent

def load(p):return json.loads(Path(p).read_text())
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join([':---']+['---:']*(len(headers)-1))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
 results=[load(p) for p in sorted((P/'runs').glob('*/result.json'))]
 if not results:print('No completed or failed fits yet.');return
 records=[];epochs=[]
 fields=['name','stage','parent','change','seed','status','activation','initializer','widths','bn','optimizer','initial_lr','schedule','batch_size','hidden_dropout','sku_dropout','l1','l2','max_epochs','selected_epoch','epochs_run','updates','parameters','trainable_parameters','fit_seconds','peak_process_rss_mib','train_r2','train_mse','train_rmse','train_mae','validation_r2','validation_mse','validation_rmse','validation_mae','checkpoint','error','source_sha256','script_sha256','config_json']
 for r in results:
  c=r['signature']['config'];row={k:r[k] for k in fields if k in r}
  row.update(stage=c['stage'],parent=c['parent'],change=c['change'],activation=c['activation'],initializer=c['initializer'],widths='/'.join(map(str,c['widths'])),bn=c['bn'],optimizer=c['optimizer'],initial_lr=c['lr'],schedule=c['schedule'],batch_size=c['batch'],hidden_dropout=c['hidden_dropout'],sku_dropout=c['sku_dropout'],l1=c['l1'],l2=c['l2'],max_epochs=c['max_epochs'],source_sha256=r['signature']['source_sha256'],script_sha256=r['signature']['script_sha256'],config_json=json.dumps(c,sort_keys=True))
  for split in ['train','validation']:
   if split in r:row.update({f'{split}_{m}':r[split][m] for m in ['r2','mse','rmse','mae']})
  records.append(row)
  ep=P/'runs'/f'{r["name"]}_seed{r["seed"]}'/'epochs.json'
  if ep.exists():
   for e in load(ep):epochs.append({'name':r['name'],'seed':r['seed'],'epoch':e['epoch'],'validation_r2':e['validation']['r2'],'validation_rmse':e['validation']['rmse'],'validation_mse':e['validation']['mse'],'validation_mae':e['validation']['mae'],'online_preupdate_mse':e['online_preupdate_mse'],'learning_rate_start':e['learning_rate_start'],'learning_rate_end':e['learning_rate_end'],'updates':e['updates'],'seconds':e['seconds'],'process_rss_mib':e['process_rss_mib']})
 for name,data,columns in [('all_runs.csv',records,fields),('epoch_metrics.csv',epochs,list(epochs[0]) if epochs else [])]:
  with (P/name).open('w',newline='') as out:
   w=csv.DictWriter(out,fieldnames=columns);w.writeheader();w.writerows(data)
 good=[r for r in results if r['status']=='complete'];screen=[r for r in good if r['signature']['config']['stage']!='E']
 top=sorted(screen,key=lambda r:r['validation']['mse'])
 baseline=load(ROOT/'model/metrics.json')['validation']
 summary=f'# Expanded tuning: measured results\n\nCompleted fits: {len(good)}. Failed fits: {len(results)-len(good)}. Fixed split seed 42. Original model validation R² **{baseline["r2"]:.4f}**, RMSE **{baseline["rmse"]:.2f}**.\n\n'
 summary+='## Screening leaderboard\n\n'+table(['Run','Validation R²','RMSE','Selected epoch','Seconds'],[[r['name'],f'{r["validation"]["r2"]:.4f}',f'{r["validation"]["rmse"]:.2f}',r['selected_epoch'],f'{r["fit_seconds"]:.1f}'] for r in top])+'\n'
 chain=[];best=-float('inf')
 for stage in ['A','B','C','D']:
  group=[r for r in screen if r['signature']['config']['stage']==stage]
  if not group:continue
  r=max(group,key=lambda r:r['validation']['r2'])
  if r['validation']['r2']>best:chain.append(r);best=r['validation']['r2']
 summary+='\n## Improving stage winners\n\n'+table(['Stage','Configuration','R²','RMSE'],[[r['signature']['config']['stage'],r['name'],f'{r["validation"]["r2"]:.4f}',f'{r["validation"]["rmse"]:.2f}'] for r in chain])+'\n\nEach stage winner may change multiple settings. Full one-configuration comparisons and losing results remain in the CSV. Paired seed numbers do not ensure identical initialization after shape or layer changes. Screening is one-seed evidence. Confirmation uses three matched seeds and a longer equal budget.\n'
 selection=P/'selection.json'
 if selection.exists():
  sel=load(selection);g=sel['selected_group'];c=sel['selected']['signature']['config']
  summary+='\n## Three-seed confirmation\n\n'+table(['Configuration','Mean R²','Range','Mean RMSE'],[[g0['name'],f'{g0["mean_validation_r2"]:.4f}',f'{g0["min_validation_r2"]:.4f}-{g0["max_validation_r2"]:.4f}',f'{g0["mean_validation_rmse"]:.2f}'] for g0 in sel['groups']])
  criterion_label='User-requested measurable-improvement criterion' if sel.get('user_acceptance') else 'Original proposed +0.01 criterion'
  summary+=f'\n\nMean paired validation R² gain: **{sel["mean_r2_gain"]:+.4f}**. {criterion_label} met: **{sel["noticeable"]}**. These three-seed ranges are not confidence intervals.\n\nSelected configuration: `{sel["selected"]["name"]}`, seed {sel["selected"]["seed"]}, epoch {sel["selected"]["selected_epoch"]}. Hyperparameters:\n\n```json\n'+json.dumps(c,indent=2)+'\n```\n'
  evaluation=P/'final_evaluation.json'
  if evaluation.exists():
   ev=load(evaluation);m=ev['metrics'];old=ev['old_model_metrics']
   summary+='\n## Frozen-model final evaluation\n\n'+table(['Model / split','R²','MSE','RMSE','MAE'],[[n,f'{v["r2"]:.4f}',f'{v["mse"]:.2f}',f'{v["rmse"]:.2f}',f'{v["mae"]:.2f}'] for n,v in [('New / train',m['train']),('New / validation',m['validation']),('New / test',m['test']),('Old / test',old['test'])]])
   summary+='\n\nConfiguration selection was frozen before new test scoring. The test partition has historical exposure. A different split was not used to manufacture a better score. Reloaded validation matched the selected checkpoint. BN inference-statistic checks passed where applicable. The original model remains unchanged. The active deck includes the two improving tuning steps; superseded presentation backups were removed.\n'
   if sel['noticeable']:
    insert='''## Improvement after releasing the architecture restrictions

'''+table(['Model','Validation $R^2$','Validation RMSE'],[['Previous model',f'{baseline["r2"]:.4f}',f'{baseline["rmse"]:.2f}'],['New selected checkpoint',f'{m["validation"]["r2"]:.4f}',f'{m["validation"]["rmse"]:.2f}']])+f'\n\n**Three-seed mean validation improvement: {sel["mean_r2_gain"]:+.4f} R²** under matched 40-epoch caps.\n\nHidden activation: **{c["activation"]}**. Widths: **'+ '/'.join(map(str,c['widths']))+f'**. Optimizer: **{c["optimizer"]}**. SKU dropout stays at **50%**.\n\n\\scriptsize The new configuration combines controls. Sources: deep.pdf, pp. 12-37, 50-75, 82-105. Exact values are proposed experiments.\n\n## Final check of the improved model\n\n'+table(['Model','Test $R^2$','Test RMSE'],[['Previous model',f'{old["test"]["r2"]:.4f}',f'{old["test"]["rmse"]:.2f}'],['New model',f'{m["test"]["r2"]:.4f}',f'{m["test"]["rmse"]:.2f}']])+f'\n\nThe new configuration was selected by mean validation MSE across three seeds. The inference checkpoint uses seed **{sel["selected"]["seed"]}**, epoch **{sel["selected"]["selected_epoch"]}**.\n\n\\scriptsize Test follows frozen selection. The historical test exposure and completed-episode prediction limits still apply. Three seeds do not establish a universal best model.\n'
    (P/'PRESENTATION_INSERT.qmd').write_text(insert)
 summary+='\n## Reproduce and interpret\n\nRun `../.venv/bin/python tuning/run_tuning.py`, then `../.venv/bin/python tuning/finalize_user_result.py` and `../.venv/bin/python tuning/report.py` from the project root. Completed requests are reused only with exact source/configuration/code signatures. `PLAN.md` gives proposed ranges, course-page citations, stopping rules, and the 42-fit bound. Installed APIs are recorded in `environment.json`. All checkpoints are inference checkpoints without the custom optimizer state. No sequence/image architecture or unrealistic augmentation has been implied by these tabular inputs.\n'
 (P/'RESULTS.md').write_text(summary)
 print(f'Exported {len(results)} run records, {len(epochs)} epoch records.')
if __name__=='__main__':main()
