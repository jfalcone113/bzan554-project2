"""Present the measured 40-cap progression and matched confirmation."""
from pathlib import Path
import json,csv
P=Path(__file__).resolve().parent

def load(name):return json.loads((P/name).read_text())
def slide(title,body):return f'\n## {title}\n\n{body}\n'
def table(items):
 return '\n'.join(['| Model | Validation $R^2$ | Validation RMSE |','|:---|---:|---:|']+[f'| {n} | {r["validation"]["r2"]:.4f} | {r["validation"]["rmse"]:.2f} |' for n,r in items])

def main():
 by={r['name']:r for r in load('results_40_epoch.json')}
 names=['00_basic','01_log','02_sqrt','03_hidden_dropout','04_sku_dropout','05_l2','09_checkpoint','b_rmsprop001','d_l2_low']
 rows=[by[n] for n in names]
 update=load('tuning_update.json')
 with (P.parent/'tuning/all_runs.csv').open() as source:
  confirmation42={r['name']:r for r in csv.DictReader(source) if r['stage']=='E' and r['seed']=='42'}
 assert load('selection.json')['winner']['name']==update['selected']['name']
 assert all((r['signature'].get('max_epochs') or r['signature']['config'].get('max_epochs'))==40 for r in rows)
 titles=['Baseline','Add log features','Add square-root features','Add hidden dropout','Add SKU dropout','Add L2 penalties','Add validation-based training controls','Switch Adam to RMSprop','Reduce the L2 penalty']
 descriptions=[
 'Three sigmoid hidden layers: **256 / 128 / 64** neurons. SKU/category embeddings, standardized raw inputs, linear output, and Adam at 0.001.',
 'Add log1p versions of price, order, and duration alongside the raw inputs. Fit scaling on training data only.',
 'Add square-root versions of price, order, and duration. The model now has nine numeric inputs.',
 'Drop **5%** of activations after each hidden layer during training. Keep this rate in every later model.',
 'Drop the whole SKU embedding for **50%** of training examples. Keep **50%** in every later model.',
 'Add a **0.0001 L2 penalty** to hidden-layer kernels. Evaluation RMSE excludes the penalty.',
 'Keep the best validation checkpoint: **epoch 15**. This combined step also includes norm clipping at 10, learning-rate reductions, and early stopping with patience 12.',
 'Switch **Adam to RMSprop**, keeping the initial learning rate at **0.001** and all other settings fixed.',
 'Lower the hidden-kernel L2 coefficient from **0.0001 to 0.00001**. Keep RMSprop and the previous settings.'
 ]
 q='''---
title: "Improving the Basic Model"
subtitle: "BZAN 554 Group Assignment 2 · Group 6"
author: "Joseph Falcone · Ian Allish · Will Schneider · Noah Hoogerheide"
date: "October 29, 2026"
format:
  beamer:
    slide-level: 2
    aspectratio: 169
    fontsize: 11pt
    pdf-engine: xelatex
    keep-tex: false
header-includes:
  - \\setbeamertemplate{footline}[frame number]
  - \\setbeamertemplate{navigation symbols}{}
---
'''
 q+=slide('How to read the comparisons','- Keep earlier additions; report gains and regressions.\n- SKU dropout stays at **50%** after introduction.\n- All tests train for **up to 40 epochs**; early stopping may end runs sooner.\n- Step tables: same split, seed 42, batch 1,024.\n- Rows: **349,975 train / 74,736 validation / 75,289 test**.\n- Fit preprocessing on training data; select by validation.\n- Early steps use final weights; step 6 adds best-checkpoint selection.\n- Final confirmation: **three seeds** with the same training budget.')
 records=[]
 for i,r in enumerate(rows):
  body=descriptions[i]+'\n\n'
  if i==0:
   body+=table([('Basic model',r)])+'\n\nNo added feature engineering, dropout, L2, clipping, rate reductions, early stopping, or best-epoch selection.'
  else:
   prev=rows[i-1];dr=r['validation']['r2']-prev['validation']['r2'];dm=r['validation']['rmse']-prev['validation']['rmse']
   comparisons=[('Before addition',prev),('After addition',r)]
   body+=table(comparisons)+f'\n\n**Change:** $R^2$ {dr:+.4f}, RMSE {dm:+.2f}.'

  q+=slide(f'{i}. {titles[i]}',body)
  records.append({'step':i,'change':titles[i],'source_run':r['name'],'validation_r2':r['validation']['r2'],'validation_rmse':r['validation']['rmse'],'validation_mse':r['validation']['mse'],'validation_mae':r['validation']['mae'],'train_r2':r['train']['r2'],'train_rmse':r['train']['rmse'],'delta_r2':r['validation']['r2']-rows[i-1]['validation']['r2'] if i else '', 'delta_rmse':r['validation']['rmse']-rows[i-1]['validation']['rmse'] if i else '', 'selected_epoch':r['selected_epoch'],'epochs_run':r['epochs_run']})
 selected=update['selected'];test=update['metrics']['test']
 old_mean=next(g for g in update['groups'] if g['name']=='e_baseline')
 new_mean=next(g for g in update['groups'] if g['name']==selected['name'])
 # Bridge the screening sequence to the different confirmation statistic explicitly.
 confirmation='**Compare seed 42 with the three-seed average.**\n\n'
 confirmation+='| Configuration | Seed 42: $R^2$ | 3-seed mean: $R^2$ |\n|:---|---:|---:|\n'
 confirmation+=f"| Adam + L2 0.0001 (step 6) | {float(confirmation42['e_baseline']['validation_r2']):.4f} | {old_mean['mean_validation_r2']:.4f} |\n"
 confirmation+=f"| RMSprop + L2 0.00001 (step 8) | {float(confirmation42['e_finalist1']['validation_r2']):.4f} | {new_mean['mean_validation_r2']:.4f} |\n"
 confirmation+=f"\n**Matched confirmation gain:** mean $R^2$ **{update['mean_r2_gain']:+.4f}**, positive in **all three seeds**."
 confirmation+='\n\nThe seed-42 scores continue directly from steps 6 and 8. The right column averages seeds 42, 101, and 202.'
 q+=slide('Confirm the improvement across three seeds',confirmation)
 final='**Chosen configuration:** RMSprop + L2 0.00001 (step 8).\n\n'
 final+=f"**Packaged checkpoint:** seed {selected['seed']}, epoch {selected['selected_epoch']}; selected using validation.\n\n"
 final+='| Reported score | $R^2$ | RMSE | MAE |\n|:---|---:|---:|---:|\n'
 final+=f"| Three-seed mean validation | {new_mean['mean_validation_r2']:.4f} | {new_mean['mean_validation_rmse']:.2f} | -- |\n"
 final+=f"| Selected checkpoint: validation | {selected['validation']['r2']:.4f} | {selected['validation']['rmse']:.2f} | {selected['validation']['mae']:.2f} |\n"
 final+=f"| Selected checkpoint: test | {test['r2']:.4f} | {test['rmse']:.2f} | {test['mae']:.2f} |\n"
 final+='\nThe mean summarizes repeatability across seeds. The other rows evaluate the saved checkpoint on validation and test data.'
 q+=slide('Final model: distinguish the reported scores',final)
 declines=load('non_improving_trials_40.json')
 lines=['| Tested change | Before $R^2$ | After $R^2$ | Change |','|:---|---:|---:|---:|']
 for item in declines['rows']:
  if item['parent']!='b_rmsprop001':continue
  result_text=r'$\approx 0$' if abs(item['result_r2'])<.00005 else f"{item['result_r2']:.4f}"
  change_label='SGD' if item['trial']=='b_sgd01' else item['change']
  lines.append(f"| {change_label} | {item['baseline_r2']:.4f} | {result_text} | {item['delta_r2']:+.4f} |")
 comparison='**Each test starts from the same RMSprop model.**\n\n'+'\n'.join(lines)

 comparison+='\n\n**Matched parent:** RMSprop with L2 0.0001, validation $R^2$ **0.6210**. Each row changes one setting from this model; the changes do not accumulate. Same split, seed 42, and training budget.'
 q+=slide('Other experiments: changes that did not help',comparison)
 q+=slide('Evaluation and course sources','- RMSE uses supplied quantity units and excludes L2 penalties.\n- Validation selects settings; test scoring follows frozen selection. The test set has historical exposure.\n- Predictions describe completed selling episodes, not advance forecasts.\n- Results apply to this setup and budget; all trials remain logged.\n\n**Course sources: deep.pdf**\n\nRMSprop: pp. 32-33; clipping: pp. 77-78; checkpoints/stopping: pp. 82-87; L2: pp. 89-91; dropout: pp. 93-98.\n\nExact tuning values and whole-SKU masking are project choices.')
 q+=slide('Team contributions','| Team member | Contribution |\n|:---|:---|\n| Ian Allish | [To be filled in] |\n| Joseph Falcone | [To be filled in] |\n| Will Schneider | [To be filled in] |\n| Noah Hoogerheide | [To be filled in] |\n\n**AI assistance:** Codex implemented the experiments, logged results, and drafted the slides.')
 (P/'Presentation.qmd').write_text(q)
 with (P/'brief_sequence.csv').open('w',newline='') as out:
  w=csv.DictWriter(out,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 md='# Model progression at a 40-epoch cap\n\nSame split, seed 42, batch 1024, and 40-epoch cap. Validation drives selection. These are the original displayed configurations rerun or reused at cap 40; historical cap-20 logs remain separate.\n\n| Step | Change | Validation R² | Validation RMSE |\n|---:|:---|---:|---:|\n'
 for r in records:md+=f'| {r["step"]} | {r["change"]} | {r["validation_r2"]:.4f} | {r["validation_rmse"]:.2f} |\n'
 md+=f"\nThe final configuration uses RMSprop, L2 0.00001, unchanged widths 256/128/64, sigmoid activation, hidden dropout 5%, SKU dropout 50%, and a linear output. The step tables above use seed 42 and cap 40.\n\nConfirmation uses seeds 42/101/202 with matched 40-epoch caps. Previous-configuration mean validation R2 {old_mean['mean_validation_r2']:.8f}; new mean {new_mean['mean_validation_r2']:.8f}; gain {update['mean_r2_gain']:+.8f}, positive in all three paired seeds. Selected checkpoint: seed {selected['seed']}, epoch {selected['selected_epoch']}, validation R2 {selected['validation']['r2']:.8f}, RMSE {selected['validation']['rmse']:.8f}. Test R2 {test['r2']:.8f}, RMSE {test['rmse']:.8f}, MAE {test['mae']:.8f}.\n"
 md+='\nStep 6 bundles training controls and its net gain cannot be attributed solely to checkpoint restoration. The following two displayed changes isolate optimizer and then L2 at the same 40-epoch cap and seed 42. Final mean confirmation scores use equal 40-epoch caps and are distinguished from single-checkpoint scores. Configuration ranking was frozen before test scoring.\n\nAll non-improving trials remain in the full logs: original restoration in `all_runs.csv`, expanded tuning in `../tuning/all_runs.csv`. The selected improved bundle is `../tuning/best_model/`; the earlier model remains at `../model/`. Contributions remain placeholders.\n'
 (P/'BRIEF_SEQUENCE.md').write_text(md)
 print(f'Built {q.count(chr(10)+"## ")+1} slides')
if __name__=='__main__':main()
