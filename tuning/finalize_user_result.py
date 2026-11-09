"""Apply the user's explicit measurable-gain criterion without changing scores or ranking."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 p=P/'selection.json';s=json.loads(p.read_text());criterion=json.loads((P/'USER_CRITERION.json').read_text())
 s.setdefault('original_proposed_threshold_met',s['noticeable'])
 wins=sum(x['difference']>0 for x in s['paired_differences'])
 accepted=s['mean_r2_gain']>criterion['minimum_mean_gain_exclusive'] and wins>=criterion['minimum_paired_wins']
 s['user_acceptance']={'criterion':criterion['criterion'],'supersedes_original_threshold':True,'paired_seed_wins':wins,'accepted':accepted,'does_not_change_configuration_or_checkpoint_selection':True}
 s['noticeable']=accepted
 p.write_text(json.dumps(s,indent=2)+'\n')
 target=P/'best_model/selection.json'
 if target.exists():target.write_text(json.dumps(s,indent=2)+'\n')
 print('User measurable-improvement criterion met:',accepted)
if __name__=='__main__':main()
