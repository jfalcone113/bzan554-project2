#!/usr/bin/env python3
"""Retrain only the frozen winning configuration into a new bundle (no search)."""
import argparse,shutil
from pathlib import Path
import run_tuning as study

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,default=study.HERE/'retrained_best')
 args=parser.parse_args();out=args.output.resolve()
 if not out.is_relative_to(study.ROOT):raise ValueError('Use a fresh output directory inside this project.')
 if out.exists() and any(out.iterdir()):raise ValueError('Output must be new or empty; no fitted bundle is overwritten.')
 selected=study.load(study.HERE/'selection.json')['selected']
 if not (study.CACHE/'preprocessing.json').exists():
  study.CACHE.mkdir(parents=True,exist_ok=True)
  study.base.prepare(study.ROOT/'data/pricing.csv',study.CACHE,study.CACHE)
 source=study.base.sha256(study.ROOT/'data/pricing.csv')
 if source!=selected['signature']['source_sha256']:raise ValueError('The selected configuration requires the original source data.')
 out.mkdir(parents=True,exist_ok=True)
 request=out/'request.json'
 study.dump(request,{'config':selected['signature']['config'],'seed':selected['seed'],'folder':str(out)})
 result=study.worker(request)
 if result['status']!='complete':raise RuntimeError(result['error'])
 for name in ['preprocessing.json','seen.npz']:shutil.copy2(study.CACHE/name,out/name)
 study.dump(out/'config.json',result['signature']['config'])
 import tensorflow as tf
 model=tf.keras.models.load_model(out/'model.keras',compile=False);prep=study.base.Preprocessor(out)
 # Configuration/seed are already frozen; this is final evaluation, not test-driven selection.
 metrics={'train':result['train'],'validation':result['validation'],'test':study.base.evaluate_partition(model,prep,study.CACHE/'test.bin')}
 study.dump(out/'metrics.json',metrics)
 print('Reproduced bundle:',out)
if __name__=='__main__':main()
