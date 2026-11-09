#!/usr/bin/env python3
"""Reproduce displayed trials with a 40-epoch cap; preserve original selection rules."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-tuning-mpl')
import sys,json,time,math,subprocess,csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import run_tuning as study
base=study.base
HERE=study.HERE/'presentation_40';CACHE=HERE/'data'
class Prep(base.Preprocessor):
 def __init__(self,path,n):super().__init__(path);self.n=n
 def transform(self,rows):
  x=super().transform(rows);x['numeric']=x['numeric'][:,:self.n];return x

def model(tf,c):
 n=3*(c['transforms']+1)
 inputs={'numeric':tf.keras.Input((n,),name='numeric'),'sku':tf.keras.Input((),dtype='int32',name='sku'),'category':tf.keras.Input((),dtype='int32',name='category')}
 sku=tf.keras.layers.Embedding(131073,8,embeddings_initializer='zeros',name='sku_encoding')(inputs['sku'])
 if c['sku_dropout']:sku=tf.keras.layers.Dropout(c['sku_dropout'],noise_shape=(None,1),name='sku_dropout')(sku)
 cat=tf.keras.layers.Embedding(65,6,embeddings_initializer='zeros',name='category_encoding')(inputs['category'])
 x=tf.keras.layers.Concatenate()([inputs['numeric'],sku,cat])
 for i,w in enumerate(c['widths'],1):
  x=tf.keras.layers.Dense(w,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L2(c['l2']) if c['l2'] else None,name=f'hidden_{i}')(x)
  if c['dropout']:x=tf.keras.layers.Dropout(c['dropout'],name=f'dropout_{i}')(x)
 return tf.keras.Model(inputs,tf.keras.layers.Dense(1,name='quantity_scaled')(x),name=c['name'])

def restore(name):
 original=next(r for r in study.load(study.ROOT/'presentation/results.json') if r['name']==name)
 c=original['signature']['config'];folder=HERE/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return study.load(folder/'result.json')
 tf=base.tensorflow();tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
 prep=Prep(CACHE,3*(c['transforms']+1));m=model(tf,c)
 opt=tf.keras.optimizers.Adam(learning_rate=c['lr'],**({'clipnorm':c['clip']} if c['clip'] else {}))
 @tf.function
 def update(x,y):
  with tf.GradientTape() as tape:
   pred=tf.squeeze(m(x,training=True),axis=1);mse=tf.reduce_mean(tf.square(y-pred));loss=mse+(tf.add_n(m.losses) if m.losses else 0.)
  opt.apply_gradients(zip(tape.gradient(loss,m.trainable_variables),m.trainable_variables));return mse
 history=[];best=float('inf');bestepoch=0;stale=plateau=updates=0;start=time.perf_counter()
 for epoch in range(1,41):
  tick=time.perf_counter();online=0.;n=0
  for rows in base.disk_chunks(CACHE/'train.bin',np.random.default_rng(42+epoch)):
   x=prep.transform(rows);y=((rows[:,2]-prep.meta['target_mean'])/prep.meta['target_std']).astype('float32')
   for off in range(0,len(rows),1024):
    stop=min(off+1024,len(rows));v=float(update({k:a[off:stop] for k,a in x.items()},y[off:stop]).numpy());assert math.isfinite(v)
    online+=v*(stop-off);n+=stop-off;updates+=1
  val=base.evaluate_partition(m,prep,CACHE/'validation.bin')
  if val['mse']<best-1e-6:
   best=val['mse'];bestepoch=epoch;stale=plateau=0
   if c['checkpoint']:m.save(folder/'model.keras')
  else:stale+=1;plateau+=1
  history.append({'epoch':epoch,'validation':val,'online_preupdate_mse':online/n*prep.meta['target_std']**2,'learning_rate':float(opt.learning_rate.numpy()),'seconds':time.perf_counter()-tick,'updates':updates})
  study.dump(folder/'epochs.json',history)
  print(f'{name} epoch={epoch} R2={val["r2"]:.8f}',flush=True)
  if epoch==20:
   reference=original['validation']
   if abs(val['r2']-reference['r2'])>1e-5:raise RuntimeError(f'20-epoch reproduction mismatch: {val} vs {reference}')
  if c['reduce_lr'] and plateau>=4:opt.learning_rate.assign(max(float(opt.learning_rate.numpy())*.5,.00003));plateau=0
  if c['early_stop'] and stale>=12:break
 if c['checkpoint']:m=tf.keras.models.load_model(folder/'model.keras',compile=False)
 else:m.save(folder/'model.keras')
 result={'name':name,'seed':42,'status':'complete','signature':{'config':c,'seed':42,'max_epochs':40,'split_seed':42,'source_sha256':base.sha256(study.ROOT/'data/pricing.csv'),'script_sha256':base.sha256(Path(__file__))},'train':base.evaluate_partition(m,prep,CACHE/'train.bin'),'validation':base.evaluate_partition(m,prep,CACHE/'validation.bin'),'selected_epoch':bestepoch if c['checkpoint'] else epoch,'best_observed_epoch':bestepoch,'epochs_run':epoch,'updates':updates,'parameters':m.count_params(),'fit_seconds':time.perf_counter()-start,'checkpoint':str((folder/'model.keras').relative_to(study.ROOT)),'reproduction_20_epoch_passed':True}
 study.dump(folder/'result.json',result);return result

def one(task):
 kind,name=task;folder=HERE/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
 with (folder/'console.log').open('a') as out:
  p=subprocess.run([sys.executable,str(Path(__file__).resolve()),kind,name],stdout=out,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{name} failed: {folder}/console.log')
 print('COMPLETE',name,flush=True)

def expanded(name):
 r=study.load(study.HERE/'runs'/f'{name}_seed42/result.json');c=r['signature']['config'];c['max_epochs']=40
 req=HERE/'runs'/name/'request.json';study.dump(req,{'config':c,'seed':42,'folder':str(req.parent)})
 study.CACHE=CACHE
 result=study.worker(req)
 if result['status']!='complete':raise RuntimeError(result)
 original=study.load(study.HERE/'runs'/f'{name}_seed42/epochs.json')
 history=study.load(req.parent/'epochs.json')
 assert abs(history[19]['validation']['r2']-original[19]['validation']['r2'])<1e-5

if __name__=='__main__':
 if len(sys.argv)>1:
  (restore if sys.argv[1]=='restore' else expanded)(sys.argv[2])
 else:
  HERE.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
  if not (CACHE/'preprocessing.json').exists():base.prepare(study.ROOT/'data/pricing.csv',CACHE,CACHE)
  tasks=[('restore',n) for n in ['00_basic','01_log','02_sqrt','03_hidden_dropout','04_sku_dropout','05_l2']]+[('expanded',n) for n in ['c_seven','d_l2_high','d_batch4096','c_bn_before']]
  study.dump(HERE/'protocol.json',{'scope':tasks,'seed':42,'epoch_cap':40,'policy':'Only epoch cap changes. Preserve final-epoch selection for early restoration steps; best validation checkpoint for expanded trials. Reproduce original epoch-20 metrics before accepting results. No test scoring. Reuse existing seed42 40-cap confirmation for steps6,7,8.','monitored_metric':'validation_mse','concurrency':2,'threads_per_fit':4})
  with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,tasks))
