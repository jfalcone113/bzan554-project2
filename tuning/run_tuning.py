#!/usr/bin/env python3
"""Bounded, validation-selected course tuning. Run with the existing Python environment."""
from __future__ import annotations
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-tuning-mpl')
import argparse,csv,gc,hashlib,inspect,json,math,resource,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model'))
import pricing_model as base
CACHE=HERE/'data'

def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(p):return json.loads(Path(p).read_text())
def script_hash():return base.sha256(Path(__file__))

@dataclass
class Config:
 name:str='a_baseline'
 stage:str='A'
 parent:str='packaged_baseline'
 change:str='Reproduce packaged sigmoid baseline'
 widths:tuple=(256,128,64)
 activation:str='sigmoid'
 initializer:str='glorot_uniform'
 bn:str='none'
 hidden_dropout:float=.05
 sku_dropout:float=.5
 l1:float=0.
 l2:float=.0001
 optimizer:str='adam'
 lr:float=.001
 schedule:str='plateau'
 batch:int=1024
 max_epochs:int=20
 patience:int=12
 clip:float=10.

def hyperkey(c):
 d=asdict(c) if isinstance(c,Config) else dict(c)
 for k in ['name','stage','parent','change','max_epochs']:d.pop(k,None)
 return json.dumps(d,sort_keys=True)

def build_model(tf,c):
 inputs={'numeric':tf.keras.Input((9,),name='numeric'),'sku':tf.keras.Input((),dtype='int32',name='sku'),'category':tf.keras.Input((),dtype='int32',name='category')}
 sku=tf.keras.layers.Embedding(131073,8,embeddings_initializer='zeros',name='sku_encoding')(inputs['sku'])
 sku=tf.keras.layers.Dropout(c.sku_dropout,noise_shape=(None,1),name='sku_dropout')(sku)
 cat=tf.keras.layers.Embedding(65,6,embeddings_initializer='zeros',name='category_encoding')(inputs['category'])
 x=tf.keras.layers.Concatenate()([inputs['numeric'],sku,cat])
 for i,w in enumerate(c.widths,1):
  initializer=tf.keras.initializers.VarianceScaling(scale=16.,mode='fan_avg',distribution='uniform') if c.initializer=='sigmoid_scaled' else c.initializer
  penalty=tf.keras.regularizers.L1L2(c.l1,c.l2) if c.l1 else tf.keras.regularizers.L2(c.l2)
  direct=c.bn=='none' and c.activation in ['sigmoid','tanh','relu','swish','softplus']
  x=tf.keras.layers.Dense(w,activation=c.activation if direct else None,kernel_initializer=initializer,use_bias=c.bn!='before',kernel_regularizer=penalty,name=f'hidden_{i}')(x)
  if c.bn=='before':x=tf.keras.layers.BatchNormalization(momentum=.99,epsilon=.001,name=f'bn_{i}')(x)
  if not direct:
   if c.activation=='leaky_relu':x=tf.keras.layers.LeakyReLU(negative_slope=.01,name=f'activation_{i}')(x)
   elif c.activation=='prelu':x=tf.keras.layers.PReLU(alpha_initializer='zeros',name=f'activation_{i}')(x)
   elif c.activation=='elu':x=tf.keras.layers.ELU(alpha=1.,name=f'activation_{i}')(x)
   else:x=tf.keras.layers.Activation(c.activation,name=f'activation_{i}')(x)
  if c.bn=='after':x=tf.keras.layers.BatchNormalization(momentum=.99,epsilon=.001,name=f'bn_{i}')(x)
  if c.hidden_dropout:x=tf.keras.layers.Dropout(c.hidden_dropout,name=f'dropout_{i}')(x)
 output=tf.keras.layers.Dense(1,name='quantity_scaled')(x)
 return tf.keras.Model(inputs,output,name='09_checkpoint' if c.name=='a_baseline' else c.name)

def optimizer_for(tf,c,n):
 lr=c.lr
 if c.schedule=='exponential':lr=tf.keras.optimizers.schedules.ExponentialDecay(c.lr,decay_steps=math.ceil(n/c.batch),decay_rate=.95,staircase=False)
 kwargs={'learning_rate':lr,'clipnorm':c.clip}
 if c.optimizer=='adam':return tf.keras.optimizers.Adam(**kwargs)
 if c.optimizer=='rmsprop':return tf.keras.optimizers.RMSprop(rho=.9,momentum=0.,epsilon=1e-7,**kwargs)
 if c.optimizer=='adagrad':return tf.keras.optimizers.Adagrad(initial_accumulator_value=.1,epsilon=1e-7,**kwargs)
 return tf.keras.optimizers.SGD(momentum=.9 if c.optimizer in ['momentum','nesterov'] else 0.,nesterov=c.optimizer=='nesterov',**kwargs)

def worker(request):
 req=load(request);c=Config(**req['config']);seed=req['seed'];folder=Path(req['folder']);folder.mkdir(parents=True,exist_ok=True)
 signature={'config':asdict(c),'seed':seed,'source_sha256':base.sha256(ROOT/'data/pricing.csv'),'split_seed':42,'script_sha256':script_hash(),'preprocessing_sha256':base.sha256(CACHE/'preprocessing.json')}
 signature=json.loads(json.dumps(signature))
 resultfile=folder/'result.json'
 if resultfile.exists():
  old=load(resultfile)
  if old['signature']!=signature:raise ValueError('Completed run has a different signature.')
  return old
 tf=base.tensorflow();tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
 prep=base.Preprocessor(CACHE);model=build_model(tf,c);opt=optimizer_for(tf,c,prep.meta['counts']['train'])
 @tf.function(input_signature=[{'numeric':tf.TensorSpec([None,9],tf.float32),'sku':tf.TensorSpec([None],tf.int32),'category':tf.TensorSpec([None],tf.int32)},tf.TensorSpec([None],tf.float32)])
 def update(x,y):
  with tf.GradientTape() as tape:
   pred=tf.squeeze(model(x,training=True),axis=1);mse=tf.reduce_mean(tf.square(y-pred));loss=mse+tf.add_n(model.losses)
  gradients=tape.gradient(loss,model.trainable_variables);opt.apply_gradients(zip(gradients,model.trainable_variables));return mse
 start=time.perf_counter();history=[];best=np.inf;bestepoch=0;stale=0;plateau=0;updates=0;process=psutil.Process()
 try:
  for epoch in range(1,c.max_epochs+1):
   tick=time.perf_counter();n=0;online=0.;lr_start=float(opt.learning_rate.numpy())
   for rows in base.disk_chunks(CACHE/'train.bin',np.random.default_rng(seed+epoch)):
    x=prep.transform(rows);y=((rows[:,2]-prep.meta['target_mean'])/prep.meta['target_std']).astype('float32')
    for offset in range(0,len(rows),c.batch):
     stop=min(offset+c.batch,len(rows));mse=float(update({k:a[offset:stop] for k,a in x.items()},y[offset:stop]).numpy())
     if not math.isfinite(mse):raise ValueError('Nonfinite training MSE')
     n+=stop-offset;online+=mse*(stop-offset);updates+=1
   val=base.evaluate_partition(model,prep,CACHE/'validation.bin');improved=val['mse']<best-1e-6
   if improved:best=val['mse'];bestepoch=epoch;stale=0;plateau=0;model.save(folder/'model.keras')
   else:stale+=1;plateau+=1
   history.append({'epoch':epoch,'validation':val,'online_preupdate_mse':online/n*prep.meta['target_std']**2,'learning_rate_start':lr_start,'learning_rate_end':float(opt.learning_rate.numpy()),'updates':updates,'seconds':time.perf_counter()-tick,'process_rss_mib':process.memory_info().rss/2**20})
   dump(folder/'epochs.json',history)
   print(f'{c.name} seed={seed} epoch={epoch} val_R2={val["r2"]:.6f} RMSE={val["rmse"]:.4f} lr={lr_start:.5g}',flush=True)
   if c.schedule=='plateau' and plateau>=4:opt.learning_rate.assign(max(float(opt.learning_rate.numpy())*.5,.00003));plateau=0
   if stale>=c.patience:break
  model=tf.keras.models.load_model(folder/'model.keras',compile=False)
  result={'name':c.name,'seed':seed,'status':'complete','signature':signature,'selected_epoch':bestepoch,'epochs_run':epoch,'updates':updates,'fit_seconds':time.perf_counter()-start,'parameters':model.count_params(),'trainable_parameters':sum(int(np.prod(x.shape)) for x in model.trainable_weights),'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'train':base.evaluate_partition(model,prep,CACHE/'train.bin'),'validation':base.evaluate_partition(model,prep,CACHE/'validation.bin'),'checkpoint':str((folder/'model.keras').relative_to(ROOT)),'optimizer_state_saved':False}
  if c.name=='a_baseline' and seed==42:
   ref=load(ROOT/'model/metrics.json')['validation']
   if abs(result['validation']['r2']-ref['r2'])>1e-6:raise AssertionError('Baseline does not reproduce packaged reference.')
 except Exception as exc:
  result={'name':c.name,'seed':seed,'status':'failed','signature':signature,'error':str(exc),'traceback':traceback.format_exc(),'fit_seconds':time.perf_counter()-start,'epochs_completed':len(history)}
 dump(resultfile,result);return result

def make_requests(configs,seeds=(42,)):
 requests=[]
 for c in configs:
  for seed in seeds:
   folder=HERE/'runs'/f'{c.name}_seed{seed}'
   request=HERE/'requests'/f'{c.name}_seed{seed}.json'
   payload={'config':asdict(c),'seed':seed,'folder':str(folder)}
   if request.exists() and load(request)!=json.loads(json.dumps(payload)):raise ValueError('Request settings changed.')
   dump(request,payload);requests.append(request)
 return requests

def run_one(request):
 req=load(request);folder=Path(req['folder']);folder.mkdir(parents=True,exist_ok=True)
 with (folder/'console.log').open('a') as out:
  done=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(request)],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
 result=folder/'result.json'
 if done.returncode!=0 or not result.exists():
  raise RuntimeError(f'Worker failed unexpectedly; see {folder}/console.log')
 r=load(result);print(f'FINISHED {r["name"]} seed={r["seed"]}: '+(f'R2={r["validation"]["r2"]:.6f}' if r['status']=='complete' else r['error']),flush=True)
 return r

def run_stage(letter,configs,seeds=(42,)):
 dump(HERE/f'plan_{letter}.json',{'configs':[asdict(c) for c in configs],'seeds':list(seeds),'frozen_before_fits':True})
 results=[]
 with ThreadPoolExecutor(max_workers=2) as executor:
  futures=[executor.submit(run_one,r) for r in make_requests(configs,seeds)]
  for future in as_completed(futures):
   results.append(future.result());dump(HERE/f'results_{letter}.json',sorted(results,key=lambda r:(r['name'],r['seed'])))
 return results

def successful(results):return [r for r in results if r['status']=='complete']
def config_from_result(r):return Config(**r['signature']['config'])
def winner(results):return config_from_result(min(successful(results),key=lambda r:r['validation']['mse']))
def derive(parent,name,stage,change,**kw):return replace(parent,name=name,stage=stage,parent=parent.name,change=change,**kw)

def stage_a():
 b=Config()
 return [b,derive(b,'a_sigmoid_scaled','A','Sigmoid-scaled initialization',initializer='sigmoid_scaled'),derive(b,'a_tanh','A','Tanh activation',activation='tanh'),derive(b,'a_relu_glorot','A','ReLU activation with matched Glorot initialization',activation='relu'),derive(b,'a_relu_he','A','ReLU and He fan-in initialization',activation='relu',initializer='he_uniform'),derive(b,'a_leaky_he','A','Leaky ReLU slope .01 and He initialization',activation='leaky_relu',initializer='he_uniform'),derive(b,'a_prelu_he','A','Learned PReLU and He initialization',activation='prelu',initializer='he_uniform'),derive(b,'a_elu_he','A','ELU alpha 1 and He initialization',activation='elu',initializer='he_uniform'),derive(b,'a_swish_he','A','Swish and He initialization',activation='swish',initializer='he_uniform')]
def stage_b(p):
 return [derive(p,'b_'+name,'B',f'{opt} learning rate {lr}',optimizer=opt,lr=lr) for name,opt,lr in [('sgd01','sgd',.01),('sgd1','sgd',.1),('momentum01','momentum',.01),('nesterov01','nesterov',.01),('nesterov1','nesterov',.1),('adagrad01','adagrad',.01),('rmsprop001','rmsprop',.001),('rmsprop0003','rmsprop',.0003)]]
def stage_c(p):
 cases=[('one',[128]),('two',[128,64]),('flat_three',[128,128,128]),('five',[256,128,128,64,64]),('wide',[512,256,128]),('seven',[256,256,128,128,64,64,32])]
 configs=[derive(p,'c_'+name,'C',f'Hidden widths {widths}',widths=tuple(widths)) for name,widths in cases]
 configs += [derive(p,'c_bn_'+place,'C',f'Batch normalization {place} hidden activation',bn=place) for place in ['before','after']]
 return configs

def stage_d(p):
 return [derive(p,'d_batch256','D','Batch size 256',batch=256),derive(p,'d_batch4096','D','Batch size 4096',batch=4096),derive(p,'d_l2_low','D','L2 coefficient 0.00001',l2=.00001),derive(p,'d_l2_high','D','L2 coefficient 0.001',l2=.001),derive(p,'d_dropout0','D','Hidden dropout 0',hidden_dropout=0.),derive(p,'d_dropout10','D','Hidden dropout .10',hidden_dropout=.10),derive(p,'d_lr_low','D','Initial rate 0.3 times parent',lr=p.lr*.3),derive(p,'d_exponential','D','Exponential LR decay .95 per epoch instead of plateau reductions',schedule='exponential')]

def prepare_and_run():
 HERE.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
 source=base.sha256(ROOT/'data/pricing.csv')
 metadata=CACHE/'preprocessing.json'
 if metadata.exists():
  if load(metadata)['source']['sha256']!=source:raise ValueError('Cached source changed.')
 else:base.prepare(ROOT/'data/pricing.csv',CACHE,CACHE)
 original=load(ROOT/'model/preprocessing.json');current=load(metadata)
 for key in ['numeric_mean','numeric_std','target_mean','target_std']:np.testing.assert_allclose(current[key],original[key],rtol=1e-14,atol=1e-14)
 dump(HERE/'protocol.json',{'source_sha256':source,'deep_md_sha256':base.sha256(ROOT/'resources/deep.md'),'script_sha256':script_hash(),'split_seed':42,'counts':current['counts'],'screen_seed':42,'screen_max_epochs':20,'confirmation_seeds':[42,101,202],'confirmation_max_epochs':40,'max_fits':42,'max_concurrent_fits':2,'cpu_threads_per_fit':4,'noticeable_mean_r2_gain':.01,'noticeable_min_paired_seed_wins':2,'selection':'mean validation MSE in confirmation; then best validation seed of that configuration','test_policy':'No candidate test scoring before frozen selection; historical exposure acknowledged.'})
 allresults=[]
 for letter,builder in [('A',lambda:stage_a()),('B',lambda:stage_b(winner(allresults))),('C',lambda:stage_c(winner(allresults))),('D',lambda:stage_d(winner(allresults)))]:
  configs=builder();rs=run_stage(letter,configs);allresults+=rs;dump(HERE/'screen_results.json',allresults)
 ranked=sorted(successful(allresults),key=lambda r:r['validation']['mse']);finalists=[];seen={hyperkey(Config())}
 for r in ranked:
  c=config_from_result(r);key=hyperkey(c)
  if key not in seen:seen.add(key);finalists.append(c)
  if len(finalists)==2:break
 configs=[derive(Config(),'e_baseline','E','Baseline under matched 40-epoch confirmation budget',max_epochs=40)]
 configs += [derive(c,f'e_finalist{i+1}','E',f'Confirm screening configuration {c.name}',max_epochs=40) for i,c in enumerate(finalists)]
 confirmation=run_stage('E',configs,seeds=(42,101,202))
 dump(HERE/'confirmation_results.json',confirmation)
 finalize(allresults,confirmation)

def finalize(screen,confirmation):
 import shutil
 good=successful(confirmation);groups=[]
 for name in sorted(set(r['name'] for r in good)):
  group=[r for r in good if r['name']==name]
  if len(group)!=3:continue
  groups.append({'name':name,'mean_validation_mse':float(np.mean([r['validation']['mse'] for r in group])),'mean_validation_r2':float(np.mean([r['validation']['r2'] for r in group])),'mean_validation_rmse':float(np.mean([r['validation']['rmse'] for r in group])),'min_validation_r2':min(r['validation']['r2'] for r in group),'max_validation_r2':max(r['validation']['r2'] for r in group)})
 if not groups:raise ValueError('No successful three-seed confirmation group.')
 selected_group=min(groups,key=lambda x:x['mean_validation_mse'])
 selected=min([r for r in good if r['name']==selected_group['name']],key=lambda r:r['validation']['mse'])
 baseline=next(g for g in groups if g['name']=='e_baseline')
 paired=[]
 for seed in [42,101,202]:
  a=next(r for r in good if r['name']=='e_baseline' and r['seed']==seed);b=next(r for r in good if r['name']==selected['name'] and r['seed']==seed)
  paired.append({'seed':seed,'baseline_r2':a['validation']['r2'],'selected_r2':b['validation']['r2'],'difference':b['validation']['r2']-a['validation']['r2']})
 gain=selected_group['mean_validation_r2']-baseline['mean_validation_r2']
 selection={'groups':groups,'selected_group':selected_group,'selected':selected,'paired_differences':paired,'mean_r2_gain':gain,'noticeable':gain>=.01 and sum(p['difference']>0 for p in paired)>=2,'frozen_before_test':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 dump(HERE/'selection.json',selection)
 bundle=HERE/'best_model';bundle.mkdir(exist_ok=True)
 for name in ['preprocessing.json','seen.npz']:shutil.copy2(CACHE/name,bundle/name)
 shutil.copy2(ROOT/selected['checkpoint'],bundle/'model.keras')
 dump(bundle/'config.json',selected['signature']['config']);shutil.copy2(HERE/'selection.json',bundle/'selection.json')
 # This is the first new model test scoring in the study.
 tf=base.tensorflow();prep=base.Preprocessor(bundle);model=tf.keras.models.load_model(bundle/'model.keras',compile=False)
 bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
 before=[(l.moving_mean.numpy().copy(),l.moving_variance.numpy().copy()) for l in bn]
 test=base.evaluate_partition(model,prep,CACHE/'test.bin');val=base.evaluate_partition(model,prep,CACHE/'validation.bin')
 if abs(val['mse']-selected['validation']['mse'])>1e-5:raise AssertionError('Reloaded validation mismatch.')
 for l,(mean,var) in zip(bn,before):
  np.testing.assert_array_equal(l.moving_mean.numpy(),mean);np.testing.assert_array_equal(l.moving_variance.numpy(),var)
 metrics={'train':selected['train'],'validation':val,'test':test};dump(bundle/'metrics.json',metrics)
 dump(HERE/'final_evaluation.json',{'selection_frozen_before_test':True,'metrics':metrics,'old_model_metrics':load(ROOT/'model/metrics.json'),'bn_statistics_unchanged_in_inference':True,'validation_reload_matches':True})
 print('TUNING COMPLETE: '+json.dumps({'noticeable':selection['noticeable'],'mean_validation_gain':gain,'validation':val,'test':test}),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path);args=parser.parse_args()
 if args.worker:worker(args.worker)
 else:prepare_and_run()
if __name__=='__main__':main()
