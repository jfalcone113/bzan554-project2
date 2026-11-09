#!/usr/bin/env python3
"""Selected quantity model: train, predict, evaluate, or verify a self-contained bundle."""
from __future__ import annotations
import argparse, hashlib, json, math, os, tempfile, time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import pandas as pd
COLUMNS=['sku','price','quantity','order','duration','category']
FEATURES=['sku','price','order','duration','category']
DTYPE=np.dtype('<f8')
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=Path(__file__).resolve().parent
CHUNK=16384
BATCH=1024
SEED=42

def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def tensorflow():
 import tensorflow as tf
 tf.config.threading.set_intra_op_parallelism_threads(4)
 tf.config.threading.set_inter_op_parallelism_threads(1)
 tf.keras.utils.set_random_seed(SEED)
 return tf

def mix64(x):
 x=np.asarray(x,dtype=np.uint64)
 with np.errstate(over='ignore'):
  x=x+np.uint64(0x9E3779B97F4A7C15)
  x=(x^(x>>30))*np.uint64(0xBF58476D1CE4E5B9)
  x=(x^(x>>27))*np.uint64(0x94D049BB133111EB)
 return x^(x>>31)
def split_codes(rows):
 key=mix64(rows[:,0].astype(np.uint64))^mix64(rows[:,3].astype(np.uint64))
 u=(mix64(key^np.uint64(SEED))>>11).astype(np.float64)/2**53
 return np.where(u<.70,0,np.where(u<.85,1,2))
def numeric(rows):
 x=rows[:,[1,3,4]]
 return np.concatenate([x,np.log1p(x*np.array([100.,1.,1.])),np.sqrt(x)],axis=1)
def bucket(values,size):return (values.astype(np.int64)%size+1).astype(np.int32)
def validate(rows):
 if not np.isfinite(rows).all() or (rows<0).any():raise ValueError('All supplied values must be finite and nonnegative.')
 ids=rows[:,[0,3,5]]
 if (ids!=np.floor(ids)).any() or (ids>2**53-1).any():raise ValueError('SKU, order, and category must be exact nonnegative integers.')

class Moments:
 def __init__(self):self.n=0;self.mean=0.;self.m2=0.
 def add(self,x):
  x=np.asarray(x,np.float64);n=len(x)
  if not n:return
  mean=x.mean(axis=0);m2=((x-mean)**2).sum(axis=0);delta=mean-self.mean;total=self.n+n
  self.m2=self.m2+m2+delta**2*self.n*n/total
  self.mean=self.mean+delta*n/total;self.n=total
class Metrics:
 def __init__(self):self.stats=Moments();self.sse=0.;self.sae=0.
 def add(self,y,pred):
  if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions.')
  self.stats.add(y);self.sse+=float(np.square(y-pred).sum());self.sae+=float(np.abs(y-pred).sum())
 def result(self):
  if self.stats.n==0:raise ValueError('Empty partition.')
  mse=self.sse/self.stats.n
  return {'n':self.stats.n,'r2':1-self.sse/float(self.stats.m2),'mse':mse,'rmse':math.sqrt(mse),'mae':self.sae/self.stats.n,'target_mean':float(self.stats.mean)}

def disk_chunks(path,rng=None):
 n=Path(path).stat().st_size//48;blocks=math.ceil(n/CHUNK)
 order=range(blocks) if rng is None else rng.permutation(blocks)
 with Path(path).open('rb') as f:
  for block in order:
   f.seek(int(block)*CHUNK*48)
   rows=np.fromfile(f,dtype=DTYPE,count=min(CHUNK,n-int(block)*CHUNK)*6).reshape(-1,6)
   if rng is not None:rng.shuffle(rows)
   yield rows
class Preprocessor:
 def __init__(self,bundle):
  self.meta=read_json(Path(bundle)/'preprocessing.json')
  self.mean=np.array(self.meta['numeric_mean']);self.std=np.array(self.meta['numeric_std'])
  with np.load(Path(bundle)/'seen.npz') as f:self.sku_seen=f['sku'];self.category_seen=f['category']
 def transform(self,rows):
  sku=bucket(rows[:,0],131072);cat=bucket(rows[:,5],64)
  return {'numeric':((numeric(rows)-self.mean)/self.std).astype('float32'),
          'sku':np.where(self.sku_seen[sku],sku,0).astype('int32'),
          'category':np.where(self.category_seen[cat],cat,0).astype('int32')}

def build_model(tf):
 inputs={'numeric':tf.keras.Input((9,),name='numeric'),
         'sku':tf.keras.Input((),dtype='int32',name='sku'),
         'category':tf.keras.Input((),dtype='int32',name='category')}
 sku=tf.keras.layers.Embedding(131073,8,embeddings_initializer='zeros',name='sku_encoding')(inputs['sku'])
 sku=tf.keras.layers.Dropout(.5,noise_shape=(None,1),name='sku_dropout')(sku)
 cat=tf.keras.layers.Embedding(65,6,embeddings_initializer='zeros',name='category_encoding')(inputs['category'])
 x=tf.keras.layers.Concatenate()([inputs['numeric'],sku,cat])
 for i,w in enumerate([256,128,64],1):
  x=tf.keras.layers.Dense(w,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L2(.0001),name=f'hidden_{i}')(x)
  x=tf.keras.layers.Dropout(.05,name=f'dropout_{i}')(x)
 y=tf.keras.layers.Dense(1,name='quantity_scaled')(x)
 return tf.keras.Model(inputs,y,name='09_checkpoint')

def infer(model,prep,rows):
 x=prep.transform(rows);result=np.empty(len(rows),np.float64)
 for start in range(0,len(rows),BATCH):
  stop=start+BATCH
  y=model({k:v[start:stop] for k,v in x.items()},training=False).numpy().ravel()
  result[start:stop]=y*prep.meta['target_std']+prep.meta['target_mean']
 return result

def evaluate_csv(model,prep,data):
 metrics=[Metrics() for _ in range(3)]
 for frame in pd.read_csv(data,usecols=COLUMNS,dtype=np.float64,chunksize=CHUNK):
  rows=np.ascontiguousarray(frame[COLUMNS].to_numpy(),dtype=DTYPE);validate(rows);codes=split_codes(rows)
  # Predict in partition order, with aggregation identical to the original cache evaluation.
  for i,m in enumerate(metrics):
   subset=rows[codes==i]
   if len(subset):m.add(subset[:,2],infer(model,prep,subset))
 return {name:m.result() for name,m in zip(['train','validation','test'],metrics)}

def prepare(data,folder,bundle):
 stats=Moments();counts={n:0 for n in ['train','validation','test']}
 seen_sku=np.zeros(131073,bool);seen_cat=np.zeros(65,bool)
 files={name:(folder/f'{name}.bin').open('wb') for name in counts}
 try:
  for frame in pd.read_csv(data,usecols=COLUMNS,dtype=np.float64,chunksize=CHUNK):
   rows=np.ascontiguousarray(frame[COLUMNS].to_numpy(),dtype=DTYPE);validate(rows);codes=split_codes(rows)
   for i,name in enumerate(counts):
    subset=np.ascontiguousarray(rows[codes==i]);subset.tofile(files[name]);counts[name]+=len(subset)
    if i==0:
     stats.add(subset[:,2]);seen_sku[bucket(subset[:,0],131072)]=True;seen_cat[bucket(subset[:,5],64)]=True
 finally:
  for f in files.values():f.close()
 if min(counts.values())<2 or stats.m2<=0:raise ValueError('Need nonempty splits and varying targets.')
 xs=Moments()
 for rows in disk_chunks(folder/'train.bin'):xs.add(numeric(rows))
 meta={'source':{'name':Path(data).name,'sha256':sha256(data)},'counts':counts,'split_seed':SEED,
       'numeric_mean':xs.mean.tolist(),'numeric_std':np.maximum(np.sqrt(xs.m2/xs.n),1e-6).tolist(),
       'target_mean':float(stats.mean),'target_std':float(np.sqrt(stats.m2/stats.n))}
 write_json(bundle/'preprocessing.json',meta);np.savez(bundle/'seen.npz',sku=seen_sku,category=seen_cat)
 return Preprocessor(bundle)

def evaluate_partition(model,prep,path):
 m=Metrics()
 for rows in disk_chunks(path):m.add(rows[:,2],infer(model,prep,rows))
 return m.result()

def train(tf,data,output):
 if output.exists() and any(output.iterdir()):raise ValueError('Training output must be a new or empty directory; fitted models are never overwritten.')
 output.mkdir(parents=True,exist_ok=True);tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(SEED)
 with tempfile.TemporaryDirectory(prefix='pricing-training-') as tmp:
  cache=Path(tmp);prep=prepare(data,cache,output);model=build_model(tf)
  optimizer=tf.keras.optimizers.Adam(learning_rate=.001,clipnorm=10.)
  @tf.function(input_signature=[{'numeric':tf.TensorSpec([None,9],tf.float32),'sku':tf.TensorSpec([None],tf.int32),'category':tf.TensorSpec([None],tf.int32)},tf.TensorSpec([None],tf.float32)])
  def update(x,y):
   with tf.GradientTape() as tape:
    pred=tf.squeeze(model(x,training=True),axis=1)
    mse=tf.reduce_mean(tf.square(y-pred));loss=mse+tf.add_n(model.losses)
   gradients=tape.gradient(loss,model.trainable_variables)
   optimizer.apply_gradients(zip(gradients,model.trainable_variables));return mse
  best=np.inf;stale=0;plateau=0;best_epoch=0;updates=0;history=[];start=time.perf_counter()
  for epoch in range(1,21):
   tick=time.perf_counter();online=0.;n=0
   for rows in disk_chunks(cache/'train.bin',np.random.default_rng(SEED+epoch)):
    x=prep.transform(rows);y=((rows[:,2]-prep.meta['target_mean'])/prep.meta['target_std']).astype('float32')
    for offset in range(0,len(rows),BATCH):
     stop=min(offset+BATCH,len(rows));v=float(update({k:a[offset:stop] for k,a in x.items()},y[offset:stop]).numpy())
     if not math.isfinite(v):raise ValueError('Nonfinite training loss.')
     online+=v*(stop-offset);n+=stop-offset;updates+=1
   val=evaluate_partition(model,prep,cache/'validation.bin');improved=val['mse']<best-1e-6
   if improved:
    best=val['mse'];best_epoch=epoch;stale=0;plateau=0;model.save(output/'model.keras')
   else:stale+=1;plateau+=1
   history.append({'epoch':epoch,'validation':val,'online_preupdate_mse':online/n*prep.meta['target_std']**2,'learning_rate':float(optimizer.learning_rate.numpy()),'seconds':time.perf_counter()-tick,'updates':updates})
   write_json(output/'epochs.json',history)
   print(f'Epoch {epoch}: validation R2={val["r2"]:.8f} RMSE={val["rmse"]:.8f}',flush=True)
   if plateau>=4:optimizer.learning_rate.assign(max(float(optimizer.learning_rate.numpy())*.5,.00003));plateau=0
   if stale>=12:break
  model=tf.keras.models.load_model(output/'model.keras',compile=False)
  metrics={name:evaluate_partition(model,prep,cache/f'{name}.bin') for name in ['train','validation','test']}
  write_json(output/'metrics.json',metrics)
  write_json(output/'training.json',{'seed':42,'split_seed':42,'selected_epoch':best_epoch,'epochs_run':epoch,'updates':updates,'parameters':model.count_params(),'seconds':time.perf_counter()-start,'tensorflow':tf.__version__,'keras':tf.keras.__version__,'optimizer_state_saved':False})
 return metrics

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 sub=parser.add_subparsers(dest='command',required=True)
 t=sub.add_parser('train');t.add_argument('--data',type=Path,default=ROOT/'data/pricing.csv');t.add_argument('--output',type=Path,default=BUNDLE/'retrained')
 for name in ['evaluate','verify','predict']:
  p=sub.add_parser(name);p.add_argument('--bundle',type=Path,default=BUNDLE)
  if name=='predict':p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
  else:p.add_argument('--data',type=Path,default=ROOT/'data/pricing.csv')
 args=parser.parse_args();tf=tensorflow()
 if args.command=='train':print(json.dumps(train(tf,args.data,args.output),indent=2));return
 prep=Preprocessor(args.bundle);model=tf.keras.models.load_model(args.bundle/'model.keras',compile=False)
 if args.command=='predict':
  if args.input.resolve()==args.output.resolve():raise ValueError('Input and output CSV must differ.')
  first=True
  for frame in pd.read_csv(args.input,dtype={k:np.float64 for k in FEATURES},chunksize=CHUNK):
   rows=np.zeros((len(frame),6),dtype=DTYPE)
   for feature in FEATURES:rows[:,COLUMNS.index(feature)]=frame[feature].to_numpy()
   validate(rows);frame['predicted_quantity']=infer(model,prep,rows)
   frame.to_csv(args.output,index=False,mode='w' if first else 'a',header=first);first=False
  print(f'Predictions saved to {args.output}');return
 actual=evaluate_csv(model,prep,args.data)
 if args.command=='verify':
  expected=read_json(args.bundle/'metrics.json')
  source=prep.meta.get('source',prep.meta.get('signature',{}).get('source',{}))
  if source.get('sha256')!=sha256(args.data):raise ValueError('Verification requires the original dataset hash.')
  for split in actual:
   for key in ['r2','mse','rmse','mae']:
    if not np.isclose(actual[split][key],expected[split][key],rtol=1e-6,atol=1e-6):raise AssertionError((split,key,actual[split][key],expected[split][key]))
  print('PASS: model metrics reproduce the saved reference within tolerance.')
 print(json.dumps(actual,indent=2))
if __name__=='__main__':main()
