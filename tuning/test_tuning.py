"""Check baseline equivalence, supported architectures, BN inference, and schedule units."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-tuning-mpl')
import unittest,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import run_tuning as t

class TuningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tf=t.base.tensorflow()
 def test_baseline_builder_is_identical(self):
  tf=self.tf
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);a=t.base.build_model(tf)
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);b=t.build_model(tf,t.Config())
  self.assertEqual(a.count_params(),b.count_params())
  for x,y in zip(a.get_weights(),b.get_weights()):np.testing.assert_array_equal(x,y)
 def test_all_activation_variants_roundtrip(self):
  tf=self.tf;x={'numeric':np.zeros((3,9),np.float32),'sku':np.array([0,1,2],np.int32),'category':np.array([0,1,2],np.int32)}
  for c in t.stage_a():
   model=t.build_model(tf,c);before=model(x,training=False).numpy()
   self.assertTrue(np.isfinite(before).all())
   with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
    p=Path(tmp)/'model.keras';model.save(p);loaded=tf.keras.models.load_model(p,compile=False)
    np.testing.assert_array_equal(loaded(x,training=False).numpy(),before)
 def test_bn_inference_does_not_update_statistics(self):
  tf=self.tf
  for place in ['before','after']:
   model=t.build_model(tf,replace(t.Config(),bn=place,activation='elu',initializer='he_uniform'))
   x={'numeric':np.ones((8,9),np.float32),'sku':np.zeros(8,np.int32),'category':np.zeros(8,np.int32)}
   model(x,training=True)
   bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
   before=[l.get_weights() for l in bn];model(x,training=False)
   for l,weights in zip(bn,before):
    for a,b in zip(l.get_weights(),weights):np.testing.assert_array_equal(a,b)
 def test_exponential_schedule_counts_updates(self):
  tf=self.tf;n=349975
  for batch in [256,1024,4096]:
   c=replace(t.Config(),schedule='exponential',batch=batch)
   opt=t.optimizer_for(tf,c,n);steps=int(np.ceil(n/batch))
   self.assertAlmostEqual(float(opt._learning_rate(steps).numpy()),c.lr*.95,places=9)
 def test_fixed_budget_and_sku_dropout(self):
  a=t.stage_a();b=t.stage_b(a[0]);c=t.stage_c(a[0]);d=t.stage_d(a[0])
  self.assertEqual(len(a)+len(b)+len(c)+len(d)+9,42)
  self.assertTrue(all(x.sku_dropout==.5 for x in a+b+c+d))
if __name__=='__main__':unittest.main(verbosity=2)
