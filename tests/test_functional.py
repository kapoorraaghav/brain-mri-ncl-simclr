"""Synthetic unit checks: no private data, GPU, network or full experiment."""
from pathlib import Path
import json,types,unittest
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
def module(extension):
    source=(ROOT/'src/rank_control_runtime.py').read_text()
    if extension:source+='\n'+(ROOT/'src'/extension).read_text()
    e=types.ModuleType('test_runtime');exec(source,e.__dict__);return e
class FunctionalChecks(unittest.TestCase):
    def test_topk(self):
        e=module(None);m=e.SAE(3,8,2);_,acts=m(torch.ones(4,3))
        self.assertTrue(bool(((acts>0).sum(1)<=2).all()))
    def test_retained_rank(self):
        e=module('retained_variance_extension.py')
        self.assertEqual(e.retained_rank(np.array([95.,4.,1.]),.95),1)
        self.assertEqual(e.retained_rank(np.array([80.,10.,10.]),.95),3)
    def test_auxk_gradient(self):
        e=module('auxk_control_extension.py');m=e.SAE(3,4,1)
        with torch.no_grad():m.encoder.weight.zero_();m.encoder.bias.copy_(torch.tensor([4.,3.,2.,1.]));m.decoder.weight.fill_(.1)
        _,_,loss,_,dead,positive=e.auxk_objective(m,torch.ones(2,3),torch.full((4,),8),4,2)
        loss.backward();self.assertEqual(dead,3);self.assertGreater(positive,0)
        self.assertGreater(float(m.encoder.bias.grad[1:3].abs().sum()),0)
    def test_weighted_macro_f1(self):
        e=module('probe_metrics_extension.py');y=np.array([0,0,1,1]);prob=np.array([.1,.8,.4,.9]);pred=(prob>.5).astype(int);w=np.array([2.,1.,2.,1.])
        result=e.binary_metrics(y,prob,pred,w)
        self.assertAlmostEqual(result['macro_f1'],f1_score(y,pred,average='macro',sample_weight=w))
        missing=e.binary_metrics(y,prob,pred,np.array([1.,1.,0.,0.]))
        self.assertTrue(np.isnan(missing['auroc']))
    def test_feature_permutation(self):
        e=module('patient_stability_extension.py');rng=np.random.default_rng(3)
        train=rng.random((30,4)).astype(np.float32);val=rng.random((20,4)).astype(np.float32);perm=np.array([2,0,3,1]);dec=rng.random((6,4)).astype(np.float32)
        plan={**e.DEFAULT_PLAN,'random_alignment_replicates':3}
        table,_=e.feature_match(train,train[:,perm],val,val[:,perm],dec,dec[:,perm],plan,torch.device('cpu'))
        self.assertTrue(np.all(perm[table.right_feature.to_numpy()]==table.left_feature.to_numpy()))
        self.assertTrue(np.all(table.validation_correlation>.999))
if __name__=='__main__':unittest.main()
