"""Deterministic checks for optimization, metrics, and update baselines."""
import unittest
import numpy as np
from harm_budgeted_updates.allocation import (
    all_updates, allocate_binary, common_strength_binary, uniform_cap_binary,
    dinfinity_binary, floor_bounds, binary_cross_entropy,
    robust_common_binary, logit_matched_binary, cap_path_binary,
)
from harm_budgeted_updates.bayesian import confusion,beliefs
from harm_budgeted_updates.metrics import BinaryMetricContext


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(13)
        self.b=rng.uniform(.01,.99,41)
        self.t=rng.uniform(.01,.99,41)

    def test_budget_and_target_optimum(self):
        for h in [0,.0025,.01,.1]:
            p,_=allocate_binary(self.b,self.t,h)
            q,_=common_strength_binary(self.b,self.t,h)
            u,_=uniform_cap_binary(self.b,self.t,h)
            for x in [p,q,u]:
                c=dinfinity_binary(self.b,x)
                self.assertLessEqual(c.mean(),h+2e-14)
                self.assertLessEqual(c.max(),.1+2e-14)
            self.assertLessEqual(binary_cross_entropy(self.t,p),
                                 binary_cross_entropy(self.t,q)+1e-12)
            self.assertLessEqual(binary_cross_entropy(self.t,p),
                                 binary_cross_entropy(self.t,u)+1e-12)

    def test_uniform_cap_tails(self):
        p,_=allocate_binary(self.b,self.t,.01)
        u,_=uniform_cap_binary(self.b,self.t,.01)
        uc=np.sort(dinfinity_binary(self.b,u))[::-1].cumsum()
        pc=np.sort(dinfinity_binary(self.b,p))[::-1].cumsum()
        self.assertTrue(np.all(uc<=pc+2e-12))

    def test_full_intervals_retain_comparator(self):
        q,_=uniform_cap_binary(self.b,self.t,.0025)
        p,_=robust_common_binary(self.b,np.zeros(41),np.ones(41),.0025,comparator=q)
        np.testing.assert_array_equal(p,q)

    def test_logit_cost_and_cap_path(self):
        q,_=common_strength_binary(self.b,self.t,.0025)
        p,_=logit_matched_binary(self.b,self.t,.0025)
        self.assertAlmostEqual(dinfinity_binary(self.b,p).mean(),
                               dinfinity_binary(self.b,q).mean(),places=12)
        p,_=cap_path_binary(self.b,self.t,.0025,level=.1)
        original,_=allocate_binary(self.b,self.t,.0025)
        np.testing.assert_array_equal(p,original)

    def test_metrics_against_independent_reference(self):
        from sklearn.metrics import average_precision_score,roc_auc_score
        y=np.array([0,1,0,1,0,1,0,1])
        b=np.array([.1,.3,.5,.4,.2,.8,.9,.6])
        p=np.array([.3,.3,.5,.5,.5,.8,.6,.8])
        r=BinaryMetricContext(b,y)(p)
        self.assertAlmostEqual(r['AP'],average_precision_score(y,p),places=14)
        self.assertAlmostEqual(r['AUROC'],roc_auc_score(y,p),places=14)
        self.assertEqual(r['recall_at_capacity'],.25)

    def test_bayesian_probabilities(self):
        y=np.array([0,0,1,1]);old=np.array([.1,.2,.7,.8]);new=np.array([.2,.3,.8,.9])
        prior,matrices=confusion(y,old,new)
        for soft in [False,True]:
            initial,posterior=beliefs(old,new,prior,matrices,soft)
            self.assertTrue(np.all((initial>0)&(initial<1)))
            self.assertTrue(np.all((posterior>0)&(posterior<1)))


if __name__=='__main__':unittest.main()
