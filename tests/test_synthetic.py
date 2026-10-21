"""Small deterministic checks for inference, intervals, and unknown responses."""
import unittest
import numpy as np
from harm_budgeted_updates import tree, queues
from harm_budgeted_updates.finite_intervals import clopper_pearson_table
from harm_budgeted_updates.dependence import allocate, harm_vectors
from harm_budgeted_updates.allocation import dinfinity_binary


class SyntheticChecks(unittest.TestCase):
    def test_tree_messages_match_exact_enumeration(self):
        rng=np.random.default_rng(1121)
        for _ in range(4):
            n=9;edges=np.array([[(v-1)//2,v] for v in range(1,n)])
            fields=rng.normal(-.3,.8,n);j=rng.uniform(-.4,.4,n-1)
            observed=rng.random(n)<.25;labels=rng.integers(0,2,n)
            eta,_,_=tree.exact_tree(fields,edges,j,observed,labels)
            enumerated=tree.exact_enumeration(fields,edges,j,observed,labels)
            np.testing.assert_allclose(eta,enumerated,atol=3e-14,rtol=0)

    def test_interval_binomial_endpoints(self):
        cp=clopper_pearson_table(10,groups=2,alpha=.05)
        self.assertEqual(cp[0,0],0.)
        self.assertEqual(cp[10,1],1.)
        self.assertAlmostEqual(cp[0,1],1-(.05/4)**.1,14)
        self.assertTrue(np.all(cp[:,0]<=np.arange(11)/10))
        self.assertTrue(np.all(cp[:,1]>=np.arange(11)/10))

    def test_unknown_responses_stay_in_deployment_queue(self):
        b=np.array([.6,.5,.5,.2]);p=np.array([.4,.7,.5,.3])
        y=np.array([1,-1,0,1]);ids=np.array([3,2,1,0])
        outcome,_=queues.evaluate_queue(b,p,y,.5,ids)
        self.assertEqual(outcome['capacity'],2)
        self.assertEqual(outcome['unknown'],1)
        self.assertEqual(outcome['reference_known_TP'],1)
        self.assertEqual(outcome['known_TP'],0)
        self.assertEqual(outcome['known_lost_TP'],1)
        self.assertEqual(outcome['known_enter_TP'],0)
        self.assertEqual(outcome['known_positive_recall'],0.)

    def test_deterministic_harm_limit_all_response_vectors(self):
        data,_,_=allocate(8)
        y=np.array([[int(c) for c in f'{i:08b}'] for i in range(256)])
        for name in ['UC','Point','RUC']:
            harm=harm_vectors(y,data,data[name])
            cost=dinfinity_binary(data['b'],data[name])
            self.assertTrue(np.all(harm.mean(axis=1)<=cost.mean()+1e-13))
            self.assertLessEqual(float(cost.mean()),.0025+1e-13)


if __name__=='__main__':unittest.main()
