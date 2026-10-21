"""Small CPU checks for role separation and original training components."""
import unittest
import numpy as np
from harm_budgeted_updates.training.data import uniform_roles,validate_roles
from harm_budgeted_updates.training.tree_pipeline import enlarged_roles


class TrainingTests(unittest.TestCase):
    def test_disjoint_nested_acquisition(self):
        y=np.tile([0,1],1000)
        masks=uniform_roles(y,"demo",0)
        validate_roles(masks,y)
        fit,evaluation=enlarged_roles(masks,"demo",0,1000)
        self.assertEqual(int(fit.sum()),1000)
        self.assertFalse(np.any(fit&evaluation))
        self.assertFalse(np.any(masks['calibrate']&(fit|evaluation)))
        self.assertTrue(np.all(masks['train']<=fit))
        self.assertEqual(int(evaluation.sum()),800)


if __name__=='__main__':unittest.main()
