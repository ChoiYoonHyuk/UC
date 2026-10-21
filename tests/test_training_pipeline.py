"""Small deterministic checks for the optional predictor-training modules."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

SCIENTIFIC = all(importlib.util.find_spec(name) is not None for name in ("scipy", "sklearn"))
TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(SCIENTIFIC, "Optional scipy/scikit-learn training dependencies are not installed.")
class TrainingTests(unittest.TestCase):
    def test_tree_fitting_excludes_evaluation_responses(self):
        from harm_budgeted_updates.training.tree_pipeline import run
        from harm_budgeted_updates.training.data import uniform_roles
        rng = np.random.default_rng(814)
        n = 840
        x = rng.normal(size=(n, 4)).astype(np.float32)
        y = (x[:, 0] + rng.normal(size=n) > 0).astype(int)
        roles = uniform_roles(y, "toy", 0)
        nodes = np.arange(n)
        edges = np.vstack((np.r_[nodes, (nodes + 1) % n], np.r_[(nodes + 1) % n, nodes]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (0, 1):
                response = y.copy()
                if index:
                    response[roles["test"]] = 1 - response[roles["test"]]
                path = root / ("input_" + str(index) + ".npz")
                np.savez_compressed(path, x=x, y=response, edge_index=edges, **roles)
                run(path, root / str(index), "toy", 0, 550, ("HGBGraph",))
            name = "toy__draw_0__HGBGraph.npz"
            with np.load(root / "0" / name) as a, np.load(root / "1" / name) as b:
                np.testing.assert_array_equal(a["b"], b["b"])
                np.testing.assert_array_equal(a["t"], b["t"])
                self.assertTrue(np.all((a["t"] > 0) & (a["t"] < 1)))

    def test_large_acquisition_preserves_calibration_and_test_disjointness(self):
        from harm_budgeted_updates.training.tree_pipeline import enlarged_roles
        from harm_budgeted_updates.training.data import uniform_roles
        y = np.arange(6200) % 2
        roles = uniform_roles(y, "toy", 0)
        fit, evaluate = enlarged_roles(roles, "toy", 0, 5000)
        self.assertEqual(int(fit.sum()), 5000)
        self.assertFalse(np.any(fit & roles["calibrate"]))
        self.assertFalse(np.any(fit & evaluate))
        self.assertTrue(np.all(evaluate <= roles["test"]))

    def test_chronological_fitting_excludes_current_responses(self):
        from harm_budgeted_updates.training.temporal import run
        rng = np.random.default_rng(941)
        steps = np.repeat(np.arange(1, 8), 40)
        x = rng.normal(size=(len(steps), 93)).astype(np.float32)
        y = (np.arange(len(steps)) % 2).astype(int)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (0, 1):
                response = y.copy()
                if index:
                    response[steps == 7] = 1 - response[steps == 7]
                path = root / ("input_" + str(index) + ".npz")
                np.savez_compressed(path, x=x, y=response, time_step=steps)
                run(path, root / str(index), (7,), ("HGBLocal",))
            name = "elliptic__step_7__HGBLocal.npz"
            with np.load(root / "0" / name) as a, np.load(root / "1" / name) as b:
                np.testing.assert_array_equal(a["b"], b["b"])
                np.testing.assert_array_equal(a["t"], b["t"])
                self.assertEqual(len(a["b"]), 40)


@unittest.skipUnless(SCIENTIFIC and TORCH, "Optional PyTorch/scientific training dependencies are not installed.")
class GraphSAGETests(unittest.TestCase):
    def test_original_objectives_execute_on_cpu(self):
        import torch
        from scipy import sparse
        from harm_budgeted_updates.training.graphsage import adj, train
        torch.set_num_threads(1)
        rng = np.random.default_rng(517)
        n = 36
        x = torch.tensor(rng.normal(size=(n, 4)).astype(np.float32))
        nodes = np.arange(n)
        adjacency = sparse.csr_matrix((np.ones(n), (nodes, (nodes + 1) % n)), shape=(n, n))
        P, _ = adj(adjacency, "cpu")
        px = torch.sparse.mm(P, x)
        y = np.arange(n) % 2
        fit = np.arange(n) < 20
        stop = (np.arange(n) >= 20) & (np.arange(n) < 30)
        teacher = np.linspace(.1, .9, n)
        for mode in ("CE", "Distill", "FD"):
            _, probability, record = train(x, P, px, y, fit, stop, 155, mode, teacher, .5)
            self.assertEqual(probability.shape, (n,))
            self.assertTrue(np.all(np.isfinite(probability)))
            self.assertTrue(np.all((probability > 0) & (probability < 1)))
            self.assertLessEqual(record["epochs"], 80)
            self.assertEqual(record["fit_count"], 20)


if __name__ == "__main__":
    unittest.main()
