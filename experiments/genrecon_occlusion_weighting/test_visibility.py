"""Checks for visibility.py against GenRecon's own aggregators. Needs only torch:

    uv run --with torch python -m unittest discover -s experiments/genrecon_occlusion_weighting -p "test_*.py" -v
"""

import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
# genrecon/__init__.py imports every model and pipeline (and their GPU dependencies); the aggregators need
# only torch, so register the package without running it.
if "genrecon" not in sys.modules:
    pkg = types.ModuleType("genrecon")
    pkg.__path__ = [str(ROOT / "papers/GenRecon/genrecon")]
    sys.modules["genrecon"] = pkg
from genrecon.modules.cond_3D import aggregation_net, sparse_aggregation_net  # noqa: E402

import visibility as vis  # noqa: E402

P = vis.Params()


class Sparse:
    """What both sparse forwards use of a SparseTensor."""

    def __init__(self, feats):
        self.feats = feats

    def replace(self, feats):
        return Sparse(feats)


def randomised(agg, dtype=torch.float32):
    """GenRecon zero-initialises the last feature layer; random weights exercise every term."""
    torch.manual_seed(0)
    for prm in agg.parameters():
        prm.data.normal_(0, 0.3)
    return agg.to(dtype).eval()


def camera():
    """One camera at the origin looking along +z, focal 1 (normalised), principal point at the centre."""
    w2c = torch.eye(4)[None]
    K = torch.tensor([[[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]])
    return w2c, K


class TestVisibility(unittest.TestCase):
    def setUp(self):
        self.w2c, self.K = camera()
        self.depth = torch.full((1, 8, 8), 2.0)  # a wall at z = 2
        self.conf = torch.ones(1, 8, 8)

    def v(self, z, conf=None, voxel=0.05):
        points = torch.tensor([[0.0, 0.0, z]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        conf = self.conf if conf is None else conf
        return vis.visibility(points, self.w2c, self.K, self.depth, conf, valid, voxel, P)[0].item()

    def test_surface_kept(self):
        self.assertGreater(self.v(2.0), 0.95)
        self.assertGreater(self.v(2.0 + 0.03), 0.9)  # well within tau = max(1.5 * 0.05, 0.02 * 2) = 0.075
        self.assertAlmostEqual(self.v(2.0 + 0.075), 0.5, places=5)  # g(tau) = 0.5

    def test_free_space_kept(self):
        self.assertGreater(self.v(0.5), 0.99)  # Delta < 0: must not be removed

    def test_occluded_removed(self):
        self.assertLess(self.v(3.0), 1e-3)  # far behind the wall

    def test_tau_grows_with_depth(self):
        # The same Delta of 0.15 is occluded at 2 m but tolerated at 20 m (tau = 0.4).
        self.assertLess(self.v(2.15), 0.1)
        self.depth.fill_(20.0)
        self.assertGreater(self.v(20.15), 0.9)

    def test_low_confidence_reverts_to_genrecon(self):
        self.assertEqual(self.v(3.0, conf=torch.zeros(1, 8, 8)), 1.0)
        self.assertAlmostEqual(self.v(3.0, conf=torch.full((1, 8, 8), 0.4)), 0.6, places=3)

    def test_unknown_depth_reverts_to_genrecon(self):
        self.depth.fill_(float("inf"))  # e.g. a scan ray that hits nothing
        self.assertEqual(self.v(3.0), 1.0)

    def test_invalid_view_is_zero(self):
        points = torch.tensor([[0.0, 0.0, -1.0]])  # behind the camera
        valid = torch.zeros(1, 1, dtype=torch.bool)
        v = vis.visibility(points, self.w2c, self.K, self.depth, self.conf, valid, 0.05, P)[0]
        self.assertEqual(v.item(), 0.0)

    def test_pixel_lookup(self):
        # Left half of the image at depth 1, right half at depth 3: x < 0 projects to the left half.
        self.depth[:, :, :4] = 1.0
        self.depth[:, :, 4:] = 3.0
        points = torch.tensor([[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        v = vis.visibility(points, self.w2c, self.K, self.depth, self.conf, valid, 0.05, P)[0][0]
        self.assertLess(v[0].item(), 1e-3)  # behind the left surface
        self.assertGreater(v[1].item(), 0.99)  # in front of the right surface

    def test_fallback(self):
        valid = torch.tensor([[True], [True], [False]])
        v, alpha = vis.fallback(torch.tensor([[0.0], [0.0], [0.0]]), valid, kappa=1.0)
        self.assertEqual(v[:, 0].tolist(), [1.0, 1.0, 0.0])  # hidden everywhere: GenRecon's weights
        v, alpha = vis.fallback(torch.tensor([[1.0], [0.0], [0.0]]), valid, kappa=1.0)
        self.assertEqual(v[:, 0].tolist(), [1.0, 0.0, 0.0])
        v, alpha = vis.fallback(torch.tensor([[0.5], [0.0], [0.0]]), valid, kappa=1.0)
        self.assertEqual(v[:, 0].tolist(), [0.75, 0.5, 0.0])


class TestAggregators(unittest.TestCase):
    """v = valid must give GenRecon's output exactly; v = 0 must remove a view."""

    def dense(self, dtype):
        agg = randomised(aggregation_net.MultiViewFeatAggregator(channels=16, resolution=4), dtype)
        torch.manual_seed(1)
        feats = torch.randn(1, 5, 64, 16).to(dtype)
        mask = torch.rand(1, 5, 64) > 0.4
        mask[:, :, :3] = False  # voxels seen by no view
        return agg, feats, mask

    def sparse(self, dtype):
        agg = randomised(sparse_aggregation_net.MultiViewFeatAggregator(channels=16), dtype)
        torch.manual_seed(1)
        feats = torch.randn(64, 5, 16).to(dtype)
        mask = torch.rand(64, 5) > 0.4
        mask[:3] = False
        return agg, feats, mask

    def test_dense_identity(self):
        for dtype in (torch.float32, torch.bfloat16):
            agg, feats, mask = self.dense(dtype)
            for mode in vis.VAR_MODES:
                with torch.no_grad():
                    ref = agg(feats, mask)
                    out = vis.dense_forward(agg, feats, mask, mask.float(), vis.Params(var_mode=mode))
                self.assertTrue(torch.equal(ref, out), (dtype, mode))

    def test_sparse_identity(self):
        for dtype in (torch.float32, torch.bfloat16):
            agg, feats, mask = self.sparse(dtype)
            for mode in vis.VAR_MODES:
                with torch.no_grad():
                    ref = agg(Sparse(feats), mask).feats
                    out = vis.sparse_forward(agg, Sparse(feats), mask, mask.float(), vis.Params(var_mode=mode)).feats
                self.assertTrue(torch.equal(ref, out), (dtype, mode))

    def test_variance_modes(self):
        # One view of three hidden: original ignores v, around_vmean = original + (mu - mu_v)^2 >= original.
        feats = torch.tensor([[[0.0], [1.0], [5.0]]])  # [S=1, N=3, D=1]
        mask = torch.ones(1, 3, dtype=torch.bool)
        v = torch.tensor([[1.0, 1.0, 0.0]])
        var = {m: vis.moments(feats, mask, v, m, one_pass=False)[1].item() for m in vis.VAR_MODES}
        mu_v = vis.moments(feats, mask, v, "original", one_pass=False)[0].item()
        self.assertAlmostEqual(mu_v, 0.5, places=5)
        self.assertAlmostEqual(var["weighted"], 0.25, places=5)  # over views 0 and 1 only
        self.assertAlmostEqual(var["original"], 14 / 3, places=5)  # around mu = 2: (4 + 1 + 9) / 3
        self.assertAlmostEqual(var["around_vmean"], 14 / 3 + 1.5 ** 2, places=5)  # (0.25 + 0.25 + 20.25) / 3
        # the dense layout [B, N, V, D]
        dense = {m: vis.moments(feats[:, :, None], mask[..., None], v[..., None], m, one_pass=True)[1].item() for m in vis.VAR_MODES}
        for m in vis.VAR_MODES:
            self.assertAlmostEqual(dense[m], var[m], places=5, msg=m)

    def test_occluded_view_is_removed(self):
        # Dropping a view with v = 0 matches (up to the eps term) running GenRecon without that view.
        agg, feats, mask = self.sparse(torch.float32)
        mask[:] = True
        v = torch.ones(64, 5)
        v[:, 2] = 0.0
        keep = [0, 1, 3, 4]
        with torch.no_grad():
            ref = agg(Sparse(feats[:, keep]), mask[:, keep]).feats
            out = vis.sparse_forward(agg, Sparse(feats), mask, v, vis.Params(eps=1e-9)).feats
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

        agg, feats, mask = self.dense(torch.float32)
        mask[:] = True
        v = torch.ones(1, 5, 64)
        v[:, 2] = 0.0
        with torch.no_grad():
            ref = agg(feats[:, keep], mask[:, keep])
            out = vis.dense_forward(agg, feats, mask, v, vis.Params(eps=1e-9))
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
