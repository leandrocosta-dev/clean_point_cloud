"""Fusao por voxel na GPU: independencia dos blocos, coordenadas negativas e contra a referencia numpy (o motor CPU foi removido)."""

import unittest

import numpy as np
from comparar_fusao_cuda import fundir

from tests.gpu_utils import CUDA_DISPONIVEL


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class FusaoTest(unittest.TestCase):
    def test_blocos_contagens_e_coordenadas_negativas(self):
        p = np.array([[-1, 0, 0], [-0.99, 0, 0], [1, 0, 0], [1.01, 0, 0], [8, 0, 0]], np.float32)
        a, _ = fundir(p, 0.1, 2, None, 2)
        b, _ = fundir(p, 0.1, 2, None, 100)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, [[-0.995, 0, 0], [1.005, 0, 0]], atol=1e-6)

    def test_gpu_corresponde_referencia_numpy(self):
        rng = np.random.default_rng(42)
        p = rng.normal(size=(10000, 3)).astype(np.float32)
        gpu_, _ = fundir(p, 0.2, 2, None, 701)
        # referencia numpy (CPU, so no teste): centroides por voxel com >= 2 pontos, voxels em ordem lexicografica
        x = p.astype(np.float64)
        lo = x.min(0) - 0.2
        k = np.floor((x - lo) / 0.2).astype(np.int64)
        u, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        ref = np.stack([np.bincount(inv, x[:, i]) / cnt for i in range(3)], 1)[cnt >= 2].astype(np.float32)
        np.testing.assert_allclose(gpu_, ref, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
