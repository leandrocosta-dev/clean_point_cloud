"""Kernels de vizinhos por raio (CuPy) contra a referencia exata em CPU; agrupamento GPU do fundir_nitido."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
import vizinhos_gpu as V


def _superficie(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.uniform(0, 8, (n, 3))
    F[:, 2] = 0.01 * rng.standard_normal(n)
    return F


@unittest.skipUnless(V.disponivel(), 'CUDA/CuPy ausente')
class GradeGPUTest(unittest.TestCase):
    def test_mean_shift_igual_a_referencia_de_raio(self):
        F = _superficie()
        Q = F[:800] + [0, 0, 0.04]
        NQ = np.tile([0, 0, 1.0], (len(Q), 1))
        W = np.random.default_rng(1).uniform(0.2, 1, len(F))
        VF = np.tile([0, 0, 1.0], (len(F), 1))
        VF[::7] = [0, 0, -1.0]                     # face oposta nao conta
        D = V.Grade(F, 0.35).mean_shift(Q, NQ, W, VF, 0.12, 0.08)
        R = V.mean_shift_cpu(F, Q, NQ, W, VF, 0.35, 0.12, 0.08)
        self.assertLess(float(np.abs(D - R).max()), 2e-6)   # mm
        self.assertLess(float(D.mean()), 0)                  # puxa de volta para a superficie

    def test_contagem_e_mais_proximo(self):
        F = _superficie()
        Q = F[:2000] + [0, 0, 0.05]
        g = V.Grade(F, 0.3)
        n = g.contar(Q)
        r = V.contar_cpu(F, Q, 0.3)
        self.assertLessEqual(int(np.abs(n - r).max()), 1)         # so empate exatamente no raio (float32)
        self.assertGreaterEqual(float(np.mean(n == r)), 0.995)
        from scipy.spatial import cKDTree
        d, j = cKDTree(F).query(Q, distance_upper_bound=0.3)
        dg, jg = g.mais_proximo(Q)
        ok = np.isfinite(d)
        self.assertTrue(np.array_equal(ok, jg >= 0))
        self.assertGreaterEqual(float(np.mean(j[ok] == jg[ok])), 0.999)
        self.assertLess(float(np.abs(d[ok] - dg[ok]).max()), 1e-5)

    def test_pca_igual_numpy(self):
        F = _superficie(20000)
        Q = F[:300]
        s0, M, _ = V.Grade(F, 0.4).pca(Q, sigma=0.18)
        from scipy.spatial import cKDTree
        arv = cKDTree(F)
        for i, lista in enumerate(arv.query_ball_point(Q, 0.4)[:50]):
            e = F[lista] - Q[i]
            w = np.exp(-(e * e).sum(1) / (2 * 0.18 ** 2))
            m = (w[:, None] * e).sum(0) / w.sum()
            y = e - m
            R = np.einsum('k,ki,kj->ij', w, y, y)
            self.assertAlmostEqual(float(s0[i]), float(w.sum()), delta=1e-3 * w.sum())
            np.testing.assert_allclose(M[i], R, atol=2e-5 * w.sum())
        # normal da superficie ~ z
        n = np.linalg.eigh(M)[1][:, :, 0]
        self.assertGreater(float(np.median(np.abs(n[:, 2]))), 0.99)

    def test_grupos_cuda_identico_ao_cpu(self):
        import fundir_nitido as fn
        k = np.random.default_rng(3).integers(-50, 50, (100000, 3))
        fn.MOTOR = 'cpu'
        a = fn.grupos(k)
        fn.MOTOR = 'cuda'
        try:
            b = fn.grupos(k)
        finally:
            fn.MOTOR = 'cpu'
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class FallbackTest(unittest.TestCase):
    def test_referencia_cpu_planicie(self):
        F = _superficie(5000)
        Q = F[:50]
        D = V.mean_shift_cpu(F, Q, np.tile([0, 0, 1.0], (50, 1)), np.ones(len(F)), np.tile([0, 0, 1.0], (len(F), 1)),
                             0.4, 0.12, 0.08)
        self.assertLess(float(np.abs(D).max()), 0.05)          # ruido de 0,01 mm, amostra esparsa


if __name__ == '__main__':
    unittest.main()
