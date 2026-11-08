"""Agrupamento de linhas repetidas mantem a particao numpy sem unique(axis=0)."""

import unittest

import gpu
import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA indisponivel")
class AgrupamentoLinhasTest(unittest.TestCase):
    def test_linhas_repetidas_negativas_e_degeneradas(self):
        rng = np.random.default_rng(31)
        for k in (
            np.zeros((20000, 3), dtype=np.int64),
            rng.integers(-4, 4, (12000, 5), dtype=np.int64),
            np.array([[2, 0], [-1, 9], [-1, 1], [2, 0]], dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((3, 0), dtype=np.int64),
        ):
            with self.subTest(forma=k.shape):
                esperado = np.unique(k, axis=0, return_inverse=True, return_counts=True)
                obtido = gpu.agrupar(k)
                for a, b in zip(obtido, esperado, strict=True):
                    np.testing.assert_array_equal(a.reshape(-1), b.reshape(-1))

    def test_saida_permanece_no_dispositivo(self):
        m = gpu.xp()
        k = m.array([[1, 2], [1, 2], [0, 9]], dtype=m.int64)
        u, inv, cnt = gpu.agrupar(k)
        self.assertIsInstance(u, m.ndarray)
        self.assertIsInstance(inv, m.ndarray)
        self.assertIsInstance(cnt, m.ndarray)
        np.testing.assert_array_equal(gpu.para_cpu(cnt), [1, 2])


if __name__ == "__main__":
    unittest.main()
