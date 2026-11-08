"""Equivalencia numerica do agrupamento CPU/CUDA e independencia dos blocos."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from comparar_fusao_cuda import fundir


class FusaoTest(unittest.TestCase):
    def test_blocos_contagens_e_coordenadas_negativas(self):
        p = np.array([[-1, 0, 0], [-.99, 0, 0], [1, 0, 0], [1.01, 0, 0], [8, 0, 0]], np.float32)
        a, _ = fundir(p, .1, 2, 'cpu', 2)
        b, _ = fundir(p, .1, 2, 'cpu', 100)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, [[-.995, 0, 0], [1.005, 0, 0]], atol=1e-6)

    def test_cuda_corresponde_cpu(self):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():
                self.skipTest('CUDA indisponivel')
        except (ImportError, RuntimeError):
            self.skipTest('CuPy indisponivel')
        rng = np.random.default_rng(42)
        p = rng.normal(size=(10000, 3)).astype(np.float32)
        cpu, _ = fundir(p, .2, 2, 'cpu', 701)
        gpu, _ = fundir(p, .2, 2, 'cuda', 701)
        np.testing.assert_array_equal(cpu, gpu)


if __name__ == '__main__':
    unittest.main()
