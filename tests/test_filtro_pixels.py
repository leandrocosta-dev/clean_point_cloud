"""Regressoes sinteticas: executar com python -m unittest discover -s tests."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "revo_frames"))
from revo_projeto import filtro_pixels


class FiltroPixelsTest(unittest.TestCase):
    def filtrar(self, d):
        return filtro_pixels(d, tol=0.2, comp_min=1, adaptativo=True)

    def test_plano_inclinado(self):
        y, x = np.mgrid[:40, :40]
        self.assertTrue(self.filtrar(100 + 0.3*x + 0.2*y).all())

    def test_pico_isolado(self):
        d = np.full((40, 40), 100.0)
        d[20, 20] += 3
        ok = self.filtrar(d)
        self.assertFalse(ok[20, 20])
        self.assertEqual(int(ok.sum()), d.size - 1)

    def test_degrau_real(self):
        d = np.full((40, 40), 100.0)
        d[:, 20:] += 5
        self.assertTrue(self.filtrar(d).all())

    def test_estrutura_fina_sem_apoio(self):
        d = np.full((40, 40), np.nan)
        d[:, 20] = 100
        np.testing.assert_array_equal(self.filtrar(d), np.isfinite(d))

    def test_invalidos(self):
        d = np.array([[np.nan, np.inf], [0, -1]])
        self.assertFalse(self.filtrar(d).any())

    def test_parametros_invalidos(self):
        for tol in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                filtro_pixels(np.ones((5, 5)), tol=tol)


if __name__ == "__main__":
    unittest.main()
