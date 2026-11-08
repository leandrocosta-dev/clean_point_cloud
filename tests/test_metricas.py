"""Metricas padrao (pipeline/metricas.py) com geometria sintetica de resposta conhecida."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import metricas as M  # noqa: E402


def plano(lado=60.0, passo=0.25, z=0.0):
    g = np.arange(0, lado, passo)
    x, y = np.meshgrid(g, g)
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


class MetricasTest(unittest.TestCase):
    def test_distancias_deslocamento_conhecido(self):
        B = plano()
        A = B + [0, 0, 0.2]
        r = M.distancias(A, B)
        self.assertAlmostEqual(r["a_para_ref"]["mediana_mm"], 0.2, places=6)
        self.assertAlmostEqual(r["ref_para_a"]["mediana_mm"], 0.2, places=6)
        self.assertEqual(r["cobertura_ref"]["0.3mm"], 1.0)
        self.assertEqual(r["cobertura_ref"]["0.1mm"] if "0.1mm" in r["cobertura_ref"] else 0.0, 0.0)
        self.assertEqual(r["a_para_ref"]["acima_limiar_pct"], 0.0)

    def test_vies_com_sinal(self):
        B = plano()
        N = np.tile([0.0, 0, 1], (len(B), 1))
        r = M.vies_por_celula(B + [0, 0, 0.05], B, N, cel=30.0, tol=0.3)
        self.assertEqual(r["celulas"], 4)
        self.assertAlmostEqual(r["vies_abs_mediana_um"], 50.0, places=3)
        self.assertAlmostEqual(r["vies_assinado_min_um"], 50.0, places=3)
        r2 = M.vies_por_celula(B - [0, 0, 0.05], B, N)
        self.assertAlmostEqual(r2["vies_assinado_max_um"], -50.0, places=3)
        self.assertEqual(M.vies_por_celula(B + [0, 0, 5.0], B, N)["celulas"], 0)   # fora da tolerancia

    def test_rugosidade_plano_vs_ruidoso(self):
        B = plano(30.0)
        liso = M.rugosidade(B, raio=1.0, n=3000)
        ruido = B + np.c_[np.zeros((len(B), 2)), np.random.default_rng(0).normal(0, 0.05, len(B))]
        aspero = M.rugosidade(ruido, raio=1.0, n=3000)
        self.assertLess(liso["mediana_um"], 1e-3)
        self.assertGreater(aspero["mediana_um"], 30.0)
        self.assertLess(aspero["mediana_um"], 70.0)

    def test_componentes(self):
        a = plano(5.0)
        b = plano(5.0) + [20.0, 0, 0]
        r = M.componentes(np.vstack([a, b]), voxel=0.5)
        self.assertEqual(r["total"], 2)
        self.assertAlmostEqual(r["fracao_maior"], 0.5)
        self.assertEqual(M.componentes(a)["pontos_fora_do_maior"], 0)

    def test_lixo_sem_confirmacao(self):
        base = plano(20.0)
        lixo = np.random.default_rng(1).uniform(0, 20, (100, 3)) + [0, 0, 10.0]
        r = M.lixo_sem_confirmacao([np.vstack([base, lixo]), base.copy(), base + [0, 0, 0.1]], tol=0.4)
        pct = r["por_nuvem"]
        self.assertAlmostEqual(pct[0]["sem_confirmacao_pct"], 100 * 100 / (len(base) + 100), places=6)
        self.assertEqual(pct[1]["sem_confirmacao_pct"], 0.0)
        self.assertEqual(pct[2]["sem_confirmacao_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
