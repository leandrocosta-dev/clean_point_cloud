"""Refino de poses por superficie de mesma face (pipeline/refinar_poses.py) e modo de 2 scans do ciclo."""

import argparse
import unittest

import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL


def relevo(seed, lo=0.0, hi=24.0):
    """Superficie z = f(x, y) com curvatura nos dois eixos (prende os 6 graus de liberdade) e normais para +z."""
    rng = np.random.default_rng(seed)
    g = np.arange(lo, hi, 0.15)
    x, y = np.meshgrid(g, g)
    x, y = x.ravel() + rng.normal(0, 0.01, x.size), y.ravel() + rng.normal(0, 0.01, y.size)
    z = 2.0 * np.sin(x / 4.0) + 1.5 * np.cos(y / 3.0) + 0.02 * x * y / 10.0
    fx = 0.5 * np.cos(x / 4.0) + 0.002 * y
    fy = -0.5 * np.sin(y / 3.0) + 0.002 * x
    N = np.c_[-fx, -fy, np.ones_like(x)]
    return np.c_[x, y, z + rng.normal(0, 0.005, x.size)], N / np.linalg.norm(N, axis=1, keepdims=True)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class RefinarPosesTest(unittest.TestCase):
    def setUp(self):
        import ciclo_uniao
        import gpu
        import refinar_poses

        self.gpu, self.CU, self.RP = gpu, ciclo_uniao, refinar_poses
        self.a = argparse.Namespace(
            voxel=0.15, raio=0.4, viz_min=6, cos_min=0.6, tol=0.15, acordo=0.10, min_scans=2, area_min=0.5,
            alcance_normal=1.2, orientados={"a", "b", "c"},
        )  # fmt: skip

    def sup(self, nomes, erro=None):
        m, out = self.gpu.xp(), {}
        for i, nome in enumerate(nomes):
            P, N = relevo(i + 1)
            if erro and nome in erro:
                R, t = erro[nome]
                cen = P.mean(0)
                P, N = (P - cen) @ R.T + cen + t, N @ R.T
            out[nome] = dict(P=self.gpu.para_gpu(P), N=self.gpu.para_gpu(N), obs=m.ones(len(P)))
        return out

    def test_recupera_pose_deslocada_e_mantem_a_referencia(self):
        R = self.RP.rodrigues(np.radians([0.25, -0.2, 0.3]))
        t = np.array([0.10, -0.08, 0.20])
        for nomes in (("a", "b"), ("a", "b", "c")):
            with self.subTest(scans=len(nomes)):
                sup = self.sup(nomes, {"b": (R, t)})
                poses = {s: np.eye(4) for s in nomes}
                poses["b"] = np.eye(4) + 1e-6  # a referencia e quem esta mais perto da identidade: "a"
                novas, rel = self.RP.refinar(sup, poses, self.a, cel=8.0)
                self.assertTrue(rel["aceito"])
                self.assertEqual(rel["referencia"], "a")
                self.assertTrue(np.array_equal(novas["a"], np.eye(4)))
                self.assertGreater(rel["antes"]["b"]["desvio_p50_mm"], 0.1)
                self.assertLess(rel["depois"]["b"]["desvio_p50_mm"], 0.02)
                self.assertLess(rel["depois"]["b"]["vies_abs_max_mm"], 0.03)
                bom, _ = relevo(2)  # onde o scan b deveria estar
                self.assertLess(float(np.abs(self.gpu.para_cpu(sup["b"]["P"]) - bom).max()), 0.05)

    def test_dois_scans_nao_cortam_nada_e_medem_o_par(self):
        m = self.gpu.xp()
        sup = self.sup(("a", "b"), {"b": (np.eye(3), np.array([0.0, 0.0, 0.5]))})  # b 0,5 mm acima: quem esta certo?
        con = self.CU.Consenso(sup, self.a)
        an = con.analisar()
        self.assertEqual(int(an["a"]["defeito"].sum()) + int(an["b"]["defeito"].sum()), 0)
        r = self.CU.resumo(an, con.vivos, self.a)
        self.assertIsNone(r["a"]["desvio_p90_mm"])  # sem consenso nao ha medida de maioria
        self.assertGreater(r["a"]["coberto_por_outro_pct"], 95)
        self.assertAlmostEqual(r["a"]["desvio_coberto_p50_mm"], 0.5, delta=0.1)
        self.assertIsNone(self.CU.ruido_entre_scans(an, con.vivos))
        self.assertAlmostEqual(self.CU.ruido_entre_scans(an, con.vivos, par=True)["p50_mm"], 0.5, delta=0.1)
        self.assertTrue(bool(m.all(an["a"]["n_cobrem"] <= 1)))


if __name__ == "__main__":
    unittest.main()
