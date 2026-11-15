"""Conferencia da fusao final (pipeline/conferir_uniao.py): apoio por numero de scans, soltos, zonas fracas e malha."""

import os
import tempfile
import unittest

import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL


def grade(lo, hi, passo, z=0.0):
    g = np.arange(lo, hi, passo)
    x, y = np.meshgrid(g, g)
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class ConferirUniaoTest(unittest.TestCase):
    def setUp(self):
        import conferir_uniao
        import fluxo_uniao
        import gpu

        self.gpu, self.CF, self.FL = gpu, conferir_uniao, fluxo_uniao

    def test_apoio_soltos_e_zona_fraca(self):
        g = self.gpu
        sup = {
            "a": dict(P=g.para_gpu(grade(0, 20, 0.15))),
            "b": dict(P=g.para_gpu(grade(0, 20, 0.15) + 0.02)),
            "c": dict(P=g.para_gpu(grade(0, 30, 0.15))),  # so c cobre x ou y > 20
        }
        corpo = grade(0, 30, 0.1)
        ilha = grade(40, 42, 0.1, z=5.0)  # solta e sem apoio de ninguem
        P = g.para_gpu(np.r_[corpo, ilha])
        rel, n, solto = self.CF.conferir_nuvem(P, sup, 0.3, 0.1)
        n, solto = g.para_cpu(n), g.para_cpu(solto)
        dentro = (corpo[:, 0] < 19) & (corpo[:, 1] < 19)
        so_c = (corpo[:, 0] > 21) | (corpo[:, 1] > 21)
        self.assertTrue((n[: len(corpo)][dentro] == 3).all())
        self.assertTrue((n[: len(corpo)][so_c] == 1).all())
        self.assertTrue((n[len(corpo) :] == 0).all())
        self.assertEqual(int(solto.sum()), len(ilha))
        self.assertEqual(rel["componentes"], 2)
        self.assertAlmostEqual(rel["sem_apoio_pct"], 100 * len(ilha) / len(P), places=2)
        self.assertEqual(rel["zonas_fracas"], 2)  # a faixa que so c cobre e a ilha
        self.assertGreater(rel["maiores_zonas_fracas"][0]["area_mm2"], 400)  # 30x30 - 20x20 = 500 mm2
        self.assertEqual(rel["maiores_zonas_fracas"][1]["sem_apoio_pct"], 100.0)
        self.assertAlmostEqual(self.CF.espacamento(P), 0.1, delta=0.01)

    def test_malha_com_buraco_e_pedaco_inventado(self):
        try:
            import open3d as o3d
            from meshlib import mrmeshpy  # noqa: F401
        except ImportError:
            self.skipTest("open3d/meshlib ausente")
        passo, n = 0.2, 60
        V = grade(0, n * passo - 1e-9, passo)
        idx = np.arange(n * n).reshape(n, n)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        T = np.r_[np.c_[a, b, c], np.c_[b, d, c]]
        cen = V[T].mean(1)
        T = T[~((np.abs(cen[:, 0] - 6) < 1.5) & (np.abs(cen[:, 1] - 6) < 1.5))]  # buraco de 3 x 3 mm
        solto = np.array([[30.0, 30, 0], [31, 30, 0], [30, 31, 0]])  # triangulo longe da nuvem
        M = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.r_[V, solto]),
            o3d.utility.Vector3iVector(np.r_[T, [[len(V), len(V) + 1, len(V) + 2]]]),
        )
        M.remove_unreferenced_vertices()
        with tempfile.TemporaryDirectory() as tmp:
            arq = os.path.join(tmp, "malha.ply")
            o3d.io.write_triangle_mesh(arq, M)
            rel = self.CF.conferir_malha(arq, self.gpu.para_gpu(grade(0, n * passo - 1e-9, 0.1)), 0.1)
        self.assertEqual(rel["componentes"], 2)
        self.assertGreaterEqual(rel["furos"], 3)  # contorno externo, buraco e o triangulo solto
        self.assertGreater(rel["inventada_pct"], 0)
        self.assertGreater(rel["vertice_a_nuvem_max_mm"], 20)
        self.assertGreater(rel["perdida_pct"], 3)  # ~9 de 144 mm2 sem malha, menos a borda de 0,3 mm
        self.assertLess(rel["perdida_pct"], 7)

    def test_tolerancias_pelo_ruido_reproduzem_as_aprovadas_no_projeto9(self):
        tol, acordo = self.FL.tolerancias_do_ruido(0.08)
        self.assertAlmostEqual(tol, 0.15, delta=0.005)
        self.assertAlmostEqual(acordo, 0.10, delta=0.005)
        self.assertEqual(self.FL.tolerancias_do_ruido(0.001), (0.10, 0.06))  # piso
        self.assertEqual(self.FL.tolerancias_do_ruido(5.0), (0.50, 0.35))  # teto


if __name__ == "__main__":
    unittest.main()
