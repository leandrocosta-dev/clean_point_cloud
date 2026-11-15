"""Ciclo fusao -> uniao -> analise (pipeline/ciclo_uniao.py): defeito = trecho de um scan fora do consenso dos outros."""

import argparse
import unittest

import ciclo_uniao as CU
import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL


def placa(dz=0.0, seed=0, lo=0.0, hi=20.0):
    rng = np.random.default_rng(seed)
    g = np.arange(lo, hi, 0.15)
    x, y = np.meshgrid(g, g)
    P = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)] + rng.normal(0, 0.01, (x.size, 3))
    P[:, 2] += dz
    return P


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class CicloUniaoTest(unittest.TestCase):
    def setUp(self):
        import gpu

        self.gpu = gpu
        self.a = argparse.Namespace(
            voxel=0.15,
            raio=0.4,
            viz_min=6,
            cos_min=0.6,
            tol=0.15,
            acordo=0.10,
            min_scans=2,
            area_min=0.5,
            alcance_normal=1.2,
            orientados={"a", "b", "c"},
        )

    def sup(self, nuvens):
        m = self.gpu.xp()
        out = {}
        for nome, P in nuvens.items():
            N = np.tile([0.0, 0.0, 1.0], (len(P), 1))
            out[nome] = dict(P=self.gpu.para_gpu(P), N=self.gpu.para_gpu(N), obs=m.ones(len(P)))
        return out

    def analisar(self, nuvens):
        sup = self.sup(nuvens)
        vivos = {s: self.gpu.xp().ones(len(x["P"]), bool) for s, x in sup.items()}
        return {s: {k: self.gpu.para_cpu(v) for k, v in x.items()} for s, x in CU.analisar(sup, vivos, self.a).items()}

    def test_so_o_trecho_deformado_do_scan_ruim_e_defeito(self):
        A, B, C = placa(seed=1), placa(seed=2), placa(seed=3)
        dentro = (np.abs(C[:, 0] - 10) < 3) & (np.abs(C[:, 1] - 10) < 3)
        C[dentro, 2] += 0.4  # bolha de 6 x 6 mm so no scan C
        an = self.analisar(dict(a=A, b=B, c=C))
        nucleo = (np.abs(C[:, 0] - 10) < 2.3) & (np.abs(C[:, 1] - 10) < 2.3)
        longe = (np.abs(C[:, 0] - 10) > 4) | (np.abs(C[:, 1] - 10) > 4)
        self.assertGreater(an["c"]["defeito"][nucleo].mean(), 0.95)
        self.assertEqual(int(an["c"]["defeito"][longe].sum()), 0)
        self.assertEqual(int(an["a"]["defeito"].sum()), 0)  # os scans bons nao perdem nada, nem sob a bolha
        self.assertEqual(int(an["b"]["defeito"].sum()), 0)

    def test_sem_maioria_nada_sai(self):
        an = self.analisar(dict(a=placa(seed=1), b=placa(dz=0.4, seed=2)))  # 2 scans divergem: quem esta certo?
        self.assertEqual(int(an["a"]["defeito"].sum()), 0)
        self.assertEqual(int(an["b"]["defeito"].sum()), 0)
        self.assertEqual(int(an["a"]["consenso"].sum()), 0)

    def test_trecho_que_so_um_scan_cobre_fica(self):
        A, B = placa(seed=1), placa(seed=2)
        C = placa(dz=0.0, seed=3, hi=30.0)  # C vai alem dos outros
        an = self.analisar(dict(a=A, b=B, c=C))
        fora = C[:, 0] > 21
        self.assertEqual(int(an["c"]["defeito"][fora].sum()), 0)
        self.assertTrue((an["c"]["n_cobrem"][fora] == 0).all())


if __name__ == "__main__":
    unittest.main()
