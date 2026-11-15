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
        an = CU.analisar(sup, vivos, self.a)
        return {s: {k: v if k == "de" else self.gpu.para_cpu(v) for k, v in x.items()} for s, x in an.items()}

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

    def test_memoria_entre_rodadas_da_o_mesmo_que_recalcular(self):
        m = self.gpu.xp()
        A, B, C = placa(seed=1), placa(seed=2), placa(seed=3)
        C[(np.abs(C[:, 0] - 10) < 3) & (np.abs(C[:, 1] - 10) < 3), 2] += 0.4
        sup = self.sup(dict(a=A, b=B, c=C))
        con = CU.Consenso(sup, self.a)
        con.analisar()  # enche a memoria com todos os pares
        fora = self.gpu.para_gpu((np.abs(A[:, 0] - 6) < 2) & (np.abs(A[:, 1] - 6) < 2))
        self.assertEqual(con.remover("a", fora), int(fora.sum()))
        com_memoria = con.analisar()
        do_zero = CU.analisar(sup, {s: v.copy() for s, v in con.vivos.items()}, self.a)
        for s in sup:
            for k in ("desvio", "n_cobrem", "consenso", "defeito"):
                self.assertTrue(bool(m.array_equal(com_memoria[s][k], do_zero[s][k])), (s, k))
        miolo = self.gpu.para_gpu((np.abs(B[:, 0] - 6) < 1.3) & (np.abs(B[:, 1] - 6) < 1.3))
        self.assertEqual(int(com_memoria["b"]["n_cobrem"][miolo].max()), 1)  # ali so sobrou o scan c

    def test_vies_de_pose_aparece_com_desvio_pequeno_demais_para_ser_defeito(self):
        m = self.gpu.xp()
        sup = self.sup(dict(a=placa(seed=1), b=placa(seed=2), c=placa(dz=0.08, seed=3)))
        con = CU.Consenso(sup, self.a)
        an = con.analisar()
        self.assertEqual(int(an["c"]["defeito"].sum()), 0)  # 0,08 mm < tol: nada e cortado
        v = {s: CU.vies_por_celula(sup[s]["P"], an[s]["desvio"], an[s]["consenso"], 10.0) for s in sup}
        self.assertGreater(v["c"]["celulas"], 3)
        self.assertAlmostEqual(v["c"]["vies_abs_max_mm"], 0.08, delta=0.015)
        self.assertAlmostEqual(v["c"]["vies_global_mm"], -0.08, delta=0.015)  # os outros estao ABAIXO de c
        self.assertAlmostEqual(v["a"]["vies_abs_max_mm"], 0.04, delta=0.015)  # consenso de a = meio entre b e c
        ruido = CU.ruido_entre_scans(an, con.vivos)
        self.assertLess(ruido["p50_mm"], ruido["p90_mm"])
        self.assertLess(ruido["p90_mm"], 0.1)
        self.assertEqual(
            CU.vies_por_celula(sup["a"]["P"], an["a"]["desvio"], m.zeros(len(sup["a"]["P"]), bool), 10.0),
            dict(celulas=0),
        )

    def test_duvida_confirmada_vira_peca_e_a_que_ninguem_confirma_continua(self):
        m = self.gpu.xp()
        C = placa(seed=3, hi=30.0)  # c vai alem de a e b
        sup = self.sup(dict(a=placa(seed=1), b=placa(seed=2), c=C))
        self.a.duvida_peca = 0.5
        x = CU.Consenso(sup, self.a).analisar()["c"]
        reg = np.zeros(len(C), np.int64)
        reg[(C[:, 0] > 5) & (C[:, 0] < 9) & (C[:, 1] > 5) & (C[:, 1] < 9)] = 7  # coberta pelos tres scans
        reg[(C[:, 0] > 24) & (C[:, 1] > 24)] = 9  # so c tem
        pend, res = CU.resolver_duvidas(x, self.gpu.para_gpu(reg), set(), [True, True], self.a)
        self.assertEqual((res["regioes"], res["viraram_peca"]), (2, 1))
        self.assertTrue(bool(m.array_equal(pend, self.gpu.para_gpu(reg == 9))))
        # regiao deitada no apoio e os outros dois scans tem o MESMO apoio: a confirmacao deles nao vale
        pend, res = CU.resolver_duvidas(x, self.gpu.para_gpu(reg), {7}, [False, False], self.a)
        self.assertEqual(res["viraram_peca"], 0)
        self.assertEqual(res["pontos_pendentes"], int((reg > 0).sum()))

    def test_descarte_so_do_scan_muito_pior_que_os_outros(self):
        pct = dict(a=2.0, b=1.1, c=9.6, d=3.5)
        self.assertEqual(CU.escolher_descarte(pct, 6.0, 2.5), ("c", 2.0))
        self.assertIsNone(CU.escolher_descarte(dict(pct, c=5.9), 6.0, 2.5))  # abaixo do minimo
        self.assertIsNone(CU.escolher_descarte(dict(a=5.0, b=4.0, c=9.6), 6.0, 2.5))  # todos ruins: nao e so ele


if __name__ == "__main__":
    unittest.main()
