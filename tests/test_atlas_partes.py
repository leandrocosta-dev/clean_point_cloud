"""Atlas de partes (pipeline/atlas_partes.py): regioes, precedencia dos rotulos e render."""

import unittest

import atlas_partes as A
import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL


def cena(voxel=0.5):
    """Placa plana 40x40 (z=0), caixa 10x10x8 apoiada nela e um tufo rugoso encostado na caixa."""
    g = np.arange(0, 40, voxel)
    x, y = np.meshgrid(g, g)
    placa = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    placa = placa[~((placa[:, 0] > 15) & (placa[:, 0] < 25) & (placa[:, 1] > 15) & (placa[:, 1] < 25))]
    h = np.arange(15, 25 + 1e-9, voxel)
    a, b = np.meshgrid(h, h)
    topo = np.c_[a.ravel(), b.ravel(), np.full(a.size, 8.0)]
    z = np.arange(voxel, 8, voxel)
    a, b = np.meshgrid(h, z)
    lados = np.vstack(
        [
            np.c_[a.ravel(), np.full(a.size, 15.0), b.ravel()],
            np.c_[a.ravel(), np.full(a.size, 25.0), b.ravel()],
            np.c_[np.full(a.size, 15.0), a.ravel(), b.ravel()],
            np.c_[np.full(a.size, 25.0), a.ravel(), b.ravel()],
        ]
    )
    rng = np.random.default_rng(0)
    tufo = rng.uniform([25.5, 17, 0.5], [31, 23, 6], (2500, 3))
    return placa, np.vstack([topo, lados]), tufo


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class AtlasPartesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gpu

        cls.gpu = gpu
        m = gpu.xp()
        cls.voxel = 0.5
        placa, caixa, tufo = cena(cls.voxel)
        P = np.vstack([placa, caixa, tufo])
        chaves = gpu.chaves_voxel(gpu.para_gpu(P), cls.voxel, A.origem_grade(cls.voxel))
        u, prim = np.unique(gpu.para_cpu(chaves), return_index=True)
        cls.origem = np.where(prim < len(placa), 0, np.where(prim < len(placa) + len(caixa), 1, 2))
        C = gpu.para_gpu(P[prim])
        N, curv = gpu.normais_pca(
            C, raio=2.5 * cls.voxel, orientar_para=np.array([20.0, 20.0, 200.0]), com_curvatura=True
        )
        rot, tipo, grupo, pares = A.segmentar_regioes(C, N, curv, cls.voxel, bloco=3.0)
        comp, _ = gpu.componentes_conexos_voxel(C, cls.voxel, 26)
        um = m.ones(len(C))
        cls.at = dict(chaves=gpu.para_gpu(u), C=C, N=N, curv=curv, rot=rot, comp=comp, obs=um, quadros=um)
        cls.at.update(voxel=cls.voxel, tipo=tipo, grupo=grupo, conf=None)
        cls.planos = A.planos_candidatos(C, cls.voxel)
        cls.linhas = A.tabela_regioes(cls.at, tipo, pares, cls.planos, cls.voxel)
        cls.rot = gpu.para_cpu(rot)

    def _maior(self, origem):
        return int(np.bincount(self.rot[self.origem == origem]).argmax())

    def test_placa_e_caixa_ficam_em_regioes_lisas_diferentes(self):
        placa = self._maior(0)
        self.assertEqual(self.at["tipo"][placa], "lisa")
        self.assertGreater((self.rot[self.origem == 0] == placa).mean(), 0.95)
        self.assertLess((self.rot[self.origem == 1] == placa).mean(), 0.02)

    def test_tufo_rugoso_vira_grupo_cortado_em_blocos(self):
        r = self.rot[self.origem == 2]
        blocos = [x for x in np.unique(r) if x > 0 and self.at["grupo"][x] > 0]
        self.assertGreaterEqual(len(blocos), 4)
        self.assertGreater(np.isin(r, blocos).mean(), 0.8)
        self.assertLess(np.isin(self.rot[self.origem == 1], blocos).mean(), 0.10)  # so quinas encostadas no tufo

    def test_plano_zero_e_a_placa_e_a_tabela_marca_a_regiao_deitada(self):
        self.assertGreater(abs(self.planos[0]["normal"][2]), 0.99)
        placa = self._maior(0)
        lin = next(x for x in self.linhas if x["id"] == placa)
        self.assertGreater(lin["fracao_no_plano0"], 0.95)
        self.assertTrue(any("plano 0" in t for t in lin["indicios"]))

    def test_precedencia_dos_rotulos(self):
        placa, topo = self._maior(0), self._maior(1)
        g = int(self.at["grupo"][self._maior(2)])
        self.assertGreater(g, 0)
        info = dict(planos=self.planos)
        dec = dict(padrao="duvida", regioes={str(placa): "apoio", str(topo): "peca"}, grupos={str(g): "residuo"})
        cls = self.gpu.para_cpu(A.classes_por_voxel(self.at, info, dec))
        self.assertTrue((cls[self.rot == placa] == A.COD["apoio"]).all())
        self.assertTrue((cls[self.rot == topo] == A.COD["peca"]).all())
        self.assertGreater((cls[self.origem == 2] == A.COD["residuo"]).mean(), 0.8)
        self.assertTrue((cls[self.rot == 0] == A.COD["ruido"]).all())
        # o corte pelo plano 0 vence a regiao, menos nas protegidas
        dec["regioes"][str(placa)] = "peca"
        dec["cortes"] = [dict(plano=0, abaixo_mm=1.0, classe="apoio")]
        cls = self.gpu.para_cpu(A.classes_por_voxel(self.at, info, dec))
        self.assertTrue((cls[self.rot == placa] == A.COD["apoio"]).all())
        dec["proteger"] = [placa]
        cls = self.gpu.para_cpu(A.classes_por_voxel(self.at, info, dec))
        self.assertTrue((cls[self.rot == placa] == A.COD["peca"]).all())
        with self.assertRaises(ValueError):
            A.classes_por_voxel(self.at, info, dict(regioes={"1": "mesa"}))

    def test_ilhas_deixa_so_o_maior_componente(self):
        topo = self._maior(1)
        dec = dict(padrao="peca", ilhas="ruido")
        cls = self.gpu.para_cpu(A.classes_por_voxel(self.at, dict(planos=self.planos), dec))
        self.assertTrue((cls[self.rot == topo] == A.COD["peca"]).all())  # tudo e um componente so: nada vira ilha
        dec = dict(
            padrao="ruido", regioes={str(topo): "peca"}, caixas=[dict(min=[0, 0, -1], max=[5, 5, 1], classe="peca")]
        )
        cls = self.gpu.para_cpu(A.classes_por_voxel(self.at, dict(planos=self.planos), dec))
        canto = (self.gpu.para_cpu(self.at["C"])[:, :2] < 5).all(1) & (self.rot > 0)
        self.assertTrue((cls[canto] == A.COD["peca"]).all())
        dec["ilhas"] = "ruido"
        cls = self.gpu.para_cpu(A.classes_por_voxel(self.at, dict(planos=self.planos), dec))
        self.assertTrue((cls[canto] == A.COD["ruido"]).all())  # canto da placa, menor e solto do topo: ilha
        self.assertTrue((cls[self.rot == topo] == A.COD["peca"]).all())

    def test_render_mostra_o_topo_na_frente_da_placa(self):
        rs = A.rasterizar(self.at["C"], [0, 0, -1.0], [0, 1.0, 0], self.voxel, tam=400)
        win = self.gpu.para_cpu(rs["win"])
        z = self.gpu.para_cpu(self.at["C"])[:, 2]
        centro = win[win.shape[0] // 2, win.shape[1] // 2]
        self.assertGreaterEqual(centro, 0)
        self.assertAlmostEqual(z[centro], 8.0, places=5)
        img = A.sombrear(rs, self.at["N"], self.voxel)
        self.assertEqual(img.shape, win.shape + (3,))
        _, escritos = A.anotar(img, rs, "teste", self.voxel, [("7", 200, 200, (1, 0, 0))])
        self.assertEqual(escritos, ["7"])


if __name__ == "__main__":
    unittest.main()
