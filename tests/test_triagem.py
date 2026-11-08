"""Triagem: foco das cameras, saltos de pose, planos, componente da peca e regras de recomendacao."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import triagem as TR

from tests.gpu_utils import CUDA_DISPONIVEL
from tests.test_precisao import criar_scan


def camera_olhando(c, alvo):
    z = np.asarray(c, float) - alvo
    z /= np.linalg.norm(z)  # camera olha para -z
    x = np.cross([0, 0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = c
    return T


def perfil(scan, diag=55, placa=True, folga=1.3, ate8=0.3, acima16=0.2, face=False, saltos=(), modos=(0,), fuse=True):
    p = dict(
        diagonal_mm=diag,
        componentes=1,
        folga_ao_apoio_mm=folga if placa else None,
        fracao_ate_8mm=ate8 if placa else None,
        fracao_acima_16mm=acima16 if placa else None,
    )
    return dict(
        scan=scan,
        tipo="quadros",
        placa=placa,
        peca=p,
        face_plana_grande=face,
        saltos_pose=list(saltos),
        quadros_inconsistentes=0,
        modos=list(modos),
        tem_fuse=fuse,
    )


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class GeometriaTest(unittest.TestCase):
    def test_foco_das_cameras(self):
        alvo = np.array([10.0, 20.0, 30.0])
        ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        T = np.array([camera_olhando(alvo + [300 * np.cos(a), 300 * np.sin(a), 150], alvo) for a in ang])
        np.testing.assert_allclose(TR.foco_das_cameras(T, np.ones(len(T), bool)), alvo, atol=1e-6)
        self.assertIsNone(TR.foco_das_cameras(T[:2], np.ones(2, bool)))
        self.assertIsNone(TR.foco_das_cameras(np.repeat(np.eye(4)[None], 5, 0), np.ones(5, bool)))  # eixos paralelos

    def test_saltos_de_pose(self):
        T = np.repeat(np.eye(4)[None], 12, 0)
        T[:, 0, 3] = np.arange(12) * 1.0
        T[6:, 0, 3] += 20
        tem = np.ones(12, bool)
        self.assertEqual(TR.saltos_de_pose(T, tem)[0], [6])
        tem[3] = False  # quadro sem profundidade nao conta
        self.assertEqual(TR.saltos_de_pose(T, tem)[0], [6])
        self.assertEqual(TR.saltos_de_pose(T[:2], tem[:2]), ([], 0.0))

    def test_planos_area_e_componente_pelo_foco(self):
        rng = np.random.default_rng(0)
        g = np.arange(0, 200, 2.0)
        x, y = np.meshgrid(g, g)
        mesa = np.c_[x.ravel(), y.ravel(), rng.normal(0, 0.1, x.size)]
        g2 = np.arange(0, 20, 0.5)
        x2, y2 = np.meshgrid(g2, g2)
        face = np.c_[x2.ravel() + 50, y2.ravel() + 50, np.full(x2.size, 10.0)]
        pl = TR.planos(np.vstack([mesa, face]), n_planos=2)
        self.assertGreater(pl[0]["area_mm2"], 30000)
        self.assertAlmostEqual(abs(pl[0]["normal"][2]), 1.0, places=2)
        self.assertLess(pl[1]["area_mm2"], 0.25 * pl[0]["area_mm2"])
        a = rng.uniform(0, 30, (3000, 3))
        b = rng.uniform(0, 10, (800, 3)) + 200
        m, diag, n = TR.componente_principal(np.vstack([a, b]), foco=np.array([205.0, 205, 205]))
        self.assertEqual(n, 2)
        self.assertEqual(int(m.sum()), 800)
        self.assertLess(diag, 20)
        m2, _, _ = TR.componente_principal(np.vstack([a, b]))
        self.assertEqual(int(m2.sum()), 3000)

    def test_medir_scan_sintetico_nao_quebra(self):
        with tempfile.TemporaryDirectory() as t:
            criar_scan(Path(t) / "scan")
            r = TR.medir_scan(str(Path(t) / "scan"))
            self.assertEqual(r["tipo"], "quadros")
            self.assertEqual(r["quadros_com_profundidade"], 5)
            self.assertIsNone(r["foco_cameras"])  # todas as cameras iguais: sem foco
            self.assertFalse(r["placa"])
            (Path(t) / "vazio").mkdir()
            self.assertEqual(TR.medir_scan(str(Path(t) / "vazio"))["tipo"], "sem_quadros")

    def test_triar_projeto(self):
        with tempfile.TemporaryDirectory() as t:
            criar_scan(Path(t) / "p/data/scan")
            (Path(t) / "p/p.revo").write_text(json.dumps({"nodes": [{"guid": "scan"}]}))
            r = TR.triar(str(Path(t) / "p"))
            self.assertEqual(r["recomendacao"]["scans"], 1)
            self.assertTrue(any("receita A" in p for p in r["recomendacao"]["passos"]))


class RecomendacaoTest(unittest.TestCase):
    def test_suspensa(self):
        self.assertTrue(TR._suspensa(perfil("a", ate8=0.05, acima16=0.9)))
        self.assertFalse(TR._suspensa(perfil("a", ate8=0.3, acima16=0.9)))
        self.assertFalse(TR._suspensa(perfil("a", placa=False)))

    def test_peca_pequena_apoiada_projeto8(self):
        r = TR.recomendar([perfil(f"s{i}") for i in range(4)])
        self.assertIn("pequena", r["caso"])
        self.assertEqual(r["limpar_raw"], dict(ext_max_mm=100, so_peca=True, min_mantido=0.02))
        self.assertEqual(r["limpar_por_placa"]["altura_min"], 1.0)
        self.assertNotIn("plano", r["limpar_por_placa"])
        self.assertTrue(any("apoiada" in w for w in r["avisos"]))

    def test_projeto9_scan_sem_placa_e_face_plana(self):
        scans = [perfil("semplaca", placa=False)] + [perfil(f"s{i}", face=(i == 0)) for i in range(7)]
        r = TR.recomendar(scans)
        self.assertEqual(r["limpar_por_placa"]["plano"], "extenso")
        self.assertNotIn("semplaca", r["limpar_por_placa"]["scans"])
        self.assertTrue(any("semplaca" in w and "NAO aplicar" in w for w in r["avisos"]))
        self.assertTrue(any("consenso_visibilidade" in p for p in r["passos"]))

    def test_pequena_suspensa_e_ext_proporcional(self):
        r = TR.recomendar([perfil("a", diag=104, ate8=0.0, acima16=1.0), perfil("b", diag=104, ate8=0.0, acima16=1.0)])
        self.assertEqual(r["limpar_por_placa"]["altura_min"], 16.0)
        self.assertEqual(r["limpar_raw"]["ext_max_mm"], 156)
        self.assertTrue(any("receita B" in p for p in r["passos"]))

    def test_peca_grande(self):
        r = TR.recomendar([perfil(f"s{i}", diag=1400) for i in range(8)])
        self.assertEqual(r["limpar_raw"], dict(periferia_mm=5.0, periferia_banda=12.0, periferia_alargue=10.0))
        self.assertIsNone(r["limpar_por_placa"])
        self.assertTrue(any("receita C" in p for p in r["passos"]))
        r = TR.recomendar([perfil("s", diag=600, ate8=0.0, acima16=0.95)])
        self.assertEqual(r["limpar_por_placa"]["voxel"], 6.0)
        self.assertIn("suspensa", r["caso"])
        r = TR.recomendar([perfil("s", diag=600, placa=False)])
        self.assertEqual(r["limpar_raw"], {})

    def test_avisos(self):
        r = TR.recomendar([perfil("a", saltos=[40, 41], modos=(0, 1), fuse=False), perfil("b")])
        txt = " ".join(r["avisos"])
        self.assertIn("salto de pose", txt)
        self.assertIn("modos de calibracao", txt)
        self.assertIn("sem fuse.ply", txt)
        self.assertEqual(TR.recomendar([dict(scan="x", tipo="sem_quadros")])["caso"], "sem scans de quadros")


if __name__ == "__main__":
    unittest.main()
