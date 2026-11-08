"""Consultas 3D (pipeline/consulta3d.py): selecoes, corte, comparacao, origem quadro/pixel, registro de inspecao."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import consulta3d as C
import numpy as np
import open3d as o3d

from tests.gpu_utils import CUDA_DISPONIVEL


def gravar(caminho, P):
    o3d.io.write_point_cloud(caminho, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float))))


def rodar(*args):
    with contextlib.redirect_stdout(io.StringIO()):
        return C.main(list(args))


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class Consulta3DTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        g = np.arange(0, 40, 0.5)
        x, y = np.meshgrid(g, g)
        self.A = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        self.B = np.vstack([self.A[self.A[:, 0] < 20], self.A[self.A[:, 0] >= 20] + [0, 0, 5.0]])
        self.a, self.b = os.path.join(d, "a.ply"), os.path.join(d, "b.ply")
        gravar(self.a, self.A)
        gravar(self.b, self.B)

    def tearDown(self):
        self.tmp.cleanup()

    def test_info_e_caixa(self):
        r = rodar("info", self.a)
        self.assertEqual(r[0]["pontos"], len(self.A))
        self.assertAlmostEqual(r[0]["espacamento_mediano_mm"], 0.5, places=5)
        r = rodar("caixa", self.a, "--min", "0", "0", "-1", "--max", "9.9", "9.9", "1", "--amostra", "3")
        self.assertEqual(r["fontes"][0]["pontos"], 20 * 20)
        self.assertEqual(len(r["fontes"][0]["amostra"]), 3)

    def test_esfera_e_registro_de_cobertura(self):
        reg = os.path.join(self.tmp.name, "insp.json")
        r = rodar("esfera", self.a, "--centro", "5", "5", "0", "--raio", "2", "--registro", reg, "--regiao", "10")
        self.assertGreater(r["fontes"][0]["pontos"], 0)
        self.assertEqual(r["celulas_novas_inspecionadas"], 1)
        c = rodar("cobertura_inspecao", self.a, "--registro", reg, "--regiao", "10")
        self.assertEqual(c["celulas_com_dado"], 16)
        self.assertEqual(c["celulas_inspecionadas"], 1)
        self.assertEqual(c["faltam"], 15)
        rodar("caixa", self.a, "--min", "-1", "-1", "-1", "--max", "41", "41", "1", "--registro", reg, "--regiao", "10")
        self.assertEqual(rodar("cobertura_inspecao", self.a, "--registro", reg, "--regiao", "10")["faltam"], 0)
        with self.assertRaises(ValueError):  # registro nao mistura tamanhos de regiao
            rodar("cobertura_inspecao", self.a, "--registro", reg, "--regiao", "30")

    def test_corte_png(self):
        png = os.path.join(self.tmp.name, "corte.png")
        r = rodar(
            "corte",
            self.a,
            self.b,
            "--ponto",
            "10",
            "0",
            "0",
            "--normal",
            "1",
            "0",
            "0",
            "--espessura",
            "0.6",
            "--png",
            png,
        )
        self.assertTrue(os.path.getsize(png) > 1000)
        self.assertEqual(r["fontes"][0]["pontos"], 80)  # uma coluna x=10
        self.assertEqual(r["fontes"][1]["pontos"], 80)

    def test_comparar(self):
        r = rodar("comparar", self.a, self.b, "--tol", "0.1", "--regiao", "30")
        metade = int((self.A[:, 0] >= 20).sum())
        self.assertEqual(r["sumiram"], metade)
        self.assertEqual(r["apareceram"], metade)
        self.assertEqual(sum(x["pontos"] for x in r["regioes_sumiram"]), metade)

    def test_componentes_e_vista(self):
        r = rodar("componentes", self.b, "--voxel", "0.6")
        self.assertEqual(r[0]["total"], 2)
        html = os.path.join(self.tmp.name, "v.html")
        v = rodar("vista", self.a, self.b, "--saida", html, "--max-pontos", "1000")
        self.assertLessEqual(v["pontos_exibidos"], 1002)
        self.assertIn("Plotly.newPlot", Path(html).read_text(encoding="utf-8"))

    def test_origem_observacoes(self):
        pasta = os.path.join(self.tmp.name, "scanX", "origem")
        os.makedirs(pasta)
        dt = np.dtype([("p", "<f4", 3), ("v", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")])
        obs = np.zeros(len(self.A), dt)
        obs["p"] = self.A
        obs["quadro"] = (self.A[:, 0] // 10).astype(int)
        obs["pixel"] = np.arange(len(self.A))
        arq = os.path.join(pasta, "observacoes.npy")
        np.save(arq, obs)
        r = rodar("caixa", arq, "--min", "0", "0", "-1", "--max", "9.9", "40", "1")
        f = r["fontes"][0]
        self.assertEqual(f["arquivo"], "scanX")
        self.assertEqual(f["quadros_mais_frequentes"][0]["quadro"], 0)
        self.assertEqual(f["quadros_mais_frequentes"][0]["pontos"], 20 * 80)
        self.assertIn("pixel", f["origem_exemplos"][0])
        json.dumps(r)  # saida e JSON serializavel


if __name__ == "__main__":
    unittest.main()
