"""Receita E da GUI (gui/receitas.py): qual etapa do fluxo_uniao ela monta em cada estado da pasta de trabalho."""

import os
import tempfile
import types
import unittest
from pathlib import Path

from gui import receitas


class ReceitaETest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        raiz = Path(self.tmp.name)
        self.trabalho = raiz / "trabalho"
        self.trabalho.mkdir()
        self.raiz = raiz

    def projeto(self, nome, n_scans):
        p = self.raiz / nome
        for i in range(n_scans):
            d = p / "data" / f"2026101{i}_000000" / "cache"
            d.mkdir(parents=True)
            (d / "frames.dataset").write_bytes(b"x")
        d = p / "data" / "Frame20261010_000000" / "cache"  # quadro avulso do Revo: nao e scan
        d.mkdir(parents=True)
        (d / "frames.dataset").write_bytes(b"x")
        return p

    def app(self, projeto, scans=()):
        ctx = dict(projeto=str(projeto), trabalho=str(self.trabalho), saida=str(self.trabalho), scans=list(scans))
        return types.SimpleNamespace(estado=types.SimpleNamespace(ctx=ctx, env=dict(python="python")))

    def job(self, projeto, scans=()):
        jobs = receitas.receita_e(self.app(projeto, scans))
        self.assertEqual(len(jobs), 1)
        return jobs[0]["nome"], jobs[0]["argv"]

    def test_tres_scans_roda_direto_ate_o_revo_com_grupo_curto(self):
        p = self.projeto("Project20261108_092049", 3)
        nome, argv = self.job(p)
        self.assertTrue(nome.startswith("E ·"))
        self.assertTrue(argv[2].endswith("fluxo_uniao.py"))
        self.assertNotIn("--poses", argv)  # sem arquivo de poses o fluxo alinha sozinho
        self.assertNotIn("--so-rotulos", argv)
        self.assertIn("--revo-malha", argv)
        grupo = argv[argv.index("--revo-grupo") + 1]
        self.assertLessEqual(len(grupo), 30)  # o Revo corta o nome do grupo em 30 caracteres
        self.assertEqual(argv[argv.index("--nome") + 1], "Project20261108_092049_unido")

    def test_usa_o_arquivo_de_poses_quando_existe_e_os_scans_marcados(self):
        p = self.projeto("ProjA", 4)
        poses = self.trabalho / "ProjA_poses.json"
        poses.write_text("{}")
        _, argv = self.job(p, scans=["20261010_000000", "20261011_000000", "20261012_000000"])
        self.assertEqual(argv[argv.index("--poses") + 1], str(poses))
        i = argv.index("--scans")
        self.assertEqual(argv[i + 1 : i + 4], ["20261010_000000", "20261011_000000", "20261012_000000"])

    def test_dois_scans_para_na_revisao_e_depois_continua(self):
        p = self.projeto("ProjB", 2)
        nome, argv = self.job(p)
        self.assertTrue(nome.startswith("E1"))
        self.assertIn("--so-rotulos", argv)
        self.assertNotIn("--revo-grupo", argv)  # a 1a fase nao toca no Revo
        saida = Path(argv[argv.index("--saida") + 1])
        (saida / "atlas").mkdir(parents=True)
        (saida / "rotulos.json").write_text("{}")
        nome, argv = self.job(p)  # 2o clique, depois da revisao
        self.assertTrue(nome.startswith("E2"))
        self.assertIn("--continuar", argv)
        self.assertIn("--revo-grupo", argv)
        (saida / "limpo").mkdir()
        (saida / "ciclo" / "unido").mkdir(parents=True)
        (saida / "ciclo" / "unido" / "ProjB_unido.revo").write_text("{}")
        nome, argv = self.job(p)  # o Revo falhou ou foi interrompido: so essa etapa
        self.assertTrue(nome.startswith("E3"))
        self.assertIn("--so-revo", argv)
        (saida / "conferencia").mkdir()
        with self.assertRaises(ValueError):  # pasta completa: nada e sobrescrito
            receitas.receita_e(self.app(p))

    def test_recusa_um_scan_e_projeto_vazio(self):
        with self.assertRaises(ValueError):
            receitas.receita_e(self.app(self.projeto("ProjC", 1)))
        with self.assertRaises(ValueError):
            receitas.receita_e(self.app(""))

    def test_pasta_interrompida_antes_do_scan_unido_e_recusada(self):
        p = self.projeto("ProjD", 3)
        saida = self.trabalho / "ProjD_fluxo"
        (saida / "limpo").mkdir(parents=True)
        with self.assertRaises(ValueError):
            receitas.receita_e(self.app(p))
        self.assertTrue(os.path.isdir(saida))


class FluxoAjudantesTest(unittest.TestCase):
    def test_ponto_da_fusao_acompanha_a_escala(self):
        import fluxo_uniao as FL

        self.assertEqual(FL.ponto_pela_escala(1.0), 0.1)  # Projeto 9
        self.assertEqual(FL.ponto_pela_escala(3.9), 0.3)  # moldura de painel
        self.assertGreaterEqual(FL.ponto_pela_escala(2.0), 0.1)

    def test_grupo_do_revo_longo_e_recusado_antes_de_comecar(self):
        import fluxo_uniao as FL

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SystemExit) as e:
            FL.main([tmp, "--saida", os.path.join(tmp, "x"), "--revo-grupo", "g" * (FL.GRUPO_MAX + 1)])
        self.assertIn("30", str(e.exception))


if __name__ == "__main__":
    unittest.main()
