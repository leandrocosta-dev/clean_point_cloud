"""GUI isolada: catalogo, subcomandos reais, formularios e fila sem executar scans.

    python -m unittest tests.test_gui_engenharia -v

REVO_GUI_EVIDENCIAS permite salvar capturas Qt em uma pasta nova para revisao.
"""

import argparse
import importlib.util
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui import catalog


class CatalogoTest(unittest.TestCase):
    def test_ignora_saidas_geradas(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            (raiz / "output").mkdir()
            (raiz / "output/nao_executar.py").write_text("raise RuntimeError('teste')")
            (raiz / "gpu.py").write_text('"""biblioteca"""')
            (raiz / "ferramenta.py").write_text('"""ferramenta"""')
            with patch.object(catalog, "PIPE", tmp):
                itens = catalog.carregar()
            self.assertEqual([i["id"] for i in itens], ["ferramenta.py"])


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "GUI requer PySide6")
class InterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication

        cls.qt = QApplication.instance() or QApplication([])
        # O plugin offscreen do Windows nao enumera as fontes instaladas.
        fonte = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
        if fonte.is_file():
            QFontDatabase.addApplicationFont(str(fonte))
        cls.qt.setStyle("Fusion")
        cls.qt.setFont(QFont("Segoe UI", 10))

    def setUp(self):
        from gui import core, paginas, runner
        from gui.app import Janela
        from gui.assistente import PaginaAssistente

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        raiz = Path(self.tmp.name)
        for mod, nome, arquivo in (
            (core, "ARQ_ESTADO", "estado.json"),
            (core, "PRESETS_DIR", "presets"),
            (paginas, "PRESETS_DIR", "presets"),
            (runner, "ARQ_HISTORICO", "historico.json"),
        ):
            p = patch.object(mod, nome, str(raiz / arquivo))
            p.start()
            self.addCleanup(p.stop)
        with patch.object(PaginaAssistente, "_checar_cli"):
            self.janela = Janela()
        from gui import theme

        self.qt.setStyleSheet(theme.qss(self.janela.pal))
        self.addCleanup(self._fechar)
        self.itens = {i["id"]: i for i in self.janela.itens}

    def _fechar(self):
        self.janela._timer_salvar.stop()
        self.janela.close()
        self.qt.processEvents()

    def _form(self, script):
        from gui.widgets import FormScript

        forma = FormScript(self.itens[script], self.janela)
        self.addCleanup(forma.deleteLater)
        return forma

    def _validar_parser_real(self, script, argv):
        class ArgumentosValidados(Exception):
            pass

        original = argparse.ArgumentParser.parse_args
        lidos = []

        def capturar(parser, args=None, namespace=None):
            lidos.append(original(parser, argv, namespace))
            raise ArgumentosValidados

        with patch.object(argparse.ArgumentParser, "parse_args", capturar), self.assertRaises(ArgumentosValidados):
            runpy.run_path(self.itens[script]["caminho"], run_name="__main__")
        self.assertEqual(len(lidos), 1)
        return lidos[0]

    def test_nove_paginas_e_prioridade(self):
        self.assertEqual(self.janela.pilha.count(), 9)
        for i in range(9):
            self.janela.ir(i)
            self.qt.processEvents()
            self.assertEqual(self.janela.pilha.currentIndex(), i)
        self.assertEqual(self.janela.p_amb.prio.currentText(), "alta")
        self.assertEqual(self.janela.p_amb.omp.value(), 0)

    def test_formularios_principais_e_cuda(self):
        for script in (
            "triagem.py",
            "consenso_visibilidade.py",
            "regressao.py",
            "fundir_nitido.py",
            "rodar_ciclo_precisao.py",
        ):
            with self.subTest(script=script):
                forma = self._form(script)
                self.assertTrue(forma.linhas)
                if "motor" in forma.linhas:
                    editor = forma.linhas["motor"].editor
                    self.assertEqual([editor.itemText(i) for i in range(editor.count())], ["cuda"])

    def test_metricas_subcomandos_geram_cli_valida(self):
        forma = self._form("metricas.py")
        for comando in ("nuvem", "malha", "consenso"):
            forma.comando.setCurrentText(comando)
            forma.aplicar_valores({"arquivo": "peca.ply", "nuvens": '"vista um.ply" "vista dois.ply"'})
            argv = forma.argv()
            self.assertEqual(argv[0], comando)
            namespace = self._validar_parser_real("metricas.py", argv)
            self.assertEqual(namespace.cmd, comando)
            if comando == "consenso":
                self.assertEqual(namespace.nuvens, ["vista um.ply", "vista dois.ply"])

    def test_consulta3d_subcomandos_e_presets(self):
        forma = self._form("consulta3d.py")
        for comando in catalog.SUBCOMANDOS["consulta3d.py"]:
            forma.comando.setCurrentText(comando)
            forma.aplicar_valores(
                {
                    "arquivos": '"vista um.ply" "vista dois.ply"',
                    "antes": "antes.ply",
                    "depois": "depois.ply",
                    "min": "0 0 0",
                    "max": "1 1 1",
                    "centro": "0 0 0",
                    "raio": 1,
                    "ponto": "0 0 0",
                    "normal": "0 0 1",
                    "saida": "vista.html",
                    "registro": "registro.json",
                }
            )
            namespace = self._validar_parser_real("consulta3d.py", forma.argv())
            self.assertEqual(namespace.cmd, comando)
        salvo = forma.valores()
        copia = self._form("consulta3d.py")
        copia.aplicar(salvo)
        self.assertEqual(forma.argv(), copia.argv())
        forma.comando.setCurrentText("caixa")
        self.assertIn("--min", forma.argv())
        forma.comando.setCurrentText("info")
        self.assertNotIn("--min", forma.argv())

    def test_runner_inicia_script_fora_do_repositorio(self):
        from PySide6.QtCore import QEventLoop, QTimer

        self.janela.estado.env["cwd"] = self.tmp.name
        self.janela.estado.env["tentativas_crash"] = 0
        job = {
            "nome": "Triagem --help",
            "argv": [self.janela.estado.env["python"], self.itens["triagem.py"]["caminho"], "--help"],
        }
        loop = QEventLoop()
        resultados = []
        linhas = []
        self.janela.runner.fila_concluida.connect(lambda ok: (resultados.append(ok), loop.quit()))
        self.janela.runner.linha.connect(lambda texto, tipo: linhas.append(texto))
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(30000)
        self.janela.runner.executar([job])
        loop.exec()
        timer.stop()
        if self.janela.runner.rodando:
            self.janela.runner.parar()
        self.assertEqual(resultados, [True], "\n".join(linhas))
        self.assertTrue(any("usage:" in linha.lower() for linha in linhas))

    def test_entrada_gui_importa_em_processo_isolado(self):
        entrada = str(Path(catalog.RAIZ) / "gui/app.py")
        r = subprocess.run(
            [sys.executable, "-I", "-c", "import runpy, sys; runpy.run_path(sys.argv[1])", entrada],
            cwd=self.tmp.name,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_capturas_opcionais(self):
        destino = os.environ.get("REVO_GUI_EVIDENCIAS")
        if not destino:
            return
        raiz = Path(destino)
        raiz.mkdir(parents=True, exist_ok=True)
        self.janela.show()
        self.janela.ir(8)
        self.qt.processEvents()
        self.assertTrue(self.janela.grab().save(str(raiz / "ambiente.png")))
        for script in (
            "metricas.py",
            "consulta3d.py",
            "triagem.py",
            "consenso_visibilidade.py",
            "regressao.py",
            "fundir_nitido.py",
        ):
            self.janela.ir(4)
            self.janela.p_scripts.abrir(script)
            forma = self.janela.p_scripts.paginas[script].form
            self.qt.processEvents()
            self.assertTrue(self.janela.grab().save(str(raiz / (script[:-3] + ".png"))))
            if "motor" in forma.linhas:
                for secao, _ in forma.secoes:
                    secao.alternar(True)
                self.qt.processEvents()
                self.assertTrue(forma.linhas["motor"].grab().save(str(raiz / "motor_cuda.png")))


if __name__ == "__main__":
    unittest.main()
