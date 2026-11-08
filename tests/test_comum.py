"""Utilitarios comuns: sha256, PLY, JSON atomico, log, destino novo, e importacao das pastas do pipeline."""
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'pipeline'))
import comum  # noqa: E402

CAB = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
       "property float z\nend_header\n")


class ComumTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_sha256_igual_hashlib(self):
        p = self.dir / "a.bin"
        dados = os.urandom(300_000)
        p.write_bytes(dados)
        self.assertEqual(comum.sha256(p, bloco=1000), hashlib.sha256(dados).hexdigest())

    def test_ply_ida_e_volta_com_novo_numero_de_pontos(self):
        pts = np.arange(9, dtype=np.float32).reshape(3, 3)
        p = self.dir / "a.ply"
        comum.gravar_ply(p, CAB, pts)
        cab, lido = comum.ler_ply(p)
        np.testing.assert_array_equal(lido, pts)
        comum.gravar_ply(p, cab, pts[:2])
        cab2, lido2 = comum.ler_ply(p)
        self.assertIn("element vertex 2", cab2)
        np.testing.assert_array_equal(lido2, pts[:2])

    def test_ply_compativel_com_revo_projeto(self):
        sys.path.insert(0, str(RAIZ / 'pipeline' / 'revo_frames'))
        import revo_projeto as R
        pts = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float32)
        a, b = self.dir / "a.ply", self.dir / "b.ply"
        comum.gravar_ply(a, CAB, pts)
        R.gravar_ply(b, CAB, pts)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        np.testing.assert_array_equal(comum.ler_ply(b)[1], R.ler_ply(a)[1])

    def test_json_atomico(self):
        p = self.dir / "sub" / "r.json"
        comum.salvar_json_atomico(p, {"n": np.int64(3), "v": np.arange(3), "p": self.dir, "t": "acentuação"})
        r = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual((r["n"], r["v"], r["t"]), (3, [0, 1, 2], "acentuação"))
        with self.assertRaises(TypeError):
            comum.salvar_json_atomico(p, {"x": object()})
        self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["n"], 3)      # original intacto
        self.assertEqual([f.name for f in p.parent.iterdir()], ["r.json"])        # sem .tmp sobrando

    def test_destino_novo(self):
        novo = comum.destino_novo(self.dir / "a" / "b")
        self.assertTrue(novo.is_dir())
        comum.destino_novo(novo)                                                   # vazia: reaproveita
        (novo / "x.txt").write_text("1")
        with self.assertRaises(SystemExit) as e:
            comum.destino_novo(novo)
        self.assertIn("destino ja existe", str(e.exception))
        with self.assertRaises(SystemExit):
            comum.destino_novo(novo / "x.txt")

    def test_configurar_log_idempotente(self):
        arq = self.dir / "l" / "t.log"
        log = comum.configurar_log("teste_comum", arquivo=arq, nivel=logging.INFO)
        log = comum.configurar_log("teste_comum", arquivo=arq)
        log.info("ola")
        for h in log.handlers:
            h.flush()
        self.assertEqual(len(log.handlers), 2)
        linha = arq.read_text(encoding="utf-8").strip()
        self.assertRegex(linha, r"^\d\d:\d\d:\d\d INFO ola$")
        for h in list(log.handlers):
            log.removeHandler(h)
            h.close()


class InstalacaoEditavelTest(unittest.TestCase):
    """`pip install -e .` torna os modulos de pipeline/ e pipeline/revo_frames/ importaveis de qualquer pasta."""

    def _rodar(self, codigo, cwd):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        return subprocess.run([sys.executable, "-I", "-c", codigo], cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=120)

    def test_importa_de_outra_pasta(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._rodar("import recursos, revo_projeto, comum, vizinhos_gpu; "
                            "print(recursos.__file__); print(revo_projeto.__file__)", d)
        if r.returncode != 0 and "ModuleNotFoundError" in r.stderr:
            self.skipTest("pacote nao instalado: rode  python -m pip install -e .")
        self.assertEqual(r.returncode, 0, r.stderr)
        linhas = r.stdout.split("\n")
        self.assertTrue(Path(linhas[0]).samefile(RAIZ / "pipeline" / "recursos.py"))
        self.assertTrue(Path(linhas[1]).samefile(RAIZ / "pipeline" / "revo_frames" / "revo_projeto.py"))

    def test_script_roda_como_python_pipeline_x_py_de_outra_pasta(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, str(RAIZ / "pipeline" / "recursos.py")], cwd=d,
                               capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
