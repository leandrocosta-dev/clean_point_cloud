"""Instalacao e contrato GPU testados sem depender de driver ou inicializar CUDA."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


class InfraestruturaTest(unittest.TestCase):
    def rodar_isolado(self, codigo: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("REVO_PERMITIR_CPU", None)
        with tempfile.TemporaryDirectory() as tmp:
            return subprocess.run(
                [sys.executable, "-I", "-c", codigo],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

    def test_importacao_editavel_sem_pythonpath_e_fora_do_repositorio(self):
        codigo = (
            "import importlib.util,json,comum,recursos,revo_projeto,gpu; "
            "gui=importlib.util.find_spec('gui'); "
            "assert gui is not None, 'gui nao instalado'; "
            "print(json.dumps({'comum':comum.__file__,'recursos':recursos.__file__,"
            "'revo_projeto':revo_projeto.__file__,'gpu':gpu.__file__,"
            "'gui':list(gui.submodule_search_locations)}))"
        )
        r = self.rodar_isolado(codigo)
        self.assertEqual(r.returncode, 0, r.stderr)
        dados = json.loads(r.stdout)
        for nome in ("comum", "recursos", "gpu"):
            self.assertEqual(Path(dados[nome]).resolve(), RAIZ / "pipeline" / f"{nome}.py")
        self.assertEqual(Path(dados["revo_projeto"]).resolve(), RAIZ / "pipeline/revo_frames/revo_projeto.py")
        self.assertIn(RAIZ / "gui", [Path(p).resolve() for p in dados["gui"]])

    def test_gpu_ausente_falha_em_vez_de_habilitar_cpu(self):
        codigo = """
import sys
sys.modules['cupy'] = None
import gpu
try:
    gpu.xp()
except gpu.ErroGPU:
    print('GPU obrigatoria preservada')
else:
    raise AssertionError('Sem CuPy, producao nao pode executar na CPU')
"""
        r = self.rodar_isolado(codigo)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("GPU obrigatoria preservada", r.stdout)


if __name__ == "__main__":
    unittest.main()
