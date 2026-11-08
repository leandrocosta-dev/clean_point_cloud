r"""Instalacao editavel: expoe a raiz, `pipeline/` e `pipeline/revo_frames/` por arquivo .pth.

    .\.venv\Scripts\python.exe -m pip install -e .

Depois disso `import recursos`, `import gpu`, `import revo_projeto` e `import gui` funcionam de qualquer pasta, e os scripts
continuam rodando como `python pipeline/x.py`. Os metadados ficam em pyproject.toml; este arquivo so troca a estrategia
de instalacao editavel (o padrao do setuptools nao cobre as pastas de modulos soltos). Instalacao nao editavel nao e
suportada: o codigo e usado direto do repositorio.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.editable_wheel import _StaticPth, editable_wheel

RAIZ = Path(__file__).resolve().parent
PASTAS = [RAIZ, RAIZ / "pipeline", RAIZ / "pipeline" / "revo_frames"]


class EditavelDuasPastas(editable_wheel):
    def _select_strategy(self, name, tag, build_lib):
        return _StaticPth(self.distribution, name, PASTAS)


setup(cmdclass={"editable_wheel": EditavelDuasPastas})
