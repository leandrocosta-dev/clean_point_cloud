"""Executa somente contratos de infraestrutura/utilitarios, sem fallback numerico para CPU.

    python scripts/testes_sem_gpu.py

A suite completa, com CUDA, continua obrigatoria antes de homologar processamento.
Qualquer skip desta selecao e erro: uma instalacao ausente nao pode dar CI verde.
"""

import sys
import unittest
from pathlib import Path


def main() -> int:
    raiz = Path(__file__).resolve().parents[1]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for arquivo in ("test_comum.py", "test_infra_engenharia.py", "test_gpu_utils.py"):
        suite.addTests(loader.discover(str(raiz / "tests"), pattern=arquivo))
    resultado = unittest.TextTestRunner(verbosity=2).run(suite)
    if resultado.skipped:
        sys.stderr.write("ERRO: testes desta selecao sem GPU foram pulados; conferir instalacao.\n")
    return int(not resultado.wasSuccessful() or bool(resultado.skipped) or resultado.testsRun == 0)


if __name__ == "__main__":
    raise SystemExit(main())
