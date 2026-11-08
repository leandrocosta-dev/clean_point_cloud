"""Orcamento de recursos unico para todos os scripts (prioridade do processo e teto de threads).

    from recursos import baixa_prioridade, orcamento_threads
    jobs, por_job = orcamento_threads(jobs=4)    # define OMP/OPENBLAS/MKL/NUMEXPR/REVO_THREADS

Modo pela variavel REVO_RECURSOS:
  maximo (padrao desde 2026-11-08, pedido do usuario): prioridade ALTA e todos os processadores logicos;
          limites menores passados pelos scripts (teto=4, teto=6...) sao ignorados.
  conservador: regra antiga do guia (secao 8) - BELOW_NORMAL e no maximo 12 threads somando os processos,
          para a instabilidade registrada (WHEA, segfaults do Open3D sob carga sustentada).
REVO_TETO_THREADS=N fixa o teto em qualquer modo. A GPU nao entra nesse teto.
`baixa_prioridade()` mantem o nome antigo (muitos scripts importam) e aplica a prioridade do modo.
Execute este arquivo para ver o orcamento atual:  python pipeline/recursos.py
"""

from __future__ import annotations

import os

MODO = os.environ.get("REVO_RECURSOS", "maximo").strip().lower()
CPUS = os.cpu_count() or 1
try:
    TETO_THREADS = max(1, int(os.environ["REVO_TETO_THREADS"]))
except (KeyError, ValueError):
    TETO_THREADS = 12 if MODO == "conservador" else CPUS
VARS_THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "REVO_THREADS",
)


def aplicar_prioridade(pid: int | None = None) -> None:
    """Prioridade do modo no processo `pid` (atual por padrao): ALTA (maximo) ou abaixo do normal."""
    try:
        import psutil
    except ImportError:
        return
    try:
        p = psutil.Process(pid)
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if MODO == "conservador" else psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(10 if MODO == "conservador" else -5)
    except (psutil.Error, OSError, ValueError):
        pass


def baixa_prioridade() -> None:
    """Nome historico: aplica a prioridade do modo atual (ALTA no modo maximo)."""
    aplicar_prioridade()


def orcamento_threads(jobs: int = 1, teto: int | None = None, aplicar: bool = True) -> tuple[int, int]:
    """Divide o teto entre `jobs` processos. Devolve (jobs efetivos, threads por processo).

    jobs <= 0 escolhe automaticamente (ate 4 processos). No modo maximo o `teto` do chamador so vale se
    for MAIOR (os scripts antigos pediam 3-6 threads por causa da instabilidade).
    """
    if teto is None or MODO != "conservador":
        teto = TETO_THREADS
    teto = max(1, min(int(teto), TETO_THREADS, CPUS))
    jobs = int(jobs) if jobs and jobs > 0 else min(4, teto)
    jobs = max(1, min(jobs, teto))
    por_job = max(1, teto // jobs)
    if aplicar:
        for v in VARS_THREADS:
            os.environ[v] = str(por_job)
    return jobs, por_job


def threads_atuais(padrao: int | None = None) -> int:
    """Threads que este processo pode usar (respeita o orcamento ja aplicado)."""
    padrao = TETO_THREADS if padrao is None else padrao
    try:
        n = int(os.environ.get("REVO_THREADS", padrao))
    except ValueError:
        n = padrao
    return max(1, min(n, TETO_THREADS))


if __name__ == "__main__":
    j, t = orcamento_threads(0, aplicar=False)
    print(f"modo {MODO}; teto {TETO_THREADS} threads; cpus {CPUS}; automatico: {j} processos x {t} threads")
