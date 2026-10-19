"""MoviAI Scan: limpeza, realinhamento e correcao de deriva de scans do Revopoint Miraco.

Modulos por frente (o Open3D so e carregado por quem precisa dele):
    pipeline, clean, align, metrics, optimize, render, inspect_llm  nuvem PLY exportada (Open3D)
    malha                                                          malha Poisson (Open3D)
    cloudcompare                                                   limpeza pelo CloudCompare CLI
    revo                                                           projeto do Revo Scan quadro a quadro (Open3D so na limpeza)
"""
from .config import Config, Params, Gates

__all__ = ["Config", "Params", "Gates"]
