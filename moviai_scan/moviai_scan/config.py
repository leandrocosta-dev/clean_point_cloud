"""Configuracao da pipeline.

Todos os parametros geometricos sao MULTIPLICADORES do espacamento medio
entre pontos (calculado automaticamente para cada nuvem). Assim o mesmo
conjunto de parametros funciona para peca pequena (resolucao alta) e peca
grande (resolucao baixa), sem ajuste manual.
"""
from dataclasses import dataclass, field, asdict, fields
import json


@dataclass
class Params:
    # Base / mesa giratoria (RANSAC de plano)
    base_dist_mult: float = 2.5      # tolerancia do plano = mult * espacamento
    base_margin_mult: float = 1.5    # faixa extra removida acima do plano
    # Ruido fino (Statistical Outlier Removal)
    sor_nb: int = 20                 # vizinhos considerados
    sor_std: float = 2.0             # desvios padrao; menor = mais agressivo
    # Ilhas flutuantes (DBSCAN)
    dbscan_eps_mult: float = 3.0
    dbscan_min_points: int = 8
    # Cluster secundario com pelo menos esta fracao do principal vira
    # candidato a trecho deslocado (perda de rastreamento)
    fragment_min_frac: float = 0.02


@dataclass
class Gates:
    """Criterios objetivos de aprovacao. Ajuste com base no seu historico."""
    max_noise_removed_frac: float = 0.15   # % da peca removida como ruido
    max_isolated_frac: float = 0.002       # pontos soltos que sobraram
    max_base_residual_frac: float = 0.003  # restos de base
    max_removed_dense_frac: float = 0.03   # superficie real removida por engano


@dataclass
class Config:
    base_mode: str = "auto"          # auto | on | off
    work_max_points: int = 400_000   # acima disso trabalha numa copia reduzida
    align_fragments: bool = True
    align_min_fitness: float = 0.10  # fracao do trecho que encosta na peca apos alinhar
    align_max_rmse_mult: float = 1.5
    dedupe_mult: float = 0.7         # remove pontos duplicados na sobreposicao
    optimize_trials: int = 30
    seed: int = 42
    params: Params = field(default_factory=Params)
    gates: Gates = field(default_factory=Gates)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = Config()
        for f_ in fields(Config):
            if f_.name in raw and f_.name not in ("params", "gates"):
                setattr(cfg, f_.name, raw[f_.name])
        for k, v in raw.get("params", {}).items():
            setattr(cfg.params, k, v)
        for k, v in raw.get("gates", {}).items():
            setattr(cfg.gates, k, v)
        return cfg
