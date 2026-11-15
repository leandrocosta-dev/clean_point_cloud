"""Confere a FUSAO FINAL do scan unido (e a malha) contra as superficies de cada scan: fecha o ciclo no resultado real.

    python pipeline/conferir_uniao.py <ciclo> --fusao "Fused ... .ply" [--malha "Mesh ... .ply"] --saida PASTA

<ciclo> = pasta gravada por `ciclo_uniao.py --unir` (ou a pasta do `fluxo_uniao.py`, que tem `ciclo/` dentro). A
fusao e a que o Revo Scan gravou para o scan unido (`<biblioteca>/pointclouds/Fused *.ply`, referencial do scan base).
Mede, sem alterar nada:
  apoio       quantos scans tem superficie propria a <= --raio de cada ponto fundido (0 = ponto que nenhum scan
              sustenta; 1 = so um scan viu, ninguem confirma);
  soltos      pontos fora do maior componente conexo (migalha, ilha de suporte);
  zonas fracas manchas com apoio <= 1: onde falta outro scan (centro e area de cada uma);
  malha       triangulos, componentes, furos (MeshLib), vertices longe da nuvem (superficie inventada) e pontos da
              nuvem longe de qualquer vertice (superficie perdida).
Saida: conferencia.json e imagens (cinza = 3 ou mais scans, amarelo = 2, laranja = 1, vermelho = nenhum).
Codigo 2 = algum limite foi passado (--max-sem-apoio-pct, --max-soltos-pct, --max-malha-*). Referencia aprovada pelo
usuario (Projeto 9, 2026-11-15): 0,37% sem apoio a 0,3 mm, 3,3% com um scan so, 1 componente.
"""

import argparse
import json
import os
import sys
import time

import atlas_partes as AP
import ciclo_uniao as C
import gpu
import numpy as np
from comum import configurar_log, salvar_json_atomico
from metricas import ler_nuvem, metricas_malha

_LOG = configurar_log(__name__)

CORES = np.array([[0.90, 0.12, 0.10], [0.98, 0.50, 0.10], [0.95, 0.85, 0.25], [0.80, 0.80, 0.80]])


def espacamento(P, n=20000, seed=0):
    """Mediana da distancia ao vizinho mais proximo (mm) numa amostra."""
    m = gpu.xp()
    idx = m.asarray(np.random.default_rng(seed).choice(len(P), min(n, len(P)), replace=False))
    d, _ = gpu.knn(P, P[idx], 2)
    return float(m.median(d[:, 1]))


def apoio_por_scan(P, sup, raio):
    """Numero de scans com ponto proprio a <= raio de cada P."""
    m = gpu.xp()
    n = m.zeros(len(P), m.int32)
    for x in sup.values():
        if len(x["P"]):
            n += (gpu.vizinho_mais_proximo(x["P"], P, dist_max=raio)[1] >= 0).astype(m.int32)
    return n


def conferir_nuvem(P, sup, raio, ponto, zona_min=50, top=12):
    m = gpu.xp()
    n = apoio_por_scan(P, sup, raio)
    comp, tam = gpu.componentes_conexos_voxel(P, max(raio, 2.0 * ponto), 26)
    solto = comp > 0
    fraco = n <= 1
    zonas = []
    if bool(fraco.any()):
        iz = m.flatnonzero(fraco)
        cz, tz = gpu.componentes_conexos_voxel(P[iz], max(raio, 2.0 * ponto), 26)
        tz_cpu = gpu.para_cpu(tz)
        for k in np.flatnonzero(tz_cpu >= zona_min)[:top]:  # componentes vem do maior para o menor
            sel = iz[cz == int(k)]
            zonas.append(
                dict(
                    pontos=int(tz_cpu[k]),
                    area_mm2=round(float(tz_cpu[k]) * ponto * ponto, 2),
                    centro=[round(float(v), 2) for v in gpu.para_cpu(P[sel].mean(0))],
                    sem_apoio_pct=round(100 * float((n[sel] == 0).mean()), 1),
                )
            )
        n_zonas = int((tz_cpu >= zona_min).sum())
    else:
        n_zonas = 0
    rel = dict(
        pontos=int(len(P)),
        espacamento_mm=round(ponto, 4),
        raio_apoio_mm=raio,
        sem_apoio_pct=round(100 * float((n == 0).mean()), 3),
        um_scan_pct=round(100 * float((n == 1).mean()), 3),
        dois_scans_pct=round(100 * float((n == 2).mean()), 3),
        tres_ou_mais_pct=round(100 * float((n >= 3).mean()), 3),
        componentes=int(len(tam)),
        soltos_pontos=int(solto.sum()),
        soltos_pct=round(100 * float(solto.mean()), 4),
        zonas_fracas=n_zonas,
        maiores_zonas_fracas=zonas,
    )
    return rel, n, solto


def conferir_malha(caminho, P, ponto, fator=3.0):
    """Contagens da malha (MeshLib) e desvio vertice <-> nuvem fundida (mesmo referencial)."""
    import open3d as o3d

    m = gpu.xp()
    rel = dict(arquivo=os.path.abspath(caminho), **metricas_malha(caminho))
    if rel.get("triangulos"):
        rel["triangulos_fora_do_maior"] = int(rel["triangulos"] - rel.get("maior_componente", 0))
    V = gpu.para_gpu(np.asarray(o3d.io.read_triangle_mesh(caminho).vertices), m.float64)
    if not len(V):
        rel["erro"] = "malha sem vertices"
        return rel
    lim = fator * ponto
    dv, _ = gpu.vizinho_mais_proximo(P, V)
    dp, _ = gpu.vizinho_mais_proximo(V, P)
    rel.update(
        limite_mm=round(lim, 4),
        vertice_a_nuvem_p50_mm=round(float(m.percentile(dv, 50)), 4),
        vertice_a_nuvem_p90_mm=round(float(m.percentile(dv, 90)), 4),
        vertice_a_nuvem_max_mm=round(float(dv.max()), 4),
        inventada_pct=round(100 * float((dv > lim).mean()), 3),
        perdida_pct=round(100 * float((dp > lim).mean()), 3),
    )
    return rel


def desenhar(P, N, n, solto, pasta, ponto, titulo):
    m = gpu.xp()
    cor = gpu.para_gpu(CORES)[m.minimum(n, 3)]
    cor = m.where(solto[:, None], gpu.para_gpu(np.array([0.85, 0.20, 0.85])), cor)
    S = P[:: max(1, len(P) // 100_000)]
    _, V = np.linalg.eigh(gpu.para_cpu(m.cov((S - S.mean(0)).T)))
    os.makedirs(pasta, exist_ok=True)
    arqs = []
    passo = max(ponto, 0.1)
    for nome, d in (("frente", V[:, 0]), ("verso", -V[:, 0])):
        rs = AP.rasterizar(P, d, V[:, 2], passo, 1400)
        arq = os.path.join(pasta, f"apoio_{nome}.png")
        AP.anotar(AP.sombrear(rs, N, passo, cor), rs, f"{titulo} apoio {nome}", passo)[0].save(arq)
        arqs.append(arq)
    return arqs


def superficies_do_ciclo(ciclo, min_obs=None):
    """Fusao propria de cada scan aceito (referencial comum), poses e scan base do unido."""
    with open(os.path.join(ciclo, "relatorio.json"), encoding="utf-8") as fh:
        rel = json.load(fh)
    par = rel["parametros"]
    with open(par["poses"], encoding="utf-8") as fh:
        poses = {k: np.asarray(v, float) for k, v in json.load(fh).items()}
    scans = list(rel["scans"])
    a = argparse.Namespace(voxel=par["voxel"], min_obs=min_obs or par["min_obs"], orientados=set())
    sup = C.fundir_todos(os.path.join(ciclo, "projeto"), scans, poses, a)
    base = (rel.get("unido") or {}).get("base") or min(scans, key=lambda s: float(np.abs(poses[s] - np.eye(4)).sum()))
    return sup, poses, base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ciclo", help="pasta do ciclo_uniao (com relatorio.json e projeto/) ou do fluxo_uniao")
    ap.add_argument("--fusao", required=True, help="PLY da fusao do scan unido (referencial do scan base)")
    ap.add_argument("--malha", help="PLY da malha do scan unido (mesmo referencial da fusao)")
    ap.add_argument("--saida", required=True, help="pasta nova")
    ap.add_argument("--raio", type=float, default=0.3, help="mm para um scan contar como apoio de um ponto fundido")
    ap.add_argument(
        "--min-obs",
        type=int,
        help="observacoes por voxel para contar como superficie do scan (padrao: o do ciclo, 4); usar 1 em scan com "
        "poucos quadros, onde a fusao do Revo tem pontos que a fusao propria com 4 observacoes nao tem",
    )
    ap.add_argument("--max-sem-apoio-pct", type=float, default=1.5)
    ap.add_argument("--max-soltos-pct", type=float, default=0.05)
    ap.add_argument("--max-malha-componentes", type=int, default=1)
    ap.add_argument("--max-malha-inventada-pct", type=float, default=1.0)
    ap.add_argument("--max-malha-perdida-pct", type=float, default=1.0)
    a = ap.parse_args(argv)
    m = gpu.xp()
    ciclo, saida = os.path.abspath(a.ciclo), os.path.abspath(a.saida)
    if not os.path.exists(os.path.join(ciclo, "relatorio.json")) and os.path.exists(os.path.join(ciclo, "ciclo")):
        ciclo = os.path.join(ciclo, "ciclo")
    if os.path.exists(saida) and os.listdir(saida):
        sys.exit(f"destino ja existe: {saida}")
    t0 = time.time()
    sup, poses, base = superficies_do_ciclo(ciclo, a.min_obs)
    P, N = ler_nuvem(a.fusao, normais=True)
    P = gpu.para_gpu(P, m.float64)
    N = gpu.para_gpu(N, m.float64) if N is not None and len(N) == len(P) else gpu.normais_pca(P, k=20)
    Tb = poses[base]
    Rb, tb = gpu.para_gpu(Tb[:3, :3].T, m.float64), gpu.para_gpu(Tb[:3, 3], m.float64)
    Pr, Nr = P @ Rb + tb, N @ Rb  # no referencial comum, onde estao as superficies por scan
    ponto = espacamento(Pr)
    nuvem, n, solto = conferir_nuvem(Pr, sup, a.raio, ponto)
    falhas = []
    if nuvem["sem_apoio_pct"] > a.max_sem_apoio_pct:
        falhas.append(f"{nuvem['sem_apoio_pct']}% dos pontos fundidos sem apoio de nenhum scan")
    if nuvem["soltos_pct"] > a.max_soltos_pct:
        falhas.append(f"{nuvem['soltos_pontos']} pontos soltos do corpo da peca")
    imagens = desenhar(Pr, Nr, n, solto, os.path.join(saida, "imagens"), ponto, os.path.basename(a.fusao)[:40])
    malha = None
    if a.malha:
        malha = conferir_malha(a.malha, P, ponto)
        if malha.get("componentes", 0) > a.max_malha_componentes:
            falhas.append(
                f"malha com {malha['componentes']} componentes ({malha.get('triangulos_fora_do_maior')} triangulos soltos)"
            )
        if malha.get("inventada_pct", 0) > a.max_malha_inventada_pct:
            falhas.append(f"{malha['inventada_pct']}% dos vertices da malha longe da nuvem (superficie inventada)")
        if malha.get("perdida_pct", 0) > a.max_malha_perdida_pct:
            falhas.append(f"{malha['perdida_pct']}% da nuvem sem malha por perto")
        if "erro" in malha:
            falhas.append(f"malha nao medida: {malha['erro']}")
    rel = dict(
        ciclo=ciclo,
        fusao=os.path.abspath(a.fusao),
        scans=sorted(sup),
        base=base,
        nuvem=nuvem,
        malha=malha,
        imagens=imagens,
        limites={k: v for k, v in vars(a).items() if k.startswith("max_")},
        falhas=falhas,
        aprovado=not falhas,
        tempo_s=round(time.time() - t0, 1),
    )
    salvar_json_atomico(os.path.join(saida, "conferencia.json"), rel)
    _LOG.info(
        "fusao final: %s pontos | sem apoio %.2f%% | um scan %.2f%% | >= 3 scans %.1f%% | soltos %d | zonas fracas %d",
        f"{nuvem['pontos']:,}",
        nuvem["sem_apoio_pct"],
        nuvem["um_scan_pct"],
        nuvem["tres_ou_mais_pct"],
        nuvem["soltos_pontos"],
        nuvem["zonas_fracas"],
    )
    if malha:
        _LOG.info("malha: %s", {k: v for k, v in malha.items() if k != "arquivo"})
    _LOG.info("gravado: %s (%.0f s)%s", saida, time.time() - t0, " | REPROVADO: " + "; ".join(falhas) if falhas else "")
    return 2 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
