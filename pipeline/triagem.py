"""Triagem automatica de um projeto Revo: mede cada scan e recomenda a receita (sem alterar nada).

    python pipeline/triagem.py <pasta_do_projeto> [--scans g1 g2] [--saida perfil.json]

Por scan (lendo raw_preview.ply e FrameInfo, sem fusao): planos candidatos (RANSAC sequencial, area ocupada),
apoio = plano de maior AREA (a face plana da peca nao engana, armadilha do Projeto 9), componente principal
fora do plano (diagonal, folga ate o plano = peca apoiada ou suspensa), saltos de pose (perda de rastreio),
quadros inconsistentes, modos de calibracao e se ha fuse.ply.
Depois classifica o caso com as regras validadas do GUIA (secao 2 e Projetos 3-9) e grava `perfil.json` com as
opcoes de `limpar_raw`/`limpar_por_placa` e os proximos passos. `rodar_tudo.py --auto` usa essas opcoes.
Limites: nao sabe se a peca foi virada (--lados), nem se um scan e da face interna (precisa de alinhamento);
recomendacao e ponto de partida auditavel, nao aprovacao.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np
from comum import configurar_log

AQUI = os.path.dirname(os.path.abspath(__file__))
import gpu  # noqa: E402
from recursos import baixa_prioridade  # noqa: E402
from revo_projeto import RevoScan, ler_ply  # noqa: E402

PEQUENA_MM = 150.0  # diagonal abaixo disso = peca pequena (Projetos 3, 4, 7, 8, 9: 45-70 mm)
AREA_PLACA_MM2 = 3000.0  # plano de apoio visivel (placa 4.000-17.000 mm2 no P9; face da peca ~1.000)


_LOGGER = configurar_log(__name__)


def _g(a, dtype=None):
    return gpu.para_gpu(a, dtype)


def _n_unicos_2d(cel):
    """Numero de linhas distintas de `cel` (n, 2) int64 (cupy)."""
    if len(cel) == 0:
        return 0
    gpu.xp()
    c = cel - cel.min(0)
    return int(len(gpu.agrupar(c[:, 0] * (int(c[:, 1].max()) + 1) + c[:, 1])[0]))


def _voxel_unico(P, v):
    """Um ponto (o primeiro, na ordem original) por voxel de `v` mm; ordem original preservada. GPU."""
    m = gpu.xp()
    P = _g(P, np.float64)
    k = m.floor(P / v).astype(m.int64)
    k = k - k.min(0)
    dims = k.max(0) + 1
    chave = (k[:, 0] * int(dims[1]) + k[:, 1]) * int(dims[2]) + k[:, 2]
    _, i = m.unique(chave, return_index=True)
    return P[m.sort(i)]


def planos(
    P: Any, n_planos: int = 4, thr: float = 1.0, it: int = 600, celula: float = 5.0, seed: int = 0
) -> list[dict[str, Any]]:
    """RANSAC sequencial (gpu.ransac_plano: hipoteses em paralelo na GPU); area = celulas de `celula` mm ocupadas pela
    projecao dos inliers no plano. P numpy ou cupy."""
    m = gpu.xp()
    P = _g(P, np.float64)
    resto = m.ones(len(P), bool)
    out = []
    for i in range(n_planos):
        idx = m.flatnonzero(resto)
        if len(idx) < 500:
            break
        try:
            n, d, _ = gpu.ransac_plano(P[idx], limiar=thr, iteracoes=it, semente=seed + i)
        except ValueError:  # todas as hipoteses degeneradas
            break
        mi = m.abs(P[idx] @ _g(n) + d) < thr
        Q = P[idx[mi]]
        u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        cel = m.floor(m.stack([Q @ _g(u), Q @ _g(v)], axis=1) / celula).astype(m.int64)
        ninl = int(mi.sum())
        out.append(
            dict(
                normal=n.tolist(),
                d=float(d),
                inliers=ninl,
                fracao=float(ninl / len(P)),
                area_mm2=float(_n_unicos_2d(cel) * celula * celula),
            )
        )
        resto[idx[mi]] = False
    return out


def foco_das_cameras(T: Any, tem_prof: Any) -> np.ndarray | None:
    """Ponto mais proximo (minimos quadrados) de todos os eixos de visada: e para onde o operador apontou
    o scanner, ou seja, a peca. A camera olha para -z do proprio referencial (convencao de RevoScan.pontos).
    Soma dos projetores (I - d d^T) em float64 vetorizada; o sistema 3x3 final e resolvido na CPU."""
    ks = np.flatnonzero(tem_prof)
    if len(ks) < 3:
        return None
    m = gpu.xp()
    Tg = _g(np.asarray(T)[ks], np.float64)
    c = Tg[:, :3, 3]
    d = -Tg[:, :3, 2]
    d = d / m.maximum(m.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    M = m.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = gpu.para_cpu(M.sum(0))
    b = gpu.para_cpu(m.einsum("kij,kj->i", M, c))
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def componente_principal(P: Any, voxel: float = 2.0, foco: Any = None) -> tuple[Any, float, int]:
    """Componente 26-conexo da peca em grade de `voxel` mm: o mais perto do foco das cameras (se dado),
    senao o maior. Devolve (mascara, diagonal p1-p99, n_componentes). GPU (rotulagem esparsa de gpu.py);
    P numpy ou cupy, mascara do mesmo tipo."""
    na_gpu = type(P).__module__.split(".")[0] == "cupy"
    m = gpu.xp()
    P = _g(P, np.float64)
    if len(P) < 10:
        return (P[:0, 0] > 0 if na_gpu else np.zeros(len(P), bool)), 0.0, 0
    dims = gpu.para_cpu((P.max(0) - P.min(0)) / voxel).astype(np.int64) + 1
    if np.prod(dims.astype(float)) > 2e8:  # cena enorme: grade mais grossa
        return componente_principal(P if na_gpu else gpu.para_cpu(P), voxel * 2, foco)
    rot, tam = gpu.componentes_conexos_voxel(P, voxel)  # rotulo 0 = maior; origem da grade = minimo
    n = len(tam)
    escolha = 0
    if foco is not None:
        d = m.linalg.norm(P - _g(foco), axis=1)
        grandes = m.flatnonzero(tam >= max(50, 0.01 * len(P)))  # ignora poeira
        if len(grandes):
            dmin = m.full(n, m.inf)
            m.minimum.at(dmin, rot, d)
            escolha = int(grandes[m.argmin(dmin[grandes])])
    mk = rot == escolha
    Q = P[mk]
    diag = (
        float(np.linalg.norm(gpu.para_cpu(m.percentile(Q, 99, axis=0) - m.percentile(Q, 1, axis=0))))
        if len(Q) > 10
        else 0.0
    )
    return (mk if na_gpu else gpu.para_cpu(mk)), diag, int(n)


def saltos_de_pose(T: Any, tem_prof: Any, fator: float = 10.0, minimo_mm: float = 5.0) -> tuple[list[int], float]:
    """Quadros (indices entre os que tem profundidade) cujo deslocamento da camera para o anterior e
    > max(fator x mediana, minimo_mm): candidato a perda de rastreio (Projeto 9, scan 680250ac)."""
    ks = np.flatnonzero(tem_prof)
    if len(ks) < 3:
        return [], 0.0
    c = T[ks, :3, 3]
    passo = np.linalg.norm(np.diff(c, axis=0), axis=1)
    med = float(np.median(passo))
    lim = max(fator * med, minimo_mm)
    return [int(ks[i + 1]) for i in np.flatnonzero(passo > lim)], med


def medir_scan(pasta: str | os.PathLike[str], amostra: int = 250000) -> dict[str, Any]:
    g = os.path.basename(os.path.normpath(pasta))
    r = dict(
        scan=g,
        tem_fuse=os.path.isfile(os.path.join(pasta, "fuse.ply")),
        tem_fuse_mesh=os.path.isfile(os.path.join(pasta, "fuse_mesh.ply")),
    )
    if not (
        os.path.isfile(os.path.join(pasta, "raw_preview.ply"))
        and os.path.isfile(os.path.join(pasta, "cache", "frames.dataset"))
    ):
        r["tipo"] = "sem_quadros"
        return r
    r["tipo"] = "quadros"
    with tempfile.TemporaryDirectory() as tmp:
        sc = RevoScan(pasta, tmp)
        try:
            r["quadros"] = len(sc.ids)
            r["quadros_com_profundidade"] = int(sc.tem_prof.sum())
            r["quadros_inconsistentes"] = len(sc.inconsistentes)
            r["modos"] = sorted(set(int(m) for m in sc.modo[sc.tem_prof]))
            r["saltos_pose"], r["passo_camera_mediano_mm"] = saltos_de_pose(sc.T, sc.tem_prof)
            foco = foco_das_cameras(sc.T, sc.tem_prof)
            r["foco_cameras"] = None if foco is None else foco.round(2).tolist()
        finally:
            sc.c.close()
    xp = gpu.xp()
    _, prev = ler_ply(os.path.join(pasta, "raw_preview.ply"))
    P = prev[:, :3].astype(float)
    P = P[np.all(np.isfinite(P), axis=1)]
    r["pontos_preview"] = int(len(P))
    if len(P) > amostra:
        P = P[np.random.default_rng(0).choice(len(P), amostra, replace=False)]
    P = _voxel_unico(P, 0.75)  # daqui em diante tudo em cupy
    r["extensao_cena_mm"] = gpu.para_cpu(P.max(0) - P.min(0)).round(1).tolist() if len(P) else [0, 0, 0]
    pl = planos(P) if len(P) > 1000 else []
    r["planos"] = pl
    apoio = max(pl, key=lambda x: x["area_mm2"]) if pl else None
    r["placa"] = bool(apoio and apoio["area_mm2"] >= AREA_PLACA_MM2 and apoio["fracao"] >= 0.08)
    if r["placa"]:
        n, d = np.array(apoio["normal"]), apoio["d"]
        h = P @ _g(n) + d
        if float((h > 3).mean()) < float((h < -3).mean()):  # normal para o lado dos objetos
            n, d, h = -n, -d, -h
        # placa nunca e perfeitamente plana: 3 mm (ou 3x a dispersao dos inliers) separa objeto de placa
        perto = xp.abs(h[xp.abs(h) < 1.0])
        disp = float(xp.median(perto)) * 1.4826 if len(perto) else 0.3
        lim = max(3.0, 3 * disp)
        sel = h > lim
        acima = P[sel]
        mk, diag, ncomp = componente_principal(acima, foco=foco)
        hp = h[sel][mk]
        r["apoio"] = dict(
            normal=n.tolist(), d=float(d), area_mm2=apoio["area_mm2"], fracao=apoio["fracao"], limiar_mm=lim
        )
        # folga = percentil 5 (suportes finos encostados na placa nao baixam a medida)
        r["peca"] = dict(
            diagonal_mm=diag,
            componentes=ncomp,
            folga_ao_apoio_mm=float(xp.percentile(hp, 5)) if len(hp) else None,
            altura_mm=float(xp.percentile(hp, 99)) if len(hp) else None,
            fracao_pontos=float(mk.sum() / len(P)),
            # suspensa: quase nada da peca perto da placa e a maior parte acima de 16 mm (Projeto 5)
            fracao_ate_8mm=float((hp < 8.0).mean()) if len(hp) else None,
            fracao_acima_16mm=float((hp > 16.0).mean()) if len(hp) else None,
        )
        outros = sorted((x["area_mm2"] for x in pl if x is not apoio), reverse=True)
        r["face_plana_grande"] = bool(outros and outros[0] >= 0.25 * apoio["area_mm2"])
    else:
        mk, diag, ncomp = componente_principal(P, foco=foco)
        r["peca"] = dict(
            diagonal_mm=diag,
            componentes=ncomp,
            folga_ao_apoio_mm=None,
            altura_mm=None,
            fracao_pontos=float(mk.sum() / max(1, len(P))),
        )
        r["face_plana_grande"] = False
    return r


def _suspensa(s):
    p = s["peca"]
    if p.get("fracao_ate_8mm") is None:
        return False
    return p["fracao_ate_8mm"] < 0.10 and p["fracao_acima_16mm"] > 0.70


def recomendar(scans: Sequence[dict[str, Any]]) -> dict[str, Any]:
    validos = [s for s in scans if s.get("tipo") == "quadros"]
    if not validos:
        return dict(caso="sem scans de quadros", limpar_raw={}, passos=["nada a limpar no nivel de quadros"], avisos=[])
    diag = float(np.median([s["peca"]["diagonal_mm"] for s in validos]))
    com_placa = [s for s in validos if s["placa"]]
    pequena = diag < PEQUENA_MM
    n = len(validos)
    avisos, passos = [], []
    lr = {}
    placa = None
    if pequena:
        lr = dict(ext_max_mm=round(max(100.0, 1.5 * diag)), so_peca=True, min_mantido=0.02)
        if com_placa:
            suspensa = sum(_suspensa(s) for s in com_placa) * 2 > len(com_placa)
            placa = dict(altura_min=16.0 if suspensa else 1.0, voxel=1.0, min_pts_voxel=3)
            if any(s["face_plana_grande"] for s in com_placa):
                placa["plano"] = "extenso"
            sem = [s["scan"] for s in validos if not s["placa"]]
            if sem:
                avisos.append(
                    f"scans sem placa visivel {sem}: NAO aplicar limpar_por_placa neles (qualquer plano corta a peca)"
                )
            placa["scans"] = [s["scan"] for s in com_placa]
            caso = f"peca pequena ({diag:.0f} mm) {'suspensa' if suspensa else 'apoiada'} sobre placa"
            if not suspensa:
                avisos.append(
                    "peca apoiada: o corte de 1 mm junto ao apoio e removido de proposito (avisar na entrega)"
                )
        else:
            caso = f"peca pequena ({diag:.0f} mm) sem placa dominante"
    else:
        if com_placa and sum(_suspensa(s) for s in com_placa) * 2 <= len(com_placa):
            lr = dict(periferia_mm=5.0, periferia_banda=12.0, periferia_alargue=10.0)
            caso = f"peca grande ({diag:.0f} mm) apoiada na mesa"
        elif com_placa:
            placa = dict(altura_min=16.0, voxel=6.0, min_pts_voxel=3, scans=[s["scan"] for s in com_placa])
            caso = f"peca grande ({diag:.0f} mm) suspensa sobre placa (receita do Projeto 5)"
        else:
            caso = f"peca grande ({diag:.0f} mm), cena sem plano dominante"
    passos.append(
        "limpar_raw " + " ".join(f"--{k.replace('_', '-')}" + ("" if v is True else f" {v}") for k, v in lr.items())
    )
    if placa:
        passos.append(
            "limpar_por_placa "
            + " ".join(
                f"--{k.replace('_', '-')} " + (" ".join(v) if isinstance(v, list) else str(v)) for k, v in placa.items()
            )
        )
    if n == 1:
        passos.append("receita A: revo_pipeline (ou fusao_superficies/fundir_nitido para peca pequena)")
    elif n == 2:
        passos.append("receita B: rodar_tudo --unir A B --lados ? (so quem escaneou sabe se a peca foi virada)")
    elif pequena:
        passos += [
            "alinhar: poses do Merge Alignment do Revo (0,05 mm) ou alinhar_varios --escala auto",
            "consenso_visibilidade (contradicao por espaco livre entre scans) e conferencia 3D",
            "fundir_nitido --motor cuda; limpar_artefatos",
        ]
    else:
        passos.append(
            "receita C: alinhar_varios -> alinhar_segmentos -> alinhar_direto (3 etapas) -> consenso -> fusao"
        )
        passos.append("conferir orientacao externa/interna por scan (construcao_por_cobertura imprime a concordancia)")
    for s in validos:
        if s["saltos_pose"]:
            avisos.append(
                f"{s['scan']}: salto de pose nos quadros {s['saltos_pose'][:6]} (possivel perda de rastreio; "
                "conferir e remover o trecho)"
            )
        if s["quadros_inconsistentes"]:
            avisos.append(f"{s['scan']}: {s['quadros_inconsistentes']} quadros inconsistentes (ficam sem limpeza)")
        if len(s["modos"]) > 1:
            avisos.append(f"{s['scan']}: {len(s['modos'])} modos de calibracao no mesmo scan")
    modos = {tuple(s["modos"]) for s in validos}
    if len(modos) > 1:
        avisos.append(f"scans com modos de calibracao diferentes {sorted(modos)}: separar no alinhamento")
    sem_fuse = [s["scan"] for s in validos if not s["tem_fuse"]]
    if sem_fuse and n > 1:
        avisos.append(f"scans sem fuse.ply {sem_fuse}: consenso contra fusoes do Revo exige funde-los antes")
    return dict(
        caso=caso, diagonal_peca_mm=diag, scans=n, limpar_raw=lr, limpar_por_placa=placa, passos=passos, avisos=avisos
    )


def triar(proj: str | os.PathLike[str], scans: Sequence[str] | None = None) -> dict[str, Any]:
    proj = os.path.abspath(proj)
    revo = [f for f in os.listdir(proj) if f.endswith(".revo")]
    if len(revo) != 1:
        raise ValueError("esperado exatamente um .revo na pasta do projeto")
    with open(os.path.join(proj, revo[0]), encoding="utf-8") as f:
        nos = [n["guid"] for n in json.load(f)["nodes"]]
    sel = [g for g in nos if (not scans or g in scans) and os.path.isdir(os.path.join(proj, "data", g))]
    medidas = [medir_scan(os.path.join(proj, "data", g)) for g in sel]
    return dict(projeto=proj, scans=medidas, recomendacao=recomendar(medidas))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--scans", nargs="+")
    ap.add_argument("--saida", help="arquivo JSON do perfil (padrao: so imprime)")
    a = ap.parse_args()
    baixa_prioridade()
    perfil = triar(a.projeto, a.scans)
    for s in perfil["scans"]:
        if s.get("tipo") != "quadros":
            _LOGGER.info("%s", f"{s['scan']}: sem quadros")
            continue
        p = s["peca"]
        _LOGGER.info(
            "%s",
            f"{s['scan']}: {s['quadros_com_profundidade']} quadros | placa {('sim' if s['placa'] else 'nao')}"
            + (f" ({s['apoio']['area_mm2']:.0f} mm2)" if s["placa"] else "")
            + f" | peca {p['diagonal_mm']:.0f} mm, folga {(p['folga_ao_apoio_mm'] if p['folga_ao_apoio_mm'] is None else round(p['folga_ao_apoio_mm'], 1))} mm"
            + f" | face plana grande {s['face_plana_grande']} | saltos {len(s['saltos_pose'])} | fuse {s['tem_fuse']}",
        )
    r = perfil["recomendacao"]
    _LOGGER.info("%s", f"\nCASO: {r['caso']}")
    for p in r["passos"]:
        _LOGGER.info("%s %s", "  -", p)
    for w in r["avisos"]:
        _LOGGER.info("%s %s", "  AVISO:", w)
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as f:
            json.dump(perfil, f, indent=1, default=float)
        _LOGGER.info("%s %s", "perfil gravado em", a.saida)


if __name__ == "__main__":
    main()
