"""Limpeza do projeto do Revo Scan direto nos quadros, com o mesmo motor da nuvem PLY.

Fluxo por scan:
  1. Calibra cada modo contra o raw_preview.ply (pula o scan se o erro passar de ERRO_MAX_CALIB).
  2. Acha o prato pelas cameras (prato.achar_base). Ponto ate faixa_base acima dele e base.
  3. Funde TODOS os quadros numa grade de voxels (um ponto medio por voxel = nuvem de trabalho),
     com o voxel escolhido para a nuvem ficar perto de work_max_points.
  4. Roda o motor (pipeline.limpar_trabalho: ruido, ilhas, trechos deslocados, busca de parametros).
     A peca e o objeto que mais aparece no centro da imagem (o operador mira a peca).
     Um objeto separado so e candidato a trecho deslocado se a maior parte dos pontos dele vem de
     quadros que quase nao veem a peca: objeto de apoio aparece junto com a peca, trecho com deriva
     vem de quadros cujo rastreamento saiu do lugar.
  5. Cada pixel de cada quadro herda o rotulo do voxel onde cai. Na copia, os pixels removidos sao
     apagados do mapa de profundidade. Poses intactas.
  6. Quadro cuja maioria dos pontos caiu num trecho realinhado entra em <scan>_correcao.json
     (pose nova = T do encaixe @ pose). As poses NAO sao gravadas aqui: conferir as fusoes antes e
     depois (detalhe nao simetrico contra o gabarito) e so entao rodar 'revo aplicar'.
"""
import json
import math
import os
import struct
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

from .. import metrics as M
from ..clean import MAIN, BASE, CLUSTER, DUP, AMBIG, final_points
from ..config import Config
from ..geom import mean_spacing
from ..pipeline import limpar_trabalho, status_final, salvar_sugestoes
from ..render import render_views
from .formato import (RevoScan, ler_ply, gravar_ply, gravar_ply_xyz, gravar_copia_scan, gravar_revo,
                      arquivo_revo, preparar_destino, calibrar_modos, ERRO_MAX_CALIB)
from .prato import amostrar, achar_base

DESLOC = 1 << 20          # indice de voxel deslocado para ficar positivo (21 bits por eixo)
MASC = (1 << 21) - 1
VE_PECA = 0.2             # quadro com menos que isso dos pontos na peca "nao ve a peca"


# ---------------------------------------------------------------- grade de voxels
def _chaves(P, voxel):
    i = np.floor(P / voxel).astype(np.int64) + DESLOC
    if len(i) and (i.min() < 0 or i.max() > MASC):
        raise ValueError("cena grande demais para a grade de voxels")
    return (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]


def _engrossar(ch, m):
    """Chave do voxel m vezes maior que contem cada voxel. Com m inteiro o mapeamento e exato."""
    i = np.c_[ch >> 42, (ch >> 21) & MASC, ch & MASC] - DESLOC
    i = np.floor_divide(i, m) + DESLOC
    return (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]


def _reduzir(ch, soma, n):
    u, inv = np.unique(ch, return_inverse=True)
    inv = inv.ravel()
    s = np.c_[[np.bincount(inv, soma[:, j], len(u)) for j in range(3)]].T
    return u, s, np.bincount(inv, n, len(u))


def _voxel_trabalho(scan, A, max_pts):
    """Voxel para a nuvem de trabalho ficar perto de max_pts.

    Area da cena estimada na amostra com voxel de 3 mm. Nunca menor que o pixel na distancia tipica.
    """
    area = len(np.unique(_chaves(A, 3.0))) * 9.0
    ks = np.flatnonzero(scan.tem_prof)
    zs = [np.median(scan.profundidade(scan.ids[k])[3]) for k in ks[:: max(1, len(ks) // 10)]]
    fx = min(K[0] for K in scan.K.values())
    return max(float(np.median(zs)) / fx, math.sqrt(area / max_pts))


def _acima(P, base, faixa):
    if base is None:
        return np.ones(len(P), bool)
    return P @ base[0] + base[1] > faixa


def _fundir_quadros(scan, base, faixa, voxel, passo_amostra=8, raio_centro=0.15):
    """Uma passada por todos os quadros.

    Retorna (chaves ordenadas, soma, contagem) da grade acima da base e a amostra por quadro
    (chave, indice do quadro, pixel no centro da imagem).
    """
    ch, soma, n = np.empty(0, np.int64), np.empty((0, 3)), np.empty(0)
    pend, npend = [], 0
    am_ch, am_k, am_c = [], [], []
    for k in np.flatnonzero(scan.tem_prof):
        P, pix = scan.pontos(k)
        if P is None:
            continue
        ok = _acima(P, base, faixa)
        P, pix = P[ok], pix[ok]
        c = _chaves(P, voxel)
        pend.append((c, P))
        npend += len(c)
        w = struct.unpack("<I", scan.c.execute("select substr(data, 1, 4) from FrameRGBD where id=?",
                                               (scan.ids[k],)).fetchone()[0])[0]
        fx, fy, cx, cy = scan.K[int(scan.modo[k])]
        v, u = np.divmod(pix[::passo_amostra], w)
        am_ch.append(c[::passo_amostra])
        am_k.append(np.full(len(u), k, np.int32))
        am_c.append((u - cx) ** 2 + (v - cy) ** 2 < (raio_centro * w) ** 2)
        if npend > 5_000_000:
            ch, soma, n = _reduzir(np.concatenate([ch] + [p[0] for p in pend]),
                                   np.vstack([soma] + [p[1] for p in pend]),
                                   np.concatenate([n, np.ones(npend)]))
            pend, npend = [], 0
    ch, soma, n = _reduzir(np.concatenate([ch] + [p[0] for p in pend]),
                           np.vstack([soma] + [p[1] for p in pend]),
                           np.concatenate([n, np.ones(npend)]))
    return ch, soma, n, np.concatenate(am_ch), np.concatenate(am_k), np.concatenate(am_c)


def _rotulos(P, base, faixa, ch_ord, lab_vox, voxel):
    """Rotulo de cada ponto pelo voxel onde cai. Abaixo da faixa da base = BASE."""
    lab = np.full(len(P), BASE)
    ok = _acima(P, base, faixa)
    c = _chaves(P[ok], voxel)
    pos = np.minimum(np.searchsorted(ch_ord, c), len(ch_ord) - 1)
    lab[ok] = np.where(ch_ord[pos] == c, lab_vox[pos], CLUSTER)
    return lab


# ---------------------------------------------------------------- peca e trechos pelos quadros
def _escolha_por_quadros(n_work, am_w, am_k, am_c, estado):
    """Regra de escolha para run_clean usando de que quadro veio cada ponto (ver docstring do modulo)."""
    nq = int(am_k.max()) + 1

    def escolher(idx, cl, sizes):
        cw = np.full(n_work, -1)
        cw[idx] = cl
        c = cw[am_w]
        ok = c >= 0
        k, c, centro = am_k[ok], c[ok], am_c[ok]
        votos = np.zeros(len(sizes), int)
        if centro.any():
            # cada quadro vota no objeto com mais pontos no centro da imagem
            par, cnt = np.unique(np.c_[k[centro], c[centro]], axis=0, return_counts=True)
            par = par[np.lexsort((-cnt, par[:, 0]))]
            primeiro = np.r_[True, par[1:, 0] != par[:-1, 0]]
            votos = np.bincount(par[primeiro, 1], minlength=len(sizes))
        principal = max(range(len(sizes)), key=lambda j: (votos[j], sizes[j]))
        na_peca = np.bincount(k[c == principal], minlength=nq) / np.maximum(np.bincount(k, minlength=nq), 1)
        fora = na_peca[k] < VE_PECA
        frac_fora = np.bincount(c[fora], minlength=len(sizes)) / np.maximum(np.bincount(c, minlength=len(sizes)), 1)
        ordem = np.argsort(sizes)[::-1]
        candidatos = np.array([j for j in ordem if j != principal and frac_fora[j] >= 0.5], int)
        estado.update(votos=votos, principal=principal, frac_fora=frac_fora, sizes=sizes,
                      validos=int(len(np.unique(k[centro]))))
        return principal, candidatos

    return escolher


def _log_escolha(e, P, log):
    p = e["principal"]
    log(f"   peca: objeto com {e['sizes'][p]:,} pts, no centro de {e['votos'][p]} de {e['validos']} quadros")
    if e["validos"] and e["votos"][p] < 0.5 * e["validos"]:
        log("   AVISO: a peca esta no centro em menos da metade dos quadros; conferir a imagem")
    for j in np.argsort(e["sizes"])[::-1][:6]:
        if j == p or e["sizes"][j] < P.fragment_min_frac * e["sizes"][p]:
            continue
        tipo = "candidato a trecho deslocado" if e["frac_fora"][j] >= 0.5 else "objeto de apoio (removido)"
        log(f"   objeto {e['sizes'][j]:,} pts | {e['frac_fora'][j]*100:.0f}% vindo de quadros sem a peca "
            f"-> {tipo}")


def _rotulo_por_quadro(lab_am, am_k, trechos):
    """Rotulo dominante de cada quadro: MAIN, trecho realinhado (k) ou AMBIG."""
    rot = {}
    ini = np.flatnonzero(np.r_[True, am_k[1:] != am_k[:-1]])
    for a, b in zip(ini, np.r_[ini[1:], len(am_k)]):
        l = lab_am[a:b]
        peca = int((l == MAIN).sum())
        dup = int((l == DUP).sum())
        r, melhor = MAIN, peca
        for j in trechos:
            nj = int((l == j).sum()) + dup
            if nj > melhor:
                r, melhor = j, nj
        if int((l == AMBIG).sum()) > melhor:
            r = AMBIG
        rot[int(am_k[a])] = r
    return rot


def _faixas(nums):
    """[1, 2, 3, 7] -> '1-3, 7'"""
    nums = sorted(int(x) for x in nums)
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


# ---------------------------------------------------------------- gravacao
def _gravar_copia(scan, base, faixa, ch, lab_vox, voxel, rot, work, spacing, destino_scan, log):
    """Copia do scan com os pixels removidos apagados. Quadro de trecho mantem a peca e o trecho dele."""
    tot = {"antes": 0, "depois": 0}
    vazios = []

    def apagar_pixels(cn):
        for k in np.flatnonzero(scan.tem_prof):
            i = scan.ids[k]
            w, h, mask, z, quatro = scan.profundidade(i)
            P, pix = scan.pontos(k)
            lab = _rotulos(P, base, faixa, ch, lab_vox, voxel)
            r = rot.get(int(k), MAIN)
            ok = (lab == MAIN) | ((lab == r) | (lab == DUP) if r >= 1 else False)
            tot["antes"] += len(pix)
            if ok.sum() == 0:
                # quadro sem nenhum ponto da peca: fica como estava
                vazios.append(int(k))
                tot["depois"] += len(pix)
                continue
            nova = np.zeros(w * h, bool)
            nova[pix[ok]] = True
            dados = (struct.pack("<III", w, h, int(ok.sum()))
                     + np.packbits(nova, bitorder="little").tobytes() + quatro
                     + z[ok].astype("<f4").tobytes())
            cn.execute("update FrameRGBD set data=? where id=?", (dados, i))
            tot["depois"] += int(ok.sum())

    integ = gravar_copia_scan(scan, destino_scan, apagar_pixels)

    # raw_preview filtrado pela mesma regra (esta no referencial do mundo). Ponto do preview nao coincide
    # com o dos quadros (cai em voxel vizinho vazio): usa o rotulo do ponto de trabalho mais proximo.
    hdr, prev = ler_ply(os.path.join(scan.pasta, "raw_preview.ply"))
    Q = prev[:, :3].astype(float)
    dist, viz = cKDTree(work).query(Q, workers=-1)
    okp = _acima(Q, base, faixa) & (dist < 3 * spacing) & (lab_vox[viz] == MAIN)
    gravar_ply(os.path.join(destino_scan, "raw_preview.ply"), hdr, prev[okp])
    antes, depois = tot["antes"], tot["depois"]
    log(f"   pixels: {antes:,} -> {depois:,} ({depois/max(1,antes)*100:.0f}% mantidos) | "
        f"preview {len(prev):,} -> {okp.sum():,} | integridade: {integ}")
    if vazios:
        rv = scan.numero_revo()
        log(f"   AVISO: {len(vazios)} quadros sem nenhum ponto da peca ficaram como estavam "
            f"(Revo {_faixas(rv[vazios])})")
    return dict(pixels_antes=antes, pixels_depois=depois, quadros_vazios=vazios, integridade=integ)


def limpar_scan(scan, cfg, faixa, saida, log=print):
    """Limpa um scan (calibracao ja feita) e grava a copia em saida/data/<scan>. Retorna o relatorio."""
    nome = scan.nome
    A, cams, _ = amostrar(scan)
    b = achar_base(A, cams, log=log)
    base = (np.asarray(b[0], float), float(b[1])) if b is not None else None
    A = A[_acima(A, base, faixa)]
    voxel = _voxel_trabalho(scan, A, cfg.work_max_points)
    ch, soma, n, am_ch, am_k, am_c = _fundir_quadros(scan, base, faixa, voxel)
    if len(ch) > 1.3 * cfg.work_max_points:
        m = math.ceil(math.sqrt(len(ch) / cfg.work_max_points))
        ch, soma, n = _reduzir(_engrossar(ch, m), soma, n)
        am_ch, voxel = _engrossar(am_ch, m), voxel * m
    work = soma / n[:, None]
    am_w = np.searchsorted(ch, am_ch)
    spacing = mean_spacing(work, seed=cfg.seed)
    raw_counts = M.neighbor_counts(work, 2.5 * spacing)
    log(f"[1] {int(n.sum()):,} pontos acima da base -> trabalho {len(work):,} (voxel {voxel:.3f} mm, "
        f"espacamento {spacing:.3f} mm)")

    base_fixa = None
    if base is not None:
        base_fixa = {"n": base[0].tolist(), "d": base[1], "band": faixa, "origem": "prato pelas cameras"}
    estado, rel = {}, {}
    labels, info, T, m, P = limpar_trabalho(work, spacing, raw_counts, cfg, rel, log, base_fixa=base_fixa,
                                            escolher=_escolha_por_quadros(len(work), am_w, am_k, am_c, estado))
    if estado:
        _log_escolha(estado, P, log)
    status, alertas = status_final(m, info)

    rot = _rotulo_por_quadro(labels[am_w], am_k, list(T))
    rv = scan.numero_revo()
    grupos = []
    for j, Tj in T.items():
        qs = sorted(k for k, r in rot.items() if r == j)
        if not qs:
            alertas.append(f"trecho {j} realinhado, mas nenhum quadro tem a maioria dos pontos nele: sem correcao")
            continue
        grupos.append({"trecho": j, "T": np.asarray(Tj).tolist(), "quadros": qs, "revo": rv[qs].tolist()})
        log(f"   trecho {j}: {len(qs)} quadros com deriva (Revo {_faixas(rv[qs])}) -> correcao proposta")
    ambiguos = sorted(k for k, r in rot.items() if r == AMBIG)
    if ambiguos:
        alertas.append(f"quadros com deriva sem encaixe confiavel (Revo {_faixas(rv[ambiguos])}): "
                       "pixels do trecho apagados; conferir no Revo Scan e excluir os quadros")
    for a in alertas:
        log(f"    ALERTA: {a}")

    r = _gravar_copia(scan, base, faixa, ch, labels, voxel, rot, work, spacing,
                      os.path.join(saida, "data", nome), log)

    render_views(work, labels, T, os.path.join(saida, f"{nome}_vistas.png"), title=f"{nome}: limpeza")
    salvar_sugestoes(work, info.pop("sugestoes", []), saida, nome)
    if grupos:
        cor = {"scan": nome, "tipo": "quadros", "n_quadros": len(scan.ids), "grupos": grupos}
        with open(os.path.join(saida, f"{nome}_correcao.json"), "w", encoding="utf-8") as f:
            json.dump(cor, f, indent=2)
        manter = (labels == MAIN) | np.isin(labels, list(T)) | (labels == DUP)
        gravar_ply_xyz(os.path.join(saida, f"{nome}_fusao_antes.ply"), work[manter])
        gravar_ply_xyz(os.path.join(saida, f"{nome}_fusao_corrigida.ply"), final_points(work, labels, T)[0])
        log("   CONFIRA o detalhe nao simetrico nas fusoes antes/corrigida contra o gabarito antes de "
            "'revo aplicar'.")
    return dict(status=status, alertas=alertas, base=base_fixa, voxel=voxel, n_trabalho=len(work),
                metricas={k: v for k, v in m.items() if not k.startswith("_")}, criterios=m.get("_checks"),
                fragmentos=info.get("fragmentos"), quadros_corrigir={str(g["trecho"]): g["revo"] for g in grupos},
                quadros_ambiguos=rv[ambiguos].tolist(), **rel, **r)


def limpar_projeto(proj, scans, saida=None, cfg=None, faixa_base=2.5, log=print):
    """Gera uma COPIA do projeto so com os scans pedidos, limpos quadro a quadro. Poses inalteradas.

    Retorna (pasta, relatorio). Correcoes de pose propostas ficam em <scan>_correcao.json.
    """
    cfg = cfg or Config()
    proj = os.path.abspath(proj)
    saida, nome = preparar_destino(proj, "_limpo", saida)
    rel = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for s in scans:
            t0 = time.time()
            log(f"== {s}")
            with RevoScan(os.path.join(proj, "data", s), tmp) as scan:
                erros = calibrar_modos(scan, log)
                if max(erros.values()) > ERRO_MAX_CALIB:
                    log(f"   PULADO: calibracao com erro acima de {ERRO_MAX_CALIB} mm")
                    rel[s] = dict(status="pulado", erros=erros)
                    continue
                rel[s] = dict(erros=erros, **limpar_scan(scan, cfg, faixa_base, saida, log),
                              tempo_s=round(time.time() - t0))
            log(f"   status: {rel[s]['status']} ({rel[s]['tempo_s']} s)")
    limpos = [s for s in rel if rel[s]["status"] != "pulado"]
    if limpos:
        gravar_revo(arquivo_revo(proj), saida, nome, limpos)
    with open(os.path.join(saida, "relatorio_limpeza.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2, default=float)
    log(f"saida: {saida}")
    return saida, rel
