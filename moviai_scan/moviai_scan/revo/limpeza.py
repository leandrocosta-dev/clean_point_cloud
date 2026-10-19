"""Limpeza quadro a quadro: apaga de cada quadro os pixels fora da peca (prato, pecas de apoio).

As poses NAO sao alteradas. A limpeza so apaga pixels do mapa de profundidade
de cada quadro (mascara de bits + valores), mantendo o resto do registro igual.
"""
import json
import os
import struct
import tempfile
import time

import numpy as np
from scipy import ndimage

from .formato import (RevoScan, ler_ply, gravar_ply, eixos_do_plano, gravar_copia_scan, gravar_revo,
                      arquivo_revo, preparar_destino, calibrar_modos, ERRO_MAX_CALIB)


def plano_ransac(P, thr=1.0, it=3000, seed=0):
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        k = int((np.abs(P @ n + d) < thr).sum())
        if k > best[0]:
            best = (k, (n, d))
    n, d = best[1]
    m = np.abs(P @ n + d) < thr
    c = P[m].mean(0)
    _, _, V = np.linalg.svd(P[m] - c, full_matrices=False)
    n = V[2]
    d = -n @ c
    # orienta a normal para o lado onde estao os objetos
    if np.mean(P @ n + d > 3 * thr) < np.mean(P @ n + d < -3 * thr):
        n, d = -n, -d
    return n, d, float(m.mean())


def achar_base(A, cams, candidatos=4, min_frac=0.03, min_cam=0.8, max_abaixo=0.05, log=print):
    """Procura o plano do prato entre os maiores planos da cena.

    Um plano so e aceito como base se:
      - as cameras estao do lado de cima dele (>= 80%): o scanner sempre olha o
        prato de cima. Isso rejeita faces planas da propria peca, que sao vistas
        do lado de fora (teste real: face de cima do difusor tinha 0% de cameras
        do lado da peca);
      - quase nenhum ponto fica abaixo dele (<= 5%);
      - tem pelo menos 3% dos pontos da amostra.
    Retorna (n, d) com a normal apontando para as cameras, ou None se nao achar.
    """
    resto = A
    for i in range(candidatos):
        if len(resto) < 1000:
            break
        n, d, _ = plano_ransac(resto[:: max(1, len(resto) // 40000)], thr=1.0, it=800)
        if np.mean(cams @ n + d > 0) < 0.5:          # normal para o lado das cameras
            n, d = -n, -d
        hA = A @ n + d
        frac = float(np.mean(np.abs(hA) < 1.0))
        cam_ok = float(np.mean(cams @ n + d > 0))
        abaixo = float(np.mean(hA < -2.5))
        aceito = frac >= min_frac and cam_ok >= min_cam and abaixo <= max_abaixo
        log(f"   plano {i}: {frac*100:.1f}% dos pontos | cameras acima {cam_ok*100:.0f}% | "
            f"pontos abaixo {abaixo*100:.1f}% -> {'BASE' if aceito else 'rejeitado'}")
        if aceito:
            return n, d
        resto = resto[np.abs(resto @ n + d) >= 1.5]
    log("   nenhum plano passou como base: nada sera removido por altura, so objetos soltos")
    return None


def amostrar(scan, passo=6, nquadros=150, raio_centro=0.15):
    """Funde uma amostra dos quadros no referencial do mundo.

    Retorna (pontos, posicoes das cameras, pontos do centro da imagem de cada quadro).
    """
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    amostra, centrais, cams = [], [], []
    for k in ks[:: max(1, len(ks) // nquadros)]:
        r = scan.profundidade(scan.ids[k])
        if r is None:
            continue
        w = r[0]
        P, pix = scan.pontos(k, passo=passo)
        if P is None or len(P) < 100:
            continue
        amostra.append(P)
        cams.append(scan.T[k][:3, 3])
        fx, fy, cx, cy = scan.K[int(scan.modo[k])]
        v, u = np.divmod(pix, w)
        centro = (u - cx) ** 2 + (v - cy) ** 2 < (raio_centro * w) ** 2
        centrais.append(P[centro])
    return np.vstack(amostra), np.array(cams), centrais


def segmentar_peca(scan, voxel=1.0, faixa_base=2.5, passo=6, raio_centro=0.15, log=print):
    """Separa a peca do prato e das pecas de apoio.

    1. Funde uma amostra dos quadros (ate cerca de 150) no referencial do mundo.
    2. Acha o plano da base (prato), so aceitando planos vistos de cima (achar_base).
    3. Agrupa os pontos acima da base em componentes conexas (grade de voxels).
    4. A peca e a componente que mais aparece no CENTRO da imagem ao longo da
       captura: cada quadro vota no objeto com mais pixels num circulo de raio
       raio_centro * largura em volta do ponto principal (cx, cy). O operador sempre
       aponta o scanner para a peca. Desempate pelo tamanho.
       (Versao anterior usava um ponto no eixo optico na profundidade mediana; falhou
       no teste com gabarito quando o prato dominava a imagem e escolheu a esfera de apoio.)
    Retorna um dicionario com o plano, a grade e o conjunto de voxels a manter.
    """
    A, cams, centrais = amostrar(scan, passo, raio_centro=raio_centro)
    base = achar_base(A, cams, log=log)
    if base is None:
        n, d = np.zeros(3), 1e9          # sem base: todo ponto conta como "acima"
        acima = A
    else:
        n, d = base
        acima = A[A @ n + d > faixa_base]
        log(f"   {len(acima):,} pontos acima da base")

    lo = acima.min(0) - 3 * voxel
    dims = np.ceil((acima.max(0) + 3 * voxel - lo) / voxel).astype(int)
    idx = np.floor((acima - lo) / voxel).astype(int)
    grade = np.zeros(dims, bool)
    grade[tuple(idx.T)] = True
    rot, ncomp = ndimage.label(grade, structure=np.ones((3, 3, 3)))
    tam = np.bincount(rot[tuple(idx.T)], minlength=ncomp + 1)

    def componente(Q):
        ii = np.floor((Q - lo) / voxel).astype(int)
        dentro = np.all((ii >= 0) & (ii < np.array(dims)), axis=1)
        c = np.zeros(len(Q), int)
        c[dentro] = rot[tuple(ii[dentro].T)]
        if base is not None:
            c[Q @ n + d <= faixa_base] = 0
        return c

    votos = np.zeros(ncomp + 1, int)
    validos = 0
    for Q in centrais:
        c = componente(Q)
        c = c[c > 0]
        if len(c) < 20:
            continue
        votos[np.bincount(c).argmax()] += 1
        validos += 1
    ordem = np.argsort(-tam)
    log("   objetos acima da base (pontos | quadros com o objeto no centro):")
    for c in ordem[:6]:
        if c == 0 or tam[c] == 0:
            continue
        pts = acima[rot[tuple(idx.T)] == c]
        ext = np.ptp(pts, 0)
        log(f"     objeto {c}: {tam[c]:,} pts | {votos[c]} quadros | extensao {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
    if votos.max() > 0:
        melhor = votos.max()
        empatados = [c for c in range(1, ncomp + 1) if votos[c] == melhor]
        peca = max(empatados, key=lambda c: tam[c])
    else:
        peca = int(ordem[0] if ordem[0] else ordem[1])
    manter = ndimage.binary_dilation(rot == peca, structure=np.ones((3, 3, 3)), iterations=1)
    log(f"   peca escolhida: objeto {peca} ({votos[peca]} de {validos} quadros)")
    if validos and votos[peca] < 0.5 * validos:
        log("   AVISO: a peca escolhida esta no centro em menos da metade dos quadros; conferir a imagem")
    return dict(n=n, d=d, faixa_base=faixa_base, voxel=voxel, lo=lo, manter=manter,
                peca=peca, votos=votos.tolist(), tam=tam.tolist(), tem_base=base is not None)


def mascara_manter(P, seg):
    h = P @ seg["n"] + seg["d"]
    idx = np.floor((P - seg["lo"]) / seg["voxel"]).astype(int)
    dentro = np.all((idx >= 0) & (idx < np.array(seg["manter"].shape)), axis=1)
    ok = np.zeros(len(P), bool)
    ii = idx[dentro]
    ok[dentro] = seg["manter"][tuple(ii.T)]
    return ok & (h > seg["faixa_base"])


def limpar_scan(scan, seg, destino_scan, log=print):
    """Grava a copia do scan com os pixels fora da peca apagados em cada quadro."""
    tot = {"antes": 0, "depois": 0}
    vazios = []

    def apagar_pixels(cn):
        for k, i in enumerate(scan.ids):
            if not scan.tem_prof[k]:
                continue
            w, h, mask, z, quatro = scan.profundidade(i)
            P, pix = scan.pontos(k)
            ok = mascara_manter(P, seg)
            tot["antes"] += len(pix)
            if ok.sum() == 0:
                # quadro sem nenhum ponto da peca: fica como estava
                vazios.append(k)
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

    # raw_preview filtrado pela mesma regra (esta no referencial do mundo)
    hdr, prev = ler_ply(os.path.join(scan.pasta, "raw_preview.ply"))
    okp = mascara_manter(prev[:, :3].astype(float), seg)
    gravar_ply(os.path.join(destino_scan, "raw_preview.ply"), hdr, prev[okp])
    antes, depois = tot["antes"], tot["depois"]
    log(f"   pontos dos quadros: {antes:,} -> {depois:,} ({depois/max(1,antes)*100:.0f}% mantidos) | "
        f"preview {len(prev):,} -> {okp.sum():,} | integridade: {integ}")
    if vazios:
        log(f"   AVISO: {len(vazios)} quadros sem nenhum ponto da peca foram mantidos como estavam")
    return dict(antes=antes, depois=depois, vazios=len(vazios), integridade=integ)


def render_comparacao(scan, seg, path, nquadros=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    ks = ks[:: max(1, len(ks) // nquadros)]
    P = np.vstack([p for k in ks if (p := scan.pontos(k, passo=6)[0]) is not None])
    ok = mascara_manter(P, seg)
    n = seg["n"] if seg.get("tem_base", True) else np.array([0.0, 0.0, 1.0])
    d = seg["d"] if seg.get("tem_base", True) else 0.0
    u, v = eixos_do_plano(n)
    L = np.c_[P @ u, P @ v, P @ n + d]
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), dpi=90)
    for col, (titulo, m) in enumerate((("ANTES (todos os pontos)", np.ones(len(L), bool)),
                                       ("DEPOIS (so a peca)", ok))):
        ax[0, col].scatter(L[m, 0], L[m, 1], s=0.2, c=L[m, 2], cmap="viridis", linewidths=0)
        ax[0, col].set_aspect("equal"); ax[0, col].set_title(titulo + ": vista de cima")
        ax[1, col].scatter(L[m, 0], L[m, 2], s=0.2, c="#333", linewidths=0)
        ax[1, col].set_aspect("equal"); ax[1, col].set_title(titulo + ": vista lateral")
    for a in ax[:, 1]:
        a.set_xlim(ax[0, 0].get_xlim())
    ax[0, 1].set_ylim(ax[0, 0].get_ylim()); ax[1, 1].set_ylim(ax[1, 0].get_ylim())
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def limpar_projeto(proj, scans, saida=None, log=print):
    """Gera uma COPIA do projeto so com os scans pedidos, com prato e pecas de apoio
    apagados de cada quadro. Poses inalteradas. O original nao e tocado."""
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
                seg = segmentar_peca(scan, log=log)
                r = limpar_scan(scan, seg, os.path.join(saida, "data", s), log=log)
                render_comparacao(scan, seg, os.path.join(saida, f"{s}_antes_depois.png"))
                rel[s] = dict(status="limpo", erros=erros, peca=seg["peca"], **r,
                              tempo_s=round(time.time() - t0))
    limpos = [s for s in rel if rel[s]["status"] == "limpo"]
    if limpos:
        gravar_revo(arquivo_revo(proj), saida, nome, limpos)
    json.dump(rel, open(os.path.join(saida, "relatorio_limpeza.json"), "w", encoding="utf-8"), indent=2)
    log(f"saida: {saida}")
    return saida, rel
