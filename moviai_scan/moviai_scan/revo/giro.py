"""Deriva de giro em torno do eixo de uma peca cilindrica (trecho cego do rastreamento).

Caso que originou o metodo (scan de 15/10): os quadros #70 a #147 viam so o
cilindro liso, o giro em torno do eixo ficou sem referencia e acumulou 17,2 graus.
Correcao: rotacao em torno do eixo, linear de 0 ao giro total no trecho cego
(rampa) e constante depois dele.

Fluxo seguro, em duas etapas:
  1. estimar_giro + fundir: gera correcao.json e as nuvens fundidas antes/depois
     para conferir o detalhe NAO simetrico (plugue) contra o gabarito.
  2. aplicar_correcao: so depois da conferencia, grava uma COPIA do projeto
     com as poses corrigidas. O Revo Scan usa as poses exatamente como gravadas.
Distancia media a superficie NAO prova encaixe em corpo cilindrico (CONTEXTO.md).
"""
import json
import os
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from .formato import (RevoScan, eixos_do_plano, transformar, voxel_media, gravar_copia_scan, gravar_revo,
                      gravar_ply_xyz, arquivo_revo, preparar_destino, exigir_calibracao)
from .prato import amostrar, achar_base


def rotacao_eixo(ponto, direcao, graus):
    """Matriz 4x4 de rotacao de graus em torno do eixo (ponto, direcao)."""
    a = np.radians(graus)
    k = np.asarray(direcao, float)
    p = np.asarray(ponto, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p - R @ p
    return M


def giros_por_quadro(cor, n_quadros):
    """Giro (graus) de cada quadro: 0 antes da rampa, linear na rampa, constante ate 'ate'."""
    a, b = cor["rampa"]
    ate = cor.get("ate", n_quadros - 1)
    k = np.arange(n_quadros)
    frac = np.where(k < a, 0.0, np.where(k <= b, (k - a + 1) / (b - a + 2), 1.0))
    frac[k > ate] = 0.0
    return frac * cor["graus"]


def estimar_giro(scan, base, centro, raio, altura, ref, alvo, voxel=0.3, margem=1.5, log=print):
    """Giro em torno do eixo que leva o detalhe dos quadros 'alvo' sobre o dos quadros 'ref'.

    Eixo = normal da base passando por 'centro' (ponto do mundo no eixo).
    Detalhe = pontos fora do cilindro (raio + margem) e acima de 'altura' sobre a base.
    ref e alvo sao intervalos inclusivos de indices de quadro (ordem temporal).
    Retorna (graus, sobreposicao com a correcao, sobreposicao sem correcao).
    """
    n, d = base
    u, v = eixos_do_plano(n)
    c2 = np.array([centro @ u, centro @ v])

    def detalhe(ks):
        out = []
        for k in ks:
            P, _ = scan.pontos(k, passo=2)
            if P is None:
                continue
            q = np.c_[P @ u, P @ v, P @ n + d]
            rr = np.linalg.norm(q[:, :2] - c2, axis=1)
            out.append(q[(rr > raio + margem) & (q[:, 2] > altura)])
        return voxel_media(np.vstack(out), voxel)

    A = detalhe(range(ref[0], ref[1] + 1))
    B = detalhe(range(alvo[0], alvo[1] + 1))
    log(f"   detalhe: {len(A):,} pontos na referencia, {len(B):,} no alvo")
    tA = cKDTree(A)

    def girar(p, g):
        a = np.radians(g)
        x, y = p[:, 0] - c2[0], p[:, 1] - c2[1]
        return np.c_[c2[0] + np.cos(a) * x - np.sin(a) * y, c2[1] + np.sin(a) * x + np.cos(a) * y, p[:, 2]]

    def sobrepos(g):
        return float(np.mean(tA.query(girar(B, g))[0] < 0.5))

    grosso = max(((g, sobrepos(g)) for g in np.arange(-30, 30.01, 0.5)), key=lambda x: x[1])[0]
    graus, melhor = max(((g, sobrepos(g)) for g in np.arange(grosso - 1, grosso + 1.001, 0.05)),
                        key=lambda x: x[1])
    sem = sobrepos(0)
    log(f"   giro estimado: {graus:+.2f} graus | sobreposicao {melhor*100:.1f}% (sem correcao {sem*100:.1f}%)")
    return float(graus), melhor, sem


def fundir(scan, correcao=None, voxel=0.15, log=print):
    """Funde todos os quadros com profundidade numa nuvem (opcionalmente com a correcao de pose)."""
    novas, mud = poses_corrigidas(scan, correcao) if correcao else (None, np.zeros(len(scan.ids), bool))
    acc, blocos, n_blocos = np.empty((0, 3)), [], 0
    for k in range(len(scan.ids)):
        P, _ = scan.pontos(k)
        if P is None:
            continue
        if mud[k]:
            P = transformar(P, novas[k] @ np.linalg.inv(scan.T[k]))
        blocos.append(P)
        n_blocos += len(P)
        if n_blocos > 5_000_000:      # reduz aos poucos para caber na memoria
            acc = voxel_media(np.vstack([acc] + blocos), voxel)
            blocos, n_blocos = [], 0
    acc = voxel_media(np.vstack([acc] + blocos), voxel)
    log(f"   nuvem fundida: {len(acc):,} pontos (voxel {voxel} mm)")
    return acc


def preparar_correcao(proj, nome_scan, ref, alvo, rampa, centro, raio, altura, saida, ate=None, base=None,
                      log=print):
    """Etapa 1: estima o giro e grava correcao.json + fusao_original.ply + fusao_corrigida.ply em saida.

    Nada e gravado no projeto. Conferir o detalhe nao simetrico nas duas fusoes antes de aplicar.
    base: (n, d) do prato; se None, procura com achar_base (pode recusar o prato, ver 'revo giro --base-de').
    """
    os.makedirs(saida, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with RevoScan(os.path.join(proj, "data", nome_scan), tmp) as scan:
            exigir_calibracao(scan, log)
            if base is None:
                A, cams, _ = amostrar(scan)
                base = achar_base(A, cams, log=log)
            if base is None:
                raise ValueError("base (prato) nao encontrada: o eixo do giro e a normal da base. "
                                 "Use --base-de com o relatorio de 'run.py limpar' do raw_preview.ply")
            base = (np.asarray(base[0], float), float(base[1]))
            centro = np.asarray(centro, float)
            graus, depois, antes = estimar_giro(scan, base, centro, raio, altura, ref, alvo, log=log)
            cor = {"scan": nome_scan, "eixo_ponto": centro.tolist(), "eixo_direcao": base[0].tolist(),
                   "graus": round(graus, 3), "rampa": list(rampa),
                   "ate": int(ate if ate is not None else len(scan.ids) - 1),
                   "ref": list(ref), "alvo": list(alvo), "raio": raio, "altura": altura,
                   "sobreposicao_antes": round(antes, 4), "sobreposicao_depois": round(depois, 4)}
            for nome, c in (("fusao_original", None), ("fusao_corrigida", cor)):
                gravar_ply_xyz(os.path.join(saida, nome + ".ply"), fundir(scan, c, log=log))
    with open(os.path.join(saida, "correcao.json"), "w", encoding="utf-8") as f:
        json.dump(cor, f, indent=2)
    log(f"saida: {saida}")
    log("CONFIRA o detalhe nao simetrico (plugue) nas duas fusoes contra o gabarito antes de aplicar.")
    return cor


def aplicar(proj, caminho_correcao, saida=None, log=print):
    """Etapa 2: grava uma COPIA do projeto com as poses corrigidas. O original nao e tocado.

    Aceita a correcao de 'revo giro' e a de 'revo limpar' (tipo quadros).
    """
    with open(caminho_correcao, encoding="utf-8") as f:
        cor = json.load(f)
    proj = os.path.abspath(proj)
    saida, nome = preparar_destino(proj, "_corrigido" if cor.get("tipo") == "quadros" else "_giro", saida)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with RevoScan(os.path.join(proj, "data", cor["scan"]), tmp) as scan:
            r = aplicar_correcao(scan, cor, proj, saida, nome, tmp, log)
    log(f"saida: {saida}")
    return saida, r


def poses_corrigidas(scan, correcao):
    """Poses novas e mascara dos quadros alterados.

    tipo giro (padrao): rotacao em torno do eixo, por quadro (giros_por_quadro).
    tipo quadros: pose = T @ pose para os quadros (indice na ordem de captura) de cada grupo.
    """
    if correcao.get("tipo") == "quadros":
        if correcao["n_quadros"] != len(scan.ids):
            raise ValueError(f"a correcao e de um scan com {correcao['n_quadros']} quadros; "
                             f"este tem {len(scan.ids)}")
        novas, mud = scan.T.copy(), np.zeros(len(scan.ids), bool)
        for g in correcao["grupos"]:
            for k in g["quadros"]:
                novas[k] = np.asarray(g["T"]) @ scan.T[k]
                mud[k] = True
        return novas, mud
    giros = giros_por_quadro(correcao, len(scan.ids))
    return np.array([rotacao_eixo(correcao["eixo_ponto"], correcao["eixo_direcao"], g) @ T if g else T
                     for g, T in zip(giros, scan.T)]), giros != 0


def aplicar_correcao(scan, correcao, proj, saida, nome, tmp, log=print):
    """Grava a copia do projeto com as poses corrigidas e confere relendo."""
    novas, mud = poses_corrigidas(scan, correcao)

    def corrigir_poses(cn):
        for k, (i, b) in enumerate(zip(scan.ids, scan.bins)):
            if mud[k]:
                cn.execute("update FrameInfo set bin=? where id=?",
                           (b[:16] + novas[k].astype("<f8").tobytes() + b[144:], i))

    destino = os.path.join(saida, "data", scan.nome)
    integ = gravar_copia_scan(scan, destino, corrigir_poses)
    gravar_revo(arquivo_revo(proj), saida, nome, [scan.nome])
    with open(os.path.join(saida, "correcao.json"), "w", encoding="utf-8") as f:
        json.dump(correcao, f, indent=2)

    # rele a copia gravada: poses esperadas e cabecalho/cauda de cada registro intactos
    pasta_conf = os.path.join(tmp, "conferencia")
    os.makedirs(pasta_conf, exist_ok=True)
    with RevoScan(destino, pasta_conf) as copia:
        if copia.ids != scan.ids:
            raise RuntimeError("a copia gravada tem quadros diferentes do original")
        erro = float(np.abs(copia.T - novas).max())
        resto_igual = all(a[:16] == b[:16] and a[144:] == b[144:] for a, b in zip(copia.bins, scan.bins))
    n_mud = int(np.count_nonzero(mud))
    log(f"   {n_mud} poses corrigidas | erro de gravacao {erro:.2e} | resto do registro igual: {resto_igual} "
        f"| integridade: {integ}")
    log("   raw_preview.ply e preview.png da copia continuam os antigos")
    if erro > 1e-9 or not resto_igual:
        raise RuntimeError("a copia gravada nao confere com as poses esperadas")
    return dict(quadros_corrigidos=n_mud, erro_gravacao=erro, integridade=integ)
