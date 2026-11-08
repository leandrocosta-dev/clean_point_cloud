"""Limpeza por COMPONENTE com decisao explicita (keep/delete em JSON) e verificacao numerica por quadro.

    python pipeline/componentes.py segmentar <projeto> --saida D:/Projects/X_comp [--scans id ...] [--mesa]
    python pipeline/componentes.py aplicar   <projeto> D:/Projects/X_comp --decisoes decisoes.json --saida <projeto_novo>
    python pipeline/componentes.py verificar <projeto_novo> --saida D:/Projects/X_verif

Divisao de trabalho:
  * Codigo (regras fixas, sem LLM), em `segmentar`, por quadro: corte por distancia de trabalho (--z-min/--z-max, mm da
    camera), remocao de pixels voadores em borda de profundidade (angulo normal x visada > --angulo graus) e, com
    --mesa, o plano da mesa por RANSAC (pontos a < --mesa-mm do plano saem). Depois, no scan inteiro (coordenadas do
    mundo): componentes conexos em grade de --voxel mm (vizinhanca 26, voxels com >= --min-pts-voxel pontos), que
    reproduz o "Selecionar Conexao" do Revo, mas automatico e para o scan todo.
  * Decisao (LLM ou pessoa): `segmentar` grava, por scan, `componentes.json` (id, n de pontos, caixa, centroide,
    distancia a mesa, planaridade, n de quadros que veem o componente) e `componentes.png` (cada componente numa cor,
    id escrito, 3 vistas ortogonais + perspectiva). Quem decide responde em `decisoes.json`:
        {"<scan>": {"keep": [ids], "delete": [ids], "confianca": 0..1, "motivo": "...",
                    "remover_quadros": [[ini, fim], ...]}}     # opcional: trechos de quadros que perdem a profundidade
    Componente nao citado: fica se tiver >= --min-comp pontos, sai se for menor (ilha de ruido).
  * Propagacao entre quadros: o componente e do scan, nao do quadro; cada pixel de cada quadro herda a decisao do voxel
    onde cai. Nao ha decisao por quadro; quadros fora do padrao vao para a fila de revisao (abaixo).
  * Verificacao (`aplicar` grava `verificacao.json`; `verificar` mede de novo um projeto qualquer): componentes que
    sobraram por scan, fracao de pontos removida, e por QUADRO a sobreposicao com os quadros vizinhos (+-5): fracao dos
    pontos mantidos com vizinho a < 1 mm na nuvem dos vizinhos. Quadros com sobreposicao < --sobrep-min ou que perderam
    > --perda-max dos pontos vao para `revisao.json` + folha de contatos `revisao.png` (mapa de profundidade do quadro,
    mantido em cinza, removido em vermelho) para conferencia humana. Correcoes humanas entram em decisoes.json.
LIMITE (testado no Projeto 9): lixo ENCOSTADO na peca (massinha de apoio, fiapos nos dentes) fica no mesmo componente
da peca; separar por erosao nao funciona porque a nuvem e uma casca de 1-2 voxels. Para isso use o consenso entre scans
(`avaliar_merge_revo.py` + `limpar_por_merge_revo.py`): o apoio muda de lugar entre poses, a peca nao.
O projeto de origem nunca e alterado; o destino nao pode existir.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from recursos import baixa_prioridade

AQUI = os.path.dirname(os.path.abspath(__file__))
import gpu
import revo_projeto as RP
from revo_projeto import ArvoreGPU, RevoScan, gravar_revo, plano_ransac


def scans_do_projeto(proj, filtro=None):
    revo = next(f for f in os.listdir(proj) if f.endswith(".revo"))
    nos = json.load(open(os.path.join(proj, revo), encoding="utf-8"))["nodes"]
    ids = [n["guid"] for n in nos if n.get("type") != 5 and os.path.isdir(os.path.join(proj, "data", n["guid"]))]
    return revo, nos, [i for i in ids if not filtro or i in filtro]


def abrir_scan(proj, s, tmp):
    scan = RevoScan(os.path.join(proj, "data", s), tmp)
    for m in sorted(set(int(m) for m, p in zip(scan.modo, scan.tem_prof) if p)):
        scan.calibrar(m)
    return scan


# ---------------------------------------------------------------- regras fixas por quadro
def filtro_quadro(scan, k, a, plano=None):
    """Devolve (P_mundo cupy, pix numpy, ok cupy) do quadro k: ok = passa no corte de distancia, no filtro de borda e
    fora da mesa. Tudo na GPU (normais por diferencas finitas da imagem de pontos)."""
    r = scan.profundidade(scan.ids[k])
    if r is None:
        return None, None, None
    m = gpu.xp()
    w, h, mask, z, _ = r
    pix = np.flatnonzero(mask)
    pix_g = gpu.para_gpu(pix)
    v, u = m.divmod(pix_g, w)
    fx, fy, cx, cy = scan.K[int(scan.modo[k])]
    zf = gpu.para_gpu(z, m.float64)
    Pc = m.stack([(u - cx) * zf / fx, -(v - cy) * zf / fy, -zf], axis=1)
    ok = (zf >= a.z_min) & (zf <= a.z_max)
    # pixels voadores: normal pela imagem (diferencas finitas) quase perpendicular a visada
    img = m.full((h * w, 3), m.nan)
    img[pix_g] = Pc
    img = img.reshape(h, w, 3)
    gx = m.full_like(img, m.nan)
    gy = m.full_like(img, m.nan)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    n = m.cross(gx.reshape(-1, 3)[pix_g], gy.reshape(-1, 3)[pix_g])
    nn = m.linalg.norm(n, axis=1)
    vis = Pc / m.linalg.norm(Pc, axis=1, keepdims=True)
    cosang = m.abs((n * vis).sum(1)) / nn
    rasante = m.isfinite(cosang) & (cosang < np.cos(np.radians(a.angulo)))
    ok &= ~rasante
    T = scan.T[k]
    P = Pc @ gpu.para_gpu(T[:3, :3].T) + gpu.para_gpu(T[:3, 3])
    if plano is not None:
        nrm, d = plano
        ok &= (P @ gpu.para_gpu(nrm, m.float64) + d) > a.mesa_mm
    return P, pix, ok


# ---------------------------------------------------------------- segmentar
def componentes_do_scan(scan, a, plano):
    """Amostra os quadros (pontos que passam nas regras), voxeliza no mundo e rotula componentes conexos (GPU:
    bincount, cupyx.ndimage.label e transformada de distancia). Devolve cupy: A, Kq, lab, rot (lo numpy)."""
    m = gpu.xp()
    nd = RP._nd()
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    amostra, quadro_de = [], []
    for k in ks[:: a.amostra_quadros]:
        P, pix, ok = filtro_quadro(scan, k, a, plano)
        if P is None:
            continue
        Q = P[ok][:: a.amostra_passo]
        amostra.append(Q.astype(m.float32))
        quadro_de.append(m.full(len(Q), k, m.int32))
    A = m.vstack(amostra)
    Kq = m.concatenate(quadro_de)
    lo = gpu.para_cpu(A.min(0)).astype(np.float64) - 2 * a.voxel
    g = m.floor((A - gpu.para_gpu(lo)) / a.voxel).astype(m.int64)
    dims = [int(x) for x in gpu.para_cpu(g.max(0)) + 2]
    cont = m.bincount((g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2], minlength=dims[0] * dims[1] * dims[2]).reshape(
        dims
    )
    denso = cont >= a.min_pts_voxel
    lab, nl = nd.label(denso, structure=RP._uns(3, 3, 3))
    nl = int(nl)
    # voxels esparsos encostados num componente herdam o rotulo (borda da peca); esparsos soltos ficam 0 = ruido
    esparso = (cont > 0) & ~denso
    if bool(esparso.any()):
        dist, (ix, iy, iz) = nd.distance_transform_edt(lab == 0, return_indices=True)
        viz = esparso & (dist <= 1.8)
        lab[viz] = lab[ix[viz], iy[viz], iz[viz]]
    rot = lab[g[:, 0], g[:, 1], g[:, 2]]
    return A, Kq, lo, lab, rot, nl


def tabela(A, Kq, rot, nl, plano, voxel_lab):
    """Tabela de componentes (GPU: mascaras, extensoes, covariancia; o autovalor 3x3 final e resolvido na CPU)."""
    m = gpu.xp()
    A, Kq, rot = gpu.para_gpu(A), gpu.para_gpu(Kq), gpu.para_gpu(rot)
    linhas = []
    tam = gpu.para_cpu(m.bincount(rot, minlength=nl + 1))
    for c in np.argsort(-tam):
        if c == 0 or tam[c] == 0:
            continue
        sel = rot == c
        Q = A[sel].astype(m.float64)
        ext = gpu.para_cpu(Q.max(0) - Q.min(0))
        cen = gpu.para_cpu(Q.mean(0))
        if len(Q) > 3:
            sub = Q[:: max(1, len(Q) // 20000)]
            cov = gpu.para_cpu(m.cov((sub - Q.mean(0)).T))
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        else:
            ev = np.zeros(3)
        planar = float((ev[1] - ev[2]) / ev[0]) if ev[0] > 0 else 0.0
        lin = {
            "id": int(c),
            "pontos_amostra": int(tam[c]),
            "fracao": float(tam[c] / len(A)),
            "caixa_mm": [round(float(x), 1) for x in ext],
            "centroide": [round(float(x), 1) for x in cen],
            "planaridade": round(planar, 3),
            "espessura_mm": round(float(np.sqrt(max(ev[2], 0)) * 2), 2),
            "quadros": int(len(m.unique(Kq[sel]))),
        }
        if plano is not None:
            h = Q @ gpu.para_gpu(plano[0], m.float64) + plano[1]
            lin["dist_mesa_mm"] = [round(float(m.percentile(h, 5)), 1), round(float(m.percentile(h, 95)), 1)]
        linhas.append(lin)
    return linhas


def render(A, rot, linhas, caminho, max_cores=20):
    """Grafico (matplotlib, CPU): A e rot chegam em cupy ou numpy e sao trazidos uma vez."""
    A, rot = gpu.para_cpu(A), gpu.para_cpu(rot)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = [l["id"] for l in linhas[:max_cores]]
    cmap = plt.get_cmap("tab20")
    cor = np.full((len(A), 3), 0.75)
    for i, c in enumerate(ids):
        cor[rot == c] = cmap(i % 20)[:3]
    cor[rot == 0] = (0.9, 0.9, 0.9)
    c0 = A.mean(0)
    _, _, Vt = np.linalg.svd((A[::20] - c0).astype(float), full_matrices=False)
    R = (A - c0) @ Vt.T
    pers = R @ np.array([[0.707, -0.408, 0.577], [0.707, 0.408, -0.577], [0, 0.816, 0.577]])
    vistas = [
        ("topo (PCA 1x2)", R[:, 0], R[:, 1], R[:, 2]),
        ("frente (1x3)", R[:, 0], R[:, 2], -R[:, 1]),
        ("lado (2x3)", R[:, 1], R[:, 2], R[:, 0]),
        ("perspectiva", pers[:, 0], pers[:, 1], pers[:, 2]),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(28, 7.5))
    sub = np.arange(0, len(A), max(1, len(A) // 300000))
    for j, (t, x, y, z) in enumerate(vistas):
        o = sub[np.argsort(z[sub])]
        ax[j].scatter(x[o], y[o], s=0.15, c=cor[o])
        ax[j].set_aspect("equal")
        ax[j].set_title(t)
        for i, c in enumerate(ids):
            m = rot == c
            if m.any():
                ax[j].text(
                    np.median(x[m]),
                    np.median(y[m]),
                    str(c),
                    fontsize=12,
                    weight="bold",
                    color="k",
                    bbox=dict(facecolor=cmap(i % 20), alpha=0.8, pad=1),
                )
    plt.tight_layout()
    plt.savefig(caminho, dpi=55)
    plt.close()


def cmd_segmentar(a):
    proj = os.path.abspath(a.projeto)
    _, _, scans = scans_do_projeto(proj, a.scans)
    os.makedirs(a.saida, exist_ok=True)
    resumo = {}
    for s in scans:
        with tempfile.TemporaryDirectory() as tmp:
            scan = abrir_scan(proj, s, tmp)
            plano = None
            if a.mesa:
                ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]][:: max(1, a.amostra_quadros * 2)]
                B = gpu.xp().vstack([scan.pontos(k, passo=8, na_gpu=True)[0] for k in ks])
                nrm, d, frac = plano_ransac(B[:: max(1, len(B) // 150000)], thr=1.5, it=2000)
                plano = (nrm / np.linalg.norm(nrm), float(d))
            A, Kq, lo, lab, rot, nl = componentes_do_scan(scan, a, plano)
            linhas = tabela(A, Kq, rot, nl, plano, lab)
            d = os.path.join(a.saida, s)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(
                os.path.join(d, "rotulos.npz"),
                lo=lo,
                voxel=a.voxel,
                lab=gpu.para_cpu(lab).astype(np.int32),
                plano=np.r_[plano[0], plano[1]] if plano else np.zeros(0),
            )
            json.dump(
                {
                    "scan": s,
                    "parametros": {k: v for k, v in vars(a).items() if k not in ("f", "scans")},
                    "ruido_fracao": float((rot == 0).mean()),
                    "componentes": linhas,
                },
                open(os.path.join(d, "componentes.json"), "w", encoding="utf-8"),
                indent=1,
            )
            render(A, rot, linhas, os.path.join(d, "componentes.png"))
            scan.c.close()
        resumo[s] = [(l["id"], round(l["fracao"] * 100, 1), l["caixa_mm"]) for l in linhas[:8]]
        print(
            s[:8],
            f"{len(linhas)} componentes; ruido solto {float((rot == 0).mean()) * 100:.1f}%; maiores (id, %, caixa mm):",
            resumo[s],
            flush=True,
        )
    print("saida:", os.path.abspath(a.saida))


# ---------------------------------------------------------------- aplicar + verificar
def medir_quadros(scan, mascaras, viz=5):
    """Sobreposicao de cada quadro com os vizinhos: fracao dos pontos mantidos com vizinho < 1 mm, medida SEPARADAMENTE
    contra os `viz` quadros anteriores e os `viz` seguintes; vale o MENOR dos dois. Assim um salto de rastreio (o trecho
    depois da perda encaixa em si mesmo, mas nao no trecho anterior) aparece no quadro da fronteira.
    GPU: `mascaras[k] = (pontos, ok)` em cupy (ou numpy); indice de vizinhos na GPU por janela."""
    m = gpu.xp()
    ks = sorted(mascaras)
    nuvem = {k: gpu.para_gpu(mascaras[k][0])[gpu.para_gpu(mascaras[k][1], bool)][::4] for k in ks}
    res = {}

    def frac(Q, viz_ks):
        V = [nuvem[j] for j in viz_ks if len(nuvem[j])]
        if not V:
            return None
        d = ArvoreGPU(m.vstack(V)).query(Q, distance_upper_bound=1.0)[0]
        return float(m.isfinite(d).mean())

    for i, k in enumerate(ks):
        Q = nuvem[k]
        if len(Q) < 30:
            res[k] = 0.0
            continue
        ant = frac(Q, ks[max(0, i - viz) : i])
        seg = frac(Q, ks[i + 1 : i + 1 + viz])
        vals = [v for v in (ant, seg) if v is not None]
        res[k] = min(vals) if vals else 0.0
    return res


def folha_revisao(scan, mascaras, quadros, caminho, n=24):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quadros = quadros[:n]
    if not quadros:
        return
    col = min(6, len(quadros))
    lin = (len(quadros) + col - 1) // col
    fig, ax = plt.subplots(lin, col, figsize=(5 * col, 4 * lin), squeeze=False)
    for a_ in ax.ravel():
        a_.axis("off")
    for i, (k, motivo) in enumerate(quadros):
        w, h, mask, z, _ = scan.profundidade(scan.ids[k])
        pix = np.flatnonzero(mask)
        img = np.ones((h * w, 3))
        ok = gpu.para_cpu(mascaras[k][1]) if k in mascaras else np.zeros(len(pix), bool)
        img[pix[ok]] = 0.55
        img[pix[~ok]] = (0.85, 0.1, 0.1)
        ax.ravel()[i].imshow(img.reshape(h, w, 3))
        ax.ravel()[i].set_title(f"quadro {k}: {motivo}", fontsize=11)
    plt.tight_layout()
    plt.savefig(caminho, dpi=60)
    plt.close()


def cmd_aplicar(a):
    proj, comp, out = os.path.abspath(a.projeto), os.path.abspath(a.componentes), os.path.abspath(a.saida)
    if os.path.exists(out):
        sys.exit(f"destino ja existe: {out}")
    revo, nos, scans = scans_do_projeto(proj)
    dec = json.load(open(a.decisoes, encoding="utf-8"))
    os.makedirs(os.path.join(out, "data"))
    verif = {}
    for s in scans:
        src = os.path.join(proj, "data", s)
        if s not in dec or not os.path.exists(os.path.join(comp, s, "rotulos.npz")):
            shutil.copytree(src, os.path.join(out, "data", s))
            print(s[:8], "sem decisao: copiado")
            continue
        z = np.load(os.path.join(comp, s, "rotulos.npz"))
        lab, lo, vx = z["lab"], z["lo"], float(z["voxel"])
        prm = json.load(open(os.path.join(comp, s, "componentes.json"), encoding="utf-8"))
        P = argparse.Namespace(**prm["parametros"])
        plano = (z["plano"][:3], float(z["plano"][3])) if len(z["plano"]) else None
        tam = {l["id"]: l["pontos_amostra"] for l in prm["componentes"]}
        d = dec[s]
        manter = set(d.get("keep", [])) | {
            c for c, t in tam.items() if t >= a.min_comp and c not in set(d.get("delete", []))
        }
        manter -= set(d.get("delete", []))
        lut = np.zeros(lab.max() + 1, bool)
        lut[[c for c in manter if c <= lab.max()]] = True
        lab_g, lut_g, lo_g = (
            gpu.para_gpu(lab),
            gpu.para_gpu(lut),
            gpu.para_gpu(lo, np.float64),
        )  # grade na GPU (uma vez)
        fora = set()  # correcao humana: trechos de quadros sem profundidade (ex. rastreio perdido)
        for ini, fim in d.get("remover_quadros", []):
            fora.update(range(int(ini), int(fim) + 1))
        mascaras = {}
        with tempfile.TemporaryDirectory() as tmp:
            scan = abrir_scan(proj, s, tmp)
            estado = {"k": None, "n": -1}
            orig = scan.pontos

            def pontos(k, *x, **kw):
                r = orig(k, *x, **kw)
                estado["k"] = k
                estado["n"] = -1 if r[0] is None else len(r[0])
                return r

            scan.pontos = pontos

            def mascara(Pw, seg):
                m = gpu.xp()
                na_gpu = RP._eh_cupy(Pw)
                Pw = gpu.para_gpu(Pw, m.float64)
                k = estado["k"]
                if k is not None and len(Pw) == estado["n"] and k in fora:
                    ok = m.zeros(len(Pw), bool)
                    return ok if na_gpu else gpu.para_cpu(ok)
                if k is not None and len(Pw) == estado["n"]:
                    _, _, ok = filtro_quadro(scan, k, P, plano)
                else:
                    ok = m.ones(len(Pw), bool)
                gi = m.floor((Pw - lo_g) / vx).astype(m.int64)
                forma = gpu.para_gpu(lab.shape)
                dentro = ((gi >= 0) & (gi < forma)).all(axis=1)
                gc = m.clip(gi, 0, forma - 1)
                keep = dentro & lut_g[lab_g[gc[:, 0], gc[:, 1], gc[:, 2]]]
                ok = ok & keep
                if k is not None and len(Pw) == estado["n"]:
                    mascaras[k] = (Pw.astype(m.float32), ok)
                return ok if na_gpu else gpu.para_cpu(ok)

            RP.mascara_manter = mascara
            log = []
            RP.limpar_scan(
                scan,
                {},
                os.path.join(out, "data", s),
                log=lambda *x: log.append(" ".join(map(str, x))),
                pixels=False,
                ilhas=False,
                min_mantido=0.0,
                sem_ply_antigos=True,
                zerar_vazios=True,
                min_pix_quadro=200,
            )
            sob = medir_quadros(scan, mascaras)
            perda = {k: 1 - float(mm[1].mean()) for k, mm in mascaras.items() if len(mm[1])}
            rev = [
                (k, f"sobrep {sob[k]:.2f}")
                for k in sorted(sob)
                if sob[k] < a.sobrep_min and mascaras[k][1].sum() >= 200
            ] + [(k, f"perdeu {perda[k] * 100:.0f}%") for k in sorted(perda) if perda[k] > a.perda_max]
            d_out = os.path.join(comp, s)
            folha_revisao(scan, mascaras, rev, os.path.join(d_out, "revisao.png"))
            scan.c.close()
        tot = sum(len(mm[1]) for mm in mascaras.values())
        mant = sum(int(mm[1].sum()) for mm in mascaras.values())
        ss = np.array(list(sob.values()))
        verif[s] = {
            "mantidos": list(sorted(manter)),
            "apagados": sorted(set(tam) - manter),
            "fracao_removida": round(1 - mant / max(tot, 1), 4),
            "sobreposicao_mediana": round(float(np.median(ss)), 3) if len(ss) else None,
            "quadros_revisao": [[int(k), m] for k, m in rev],
            "confianca": d.get("confianca"),
        }
        json.dump(rev, open(os.path.join(d_out, "revisao.json"), "w"))
        print(
            s[:8],
            "\n".join(log),
            f"| removido {verif[s]['fracao_removida'] * 100:.1f}% | sobreposicao mediana "
            f"{verif[s]['sobreposicao_mediana']} | quadros p/ revisao: {len(rev)}",
            flush=True,
        )
    gravar_revo(os.path.join(proj, revo), out, os.path.splitext(revo)[0], [n["guid"] for n in nos])
    json.dump(verif, open(os.path.join(comp, "verificacao.json"), "w", encoding="utf-8"), indent=1)
    print("verificacao:", os.path.join(comp, "verificacao.json"))


def cmd_verificar(a):
    """Mede um projeto (sem decidir nada): componentes que restam por scan (mesma segmentacao) e sobreposicao entre quadros."""
    a.mesa = False
    a.scans = None
    proj = os.path.abspath(a.projeto)
    _, _, scans = scans_do_projeto(proj)
    os.makedirs(a.saida, exist_ok=True)
    rel = {}
    for s in scans:
        with tempfile.TemporaryDirectory() as tmp:
            scan = abrir_scan(proj, s, tmp)
            A, Kq, lo, lab, rot, nl = componentes_do_scan(scan, a, None)
            tam = gpu.para_cpu(gpu.xp().bincount(rot, minlength=nl + 1))
            tam[0] = 0
            grandes = int((tam >= a.min_comp).sum())
            mascaras = {}
            for k in range(len(scan.ids)):
                if scan.tem_prof[k]:
                    Pw, _ = scan.pontos(k, passo=2, na_gpu=True)
                    if Pw is not None:
                        mascaras[k] = (Pw.astype(gpu.xp().float32), gpu.xp().ones(len(Pw), bool))
            sob = medir_quadros(scan, mascaras)
            scan.c.close()
        ss = np.array(list(sob.values()))
        rel[s] = {
            "componentes_grandes": grandes,
            "componentes_total": int((tam > 0).sum()),
            "maior_fracao": round(float(tam.max() / max(tam.sum(), 1)), 4),
            "ruido_solto": round(float((rot == 0).mean()), 4),
            "sobreposicao_mediana": round(float(np.median(ss)), 3),
            "quadros_sobrep_baixa": int((ss < a.sobrep_min).sum()),
        }
        print(s[:8], rel[s], flush=True)
    json.dump(rel, open(os.path.join(a.saida, "verificacao_projeto.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def regras(p):
        p.add_argument("--z-min", type=float, default=100.0, help="mm da camera; mais perto sai")
        p.add_argument("--z-max", type=float, default=600.0, help="mm da camera; mais longe sai")
        p.add_argument(
            "--angulo", type=float, default=80.0, help="graus entre normal e visada acima dos quais o pixel e voador"
        )
        p.add_argument("--voxel", type=float, default=0.75, help="mm da grade dos componentes conexos")
        p.add_argument("--min-pts-voxel", type=int, default=2)
        p.add_argument("--amostra-quadros", type=int, default=2)
        p.add_argument("--amostra-passo", type=int, default=4)
        p.add_argument("--min-comp", type=int, default=2000, help="pontos da amostra; menor e nao citado = ruido")
        p.add_argument("--sobrep-min", type=float, default=0.5)

    p = sub.add_parser("segmentar")
    p.add_argument("projeto")
    p.add_argument("--saida", required=True)
    p.add_argument("--scans", nargs="+")
    p.add_argument("--mesa", action="store_true", help="tira o plano dominante (mesa giratoria)")
    p.add_argument("--mesa-mm", type=float, default=1.0)
    regras(p)
    p.set_defaults(f=cmd_segmentar)
    p = sub.add_parser("aplicar")
    p.add_argument("projeto")
    p.add_argument("componentes")
    p.add_argument("--decisoes", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--min-comp", type=int, default=2000)
    p.add_argument("--sobrep-min", type=float, default=0.5)
    p.add_argument("--perda-max", type=float, default=0.8)
    p.set_defaults(f=cmd_aplicar)
    p = sub.add_parser("verificar")
    p.add_argument("projeto")
    p.add_argument("--saida", required=True)
    regras(p)
    p.set_defaults(f=cmd_verificar)
    a = ap.parse_args()
    baixa_prioridade()
    a.f(a)


if __name__ == "__main__":
    main()
