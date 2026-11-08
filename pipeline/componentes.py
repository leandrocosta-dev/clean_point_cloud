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
from scipy import ndimage
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "revo_frames"))
import revo_projeto as RP
from revo_projeto import RevoScan, gravar_revo, plano_ransac


def baixa_prioridade():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


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
    """Devolve (P_mundo, pix, ok) do quadro k: ok = passa no corte de distancia, no filtro de borda e fora da mesa."""
    r = scan.profundidade(scan.ids[k])
    if r is None:
        return None, None, None
    w, h, mask, z, _ = r
    pix = np.flatnonzero(mask)
    v, u = np.divmod(pix, w)
    fx, fy, cx, cy = scan.K[int(scan.modo[k])]
    zf = z.astype(float)
    Pc = np.c_[(u - cx) * zf / fx, -(v - cy) * zf / fy, -zf]
    ok = (zf >= a.z_min) & (zf <= a.z_max)
    # pixels voadores: normal pela imagem (diferencas finitas) quase perpendicular a visada
    img = np.full((h, w, 3), np.nan)
    img.reshape(-1, 3)[pix] = Pc
    gx = np.full_like(img, np.nan); gy = np.full_like(img, np.nan)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    n = np.cross(gx.reshape(-1, 3)[pix], gy.reshape(-1, 3)[pix])
    nn = np.linalg.norm(n, axis=1)
    vis = Pc / np.linalg.norm(Pc, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.abs((n * vis).sum(1)) / nn
    rasante = np.isfinite(cosang) & (cosang < np.cos(np.radians(a.angulo)))
    ok &= ~rasante
    T = scan.T[k]
    P = Pc @ T[:3, :3].T + T[:3, 3]
    if plano is not None:
        nrm, d = plano
        ok &= (P @ nrm + d) > a.mesa_mm
    return P, pix, ok


# ---------------------------------------------------------------- segmentar
def componentes_do_scan(scan, a, plano):
    """Amostra os quadros (pontos que passam nas regras), voxeliza no mundo e rotula componentes conexos."""
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    amostra, quadro_de = [], []
    for k in ks[::a.amostra_quadros]:
        P, pix, ok = filtro_quadro(scan, k, a, plano)
        if P is None:
            continue
        Q = P[ok][::a.amostra_passo]
        amostra.append(Q.astype(np.float32)); quadro_de.append(np.full(len(Q), k, np.int32))
    A = np.vstack(amostra); Kq = np.concatenate(quadro_de)
    lo = A.min(0) - 2 * a.voxel
    g = np.floor((A - lo) / a.voxel).astype(np.int64)
    dims = g.max(0) + 2
    cont = np.bincount(np.ravel_multi_index(g.T, dims), minlength=int(np.prod(dims))).reshape(dims)
    denso = cont >= a.min_pts_voxel
    lab, nl = ndimage.label(denso, structure=np.ones((3, 3, 3)))
    # voxels esparsos encostados num componente herdam o rotulo (borda da peca); esparsos soltos ficam 0 = ruido
    esparso = (cont > 0) & ~denso
    if esparso.any():
        dist, (ix, iy, iz) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        viz = esparso & (dist <= 1.8)
        lab[viz] = lab[ix[viz], iy[viz], iz[viz]]
    rot = lab[tuple(g.T)]
    return A, Kq, lo, lab, rot, nl


def tabela(A, Kq, rot, nl, plano, voxel_lab):
    linhas = []
    tam = np.bincount(rot, minlength=nl + 1)
    for c in np.argsort(-tam):
        if c == 0 or tam[c] == 0:
            continue
        Q = A[rot == c].astype(float)
        ext = Q.max(0) - Q.min(0)
        ev = np.sort(np.linalg.eigvalsh(np.cov((Q[::max(1, len(Q) // 20000)] - Q.mean(0)).T)))[::-1] if len(Q) > 3 else np.zeros(3)
        planar = float((ev[1] - ev[2]) / ev[0]) if ev[0] > 0 else 0.0
        lin = {"id": int(c), "pontos_amostra": int(tam[c]), "fracao": float(tam[c] / len(A)),
               "caixa_mm": [round(float(x), 1) for x in ext], "centroide": [round(float(x), 1) for x in Q.mean(0)],
               "planaridade": round(planar, 3), "espessura_mm": round(float(np.sqrt(max(ev[2], 0)) * 2), 2),
               "quadros": int(len(np.unique(Kq[rot == c])))}
        if plano is not None:
            h = Q @ plano[0] + plano[1]
            lin["dist_mesa_mm"] = [round(float(np.percentile(h, 5)), 1), round(float(np.percentile(h, 95)), 1)]
        linhas.append(lin)
    return linhas


def render(A, rot, linhas, caminho, max_cores=20):
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
    vistas = [("topo (PCA 1x2)", R[:, 0], R[:, 1], R[:, 2]), ("frente (1x3)", R[:, 0], R[:, 2], -R[:, 1]),
              ("lado (2x3)", R[:, 1], R[:, 2], R[:, 0]), ("perspectiva", pers[:, 0], pers[:, 1], pers[:, 2])]
    fig, ax = plt.subplots(1, 4, figsize=(28, 7.5))
    sub = np.arange(0, len(A), max(1, len(A) // 300000))
    for j, (t, x, y, z) in enumerate(vistas):
        o = sub[np.argsort(z[sub])]
        ax[j].scatter(x[o], y[o], s=0.15, c=cor[o]); ax[j].set_aspect("equal"); ax[j].set_title(t)
        for i, c in enumerate(ids):
            m = rot == c
            if m.any():
                ax[j].text(np.median(x[m]), np.median(y[m]), str(c), fontsize=12, weight="bold",
                           color="k", bbox=dict(facecolor=cmap(i % 20), alpha=0.8, pad=1))
    plt.tight_layout(); plt.savefig(caminho, dpi=55); plt.close()


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
                ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]][::max(1, a.amostra_quadros * 2)]
                B = np.vstack([scan.pontos(k, passo=8)[0] for k in ks])
                nrm, d, frac = plano_ransac(B[::max(1, len(B) // 150000)].astype(float), thr=1.5, it=2000)
                plano = (nrm / np.linalg.norm(nrm), float(d))
            A, Kq, lo, lab, rot, nl = componentes_do_scan(scan, a, plano)
            linhas = tabela(A, Kq, rot, nl, plano, lab)
            d = os.path.join(a.saida, s); os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "rotulos.npz"), lo=lo, voxel=a.voxel, lab=lab.astype(np.int32),
                                plano=np.r_[plano[0], plano[1]] if plano else np.zeros(0))
            json.dump({"scan": s, "parametros": {k: v for k, v in vars(a).items() if k not in ("f", "scans")},
                       "ruido_fracao": float(np.mean(rot == 0)), "componentes": linhas},
                      open(os.path.join(d, "componentes.json"), "w", encoding="utf-8"), indent=1)
            render(A, rot, linhas, os.path.join(d, "componentes.png"))
            scan.c.close()
        resumo[s] = [(l["id"], round(l["fracao"] * 100, 1), l["caixa_mm"]) for l in linhas[:8]]
        print(s[:8], f"{len(linhas)} componentes; ruido solto {np.mean(rot == 0) * 100:.1f}%; maiores (id, %, caixa mm):",
              resumo[s], flush=True)
    print("saida:", os.path.abspath(a.saida))


# ---------------------------------------------------------------- aplicar + verificar
def medir_quadros(scan, mascaras, viz=5):
    """Sobreposicao de cada quadro com os vizinhos: fracao dos pontos mantidos com vizinho < 1 mm, medida SEPARADAMENTE
    contra os `viz` quadros anteriores e os `viz` seguintes; vale o MENOR dos dois. Assim um salto de rastreio (o trecho
    depois da perda encaixa em si mesmo, mas nao no trecho anterior) aparece no quadro da fronteira."""
    ks = sorted(mascaras)
    nuvem = {k: mascaras[k][0][mascaras[k][1]][::4] for k in ks}
    res = {}

    def frac(Q, viz_ks):
        V = [nuvem[j] for j in viz_ks if len(nuvem[j])]
        if not V:
            return None
        return float((cKDTree(np.vstack(V)).query(Q, distance_upper_bound=1.0)[0] < 1.0).mean())
    for i, k in enumerate(ks):
        Q = nuvem[k]
        if len(Q) < 30:
            res[k] = 0.0; continue
        ant = frac(Q, ks[max(0, i - viz):i])
        seg = frac(Q, ks[i + 1:i + 1 + viz])
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
    col = min(6, len(quadros)); lin = (len(quadros) + col - 1) // col
    fig, ax = plt.subplots(lin, col, figsize=(5 * col, 4 * lin), squeeze=False)
    for a_ in ax.ravel():
        a_.axis("off")
    for i, (k, motivo) in enumerate(quadros):
        w, h, mask, z, _ = scan.profundidade(scan.ids[k])
        pix = np.flatnonzero(mask)
        img = np.ones((h * w, 3))
        ok = mascaras[k][1] if k in mascaras else np.zeros(len(pix), bool)
        img[pix[ok]] = 0.55
        img[pix[~ok]] = (0.85, 0.1, 0.1)
        ax.ravel()[i].imshow(img.reshape(h, w, 3)); ax.ravel()[i].set_title(f"quadro {k}: {motivo}", fontsize=11)
    plt.tight_layout(); plt.savefig(caminho, dpi=60); plt.close()


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
            shutil.copytree(src, os.path.join(out, "data", s)); print(s[:8], "sem decisao: copiado"); continue
        z = np.load(os.path.join(comp, s, "rotulos.npz")); lab, lo, vx = z["lab"], z["lo"], float(z["voxel"])
        prm = json.load(open(os.path.join(comp, s, "componentes.json"), encoding="utf-8"))
        P = argparse.Namespace(**prm["parametros"])
        plano = (z["plano"][:3], float(z["plano"][3])) if len(z["plano"]) else None
        tam = {l["id"]: l["pontos_amostra"] for l in prm["componentes"]}
        d = dec[s]
        manter = set(d.get("keep", [])) | {c for c, t in tam.items() if t >= a.min_comp and c not in set(d.get("delete", []))}
        manter -= set(d.get("delete", []))
        lut = np.zeros(lab.max() + 1, bool); lut[[c for c in manter if c <= lab.max()]] = True
        fora = set()                                   # correcao humana: trechos de quadros sem profundidade (ex. rastreio perdido)
        for ini, fim in d.get("remover_quadros", []):
            fora.update(range(int(ini), int(fim) + 1))
        mascaras = {}
        with tempfile.TemporaryDirectory() as tmp:
            scan = abrir_scan(proj, s, tmp)
            estado = {"k": None, "n": -1}
            orig = scan.pontos

            def pontos(k, *x, **kw):
                r = orig(k, *x, **kw); estado["k"] = k; estado["n"] = -1 if r[0] is None else len(r[0]); return r
            scan.pontos = pontos

            def mascara(Pw, seg):
                k = estado["k"]
                if k is not None and len(Pw) == estado["n"] and k in fora:
                    return np.zeros(len(Pw), bool)
                if k is not None and len(Pw) == estado["n"]:
                    _, _, ok = filtro_quadro(scan, k, P, plano)
                else:
                    ok = np.ones(len(Pw), bool)
                gi = np.floor((Pw - lo) / vx).astype(np.int64)
                dentro = np.all((gi >= 0) & (gi < np.array(lab.shape)), axis=1)
                keep = np.zeros(len(Pw), bool)
                keep[dentro] = lut[lab[tuple(gi[dentro].T)]]
                ok = ok & keep
                if k is not None and len(Pw) == estado["n"]:
                    mascaras[k] = (Pw.astype(np.float32), ok)
                return ok
            RP.mascara_manter = mascara
            log = []
            r = RP.limpar_scan(scan, {}, os.path.join(out, "data", s), log=lambda *x: log.append(" ".join(map(str, x))),
                               pixels=False, ilhas=False, min_mantido=0.0, sem_ply_antigos=True, zerar_vazios=True,
                               min_pix_quadro=200)
            sob = medir_quadros(scan, mascaras)
            perda = {k: 1 - float(m[1].mean()) for k, m in mascaras.items() if len(m[1])}
            rev = [(k, f"sobrep {sob[k]:.2f}") for k in sorted(sob) if sob[k] < a.sobrep_min and mascaras[k][1].sum() >= 200] + \
                  [(k, f"perdeu {perda[k] * 100:.0f}%") for k in sorted(perda) if perda[k] > a.perda_max]
            d_out = os.path.join(comp, s)
            folha_revisao(scan, mascaras, rev, os.path.join(d_out, "revisao.png"))
            scan.c.close()
        tot = sum(len(m[1]) for m in mascaras.values()); mant = sum(int(m[1].sum()) for m in mascaras.values())
        ss = np.array(list(sob.values()))
        verif[s] = {"mantidos": list(sorted(manter)), "apagados": sorted(set(tam) - manter),
                    "fracao_removida": round(1 - mant / max(tot, 1), 4),
                    "sobreposicao_mediana": round(float(np.median(ss)), 3) if len(ss) else None,
                    "quadros_revisao": [[int(k), m] for k, m in rev], "confianca": d.get("confianca")}
        json.dump(rev, open(os.path.join(d_out, "revisao.json"), "w"))
        print(s[:8], "\n".join(log), f"| removido {verif[s]['fracao_removida'] * 100:.1f}% | sobreposicao mediana "
              f"{verif[s]['sobreposicao_mediana']} | quadros p/ revisao: {len(rev)}", flush=True)
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
            tam = np.bincount(rot, minlength=nl + 1); tam[0] = 0
            grandes = int((tam >= a.min_comp).sum())
            mascaras = {}
            for k in range(len(scan.ids)):
                if scan.tem_prof[k]:
                    Pw, _ = scan.pontos(k, passo=2)
                    if Pw is not None:
                        mascaras[k] = (Pw.astype(np.float32), np.ones(len(Pw), bool))
            sob = medir_quadros(scan, mascaras)
            scan.c.close()
        ss = np.array(list(sob.values()))
        rel[s] = {"componentes_grandes": grandes, "componentes_total": int((tam > 0).sum()),
                  "maior_fracao": round(float(tam.max() / max(tam.sum(), 1)), 4), "ruido_solto": round(float(np.mean(rot == 0)), 4),
                  "sobreposicao_mediana": round(float(np.median(ss)), 3), "quadros_sobrep_baixa": int((ss < a.sobrep_min).sum())}
        print(s[:8], rel[s], flush=True)
    json.dump(rel, open(os.path.join(a.saida, "verificacao_projeto.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def regras(p):
        p.add_argument("--z-min", type=float, default=100.0, help="mm da camera; mais perto sai")
        p.add_argument("--z-max", type=float, default=600.0, help="mm da camera; mais longe sai")
        p.add_argument("--angulo", type=float, default=80.0, help="graus entre normal e visada acima dos quais o pixel e voador")
        p.add_argument("--voxel", type=float, default=0.75, help="mm da grade dos componentes conexos")
        p.add_argument("--min-pts-voxel", type=int, default=2)
        p.add_argument("--amostra-quadros", type=int, default=2)
        p.add_argument("--amostra-passo", type=int, default=4)
        p.add_argument("--min-comp", type=int, default=2000, help="pontos da amostra; menor e nao citado = ruido")
        p.add_argument("--sobrep-min", type=float, default=0.5)
    p = sub.add_parser("segmentar"); p.add_argument("projeto"); p.add_argument("--saida", required=True)
    p.add_argument("--scans", nargs="+"); p.add_argument("--mesa", action="store_true", help="tira o plano dominante (mesa giratoria)")
    p.add_argument("--mesa-mm", type=float, default=1.0); regras(p); p.set_defaults(f=cmd_segmentar)
    p = sub.add_parser("aplicar"); p.add_argument("projeto"); p.add_argument("componentes")
    p.add_argument("--decisoes", required=True); p.add_argument("--saida", required=True)
    p.add_argument("--min-comp", type=int, default=2000); p.add_argument("--sobrep-min", type=float, default=0.5)
    p.add_argument("--perda-max", type=float, default=0.8); p.set_defaults(f=cmd_aplicar)
    p = sub.add_parser("verificar"); p.add_argument("projeto"); p.add_argument("--saida", required=True); regras(p)
    p.set_defaults(f=cmd_verificar)
    a = ap.parse_args()
    baixa_prioridade()
    a.f(a)


if __name__ == "__main__":
    main()
