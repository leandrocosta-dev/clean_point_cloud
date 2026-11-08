"""Consultas geometricas 3D sobre nuvens/malhas (.ply), .npy e observacoes com origem (scan/quadro/pixel).

    python pipeline/consulta3d.py info A.ply [B.ply ...]
    python pipeline/consulta3d.py caixa A.ply --min X Y Z --max X Y Z [--exportar sel.ply] [--registro insp.json]
    python pipeline/consulta3d.py esfera A.ply --centro X Y Z --raio R [--exportar sel.ply] [--registro insp.json]
    python pipeline/consulta3d.py corte A.ply B.ply --ponto X Y Z --normal NX NY NZ --espessura 0.5 --png corte.png
    python pipeline/consulta3d.py componentes A.ply [--voxel 0.5]
    python pipeline/consulta3d.py comparar ANTES.ply DEPOIS.ply [--tol 0.1] [--exportar-sumiram s.ply]
    python pipeline/consulta3d.py vista A.ply B.ply --saida vista.html [--max-pontos 200000]
    python pipeline/consulta3d.py cobertura_inspecao A.ply --registro insp.json [--regiao 30]

Saida = JSON curto no stdout (contagens, caixas, centroides, ate --amostra coordenadas; nunca milhoes de pontos).
As consultas usam TODOS os pontos na resolucao original; so a `vista` HTML reduz (navegacao).
`observacoes.npy` (fusao_superficies/ciclo_precisao: campos p, v, quadro, pixel) traz a origem de cada ponto:
as selecoes listam os quadros mais frequentes e pares (quadro, pixel) de exemplo.
`--registro` acumula, num JSON, as celulas de `--regiao` mm ja consultadas; `cobertura_inspecao` diz quantas
celulas com dados ja foram vistas e quais faltam. Superficie escondida na vista != superficie nunca medida:
o que nao tem ponto em nenhum arquivo nao aparece em consulta nenhuma (registrar como sem dado, nao inventar).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metricas import ler_nuvem, componentes as _componentes  # noqa: E402
from recursos import baixa_prioridade, orcamento_threads, threads_atuais  # noqa: E402

CORES = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


class Fonte:
    """Um arquivo carregado: pontos, rotulo e (se observacoes) origem quadro/pixel."""

    def __init__(self, caminho):
        self.caminho = os.path.abspath(caminho)
        self.nome = os.path.basename(caminho)
        self.quadro = self.pixel = None
        if caminho.endswith(".npy"):
            X = np.load(caminho, mmap_mode="r")
            if X.dtype.names and "p" in X.dtype.names:
                self.P = np.asarray(X["p"], float)
                if "quadro" in X.dtype.names:
                    self.quadro = np.asarray(X["quadro"])
                if "pixel" in X.dtype.names:
                    self.pixel = np.asarray(X["pixel"])
                # scan = pasta acima de origem/ (layout de fusao_superficies)
                self.nome = os.path.basename(os.path.dirname(os.path.dirname(self.caminho))) or self.nome
                return
        self.P = ler_nuvem(caminho)[0]


def carregar(arquivos):
    return [Fonte(a) for a in arquivos]


def _caixa_json(P):
    if not len(P):
        return None
    return dict(min=P.min(0).round(3).tolist(), max=P.max(0).round(3).tolist(), centroide=P.mean(0).round(3).tolist())


def _amostra(P, n, seed=0):
    if len(P) <= n:
        return P.round(3).tolist()
    return P[np.random.default_rng(seed).choice(len(P), n, replace=False)].round(3).tolist()


def resumo_selecao(fontes, mascaras, amostra=10):
    out = []
    for f, m in zip(fontes, mascaras):
        Q = f.P[m]
        r = dict(arquivo=f.nome, pontos=int(m.sum()), de=int(len(f.P)), caixa=_caixa_json(Q), amostra=_amostra(Q, amostra))
        if f.quadro is not None and m.any():
            q, c = np.unique(f.quadro[m], return_counts=True)
            ordem = np.argsort(-c)[:8]
            r["quadros_mais_frequentes"] = [dict(quadro=int(q[i]), pontos=int(c[i])) for i in ordem]
            idx = np.flatnonzero(m)[:amostra]
            r["origem_exemplos"] = [dict(quadro=int(f.quadro[i]), pixel=int(f.pixel[i]) if f.pixel is not None else None) for i in idx]
        out.append(r)
    return out


def sel_caixa(P, lo, hi):
    return np.all((P >= np.asarray(lo)) & (P <= np.asarray(hi)), axis=1)


def sel_esfera(P, c, r):
    return np.linalg.norm(P - np.asarray(c), axis=1) <= r


def base_plano(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return n, u, np.cross(n, u)


def corte(fontes, ponto, normal, espessura, png=None, amostra=5):
    n, u, v = base_plano(normal)
    p0 = np.asarray(ponto, float)
    mascaras, uv = [], []
    for f in fontes:
        h = (f.P - p0) @ n
        m = np.abs(h) <= espessura / 2
        mascaras.append(m)
        Q = f.P[m] - p0
        uv.append(np.c_[Q @ u, Q @ v])
    res = dict(plano=dict(ponto=p0.tolist(), normal=n.round(6).tolist(), eixo_u=u.round(6).tolist(), eixo_v=v.round(6).tolist(),
                          espessura_mm=espessura), fontes=resumo_selecao(fontes, mascaras, amostra))
    for r, c in zip(res["fontes"], uv):
        r["extensao_no_plano_mm"] = (np.ptp(c, axis=0).round(3).tolist() if len(c) else None)
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
        for i, (f, c) in enumerate(zip(fontes, uv)):
            if len(c):
                ax.scatter(c[:, 0], c[:, 1], s=0.3, c=CORES[i % len(CORES)], label=f"{f.nome} ({len(c):,})", linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm)")
        ax.set_title(f"corte: espessura {espessura:g} mm")
        ax.legend(markerscale=20, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        res["png"] = os.path.abspath(png)
    return res


def comparar(a, b, tol=0.1, regiao=30.0, top=10):
    """Pontos de A sem B a <= tol (sumiram) e de B sem A (apareceram), agrupados por regiao de `regiao` mm."""
    da, _ = cKDTree(b.P).query(a.P, distance_upper_bound=tol, workers=threads_atuais()) if len(b.P) else (np.full(len(a.P), np.inf), None)
    db, _ = cKDTree(a.P).query(b.P, distance_upper_bound=tol, workers=threads_atuais()) if len(a.P) else (np.full(len(b.P), np.inf), None)
    sumiram, apareceram = ~np.isfinite(da), ~np.isfinite(db)

    def regioes(Q):
        if not len(Q):
            return []
        k = np.floor(Q / regiao).astype(np.int64)
        u, inv, c = np.unique(k, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        ordem = np.argsort(-c)[:top]
        return [dict(celula=u[i].tolist(), origem_mm=(u[i] * regiao).tolist(), pontos=int(c[i]),
                     centroide=Q[inv == i].mean(0).round(3).tolist()) for i in ordem]
    return dict(tol_mm=tol, regiao_mm=regiao, antes=a.nome, depois=b.nome, pontos_antes=int(len(a.P)), pontos_depois=int(len(b.P)),
                sumiram=int(sumiram.sum()), apareceram=int(apareceram.sum()),
                sumiram_pct=float(100 * sumiram.mean()) if len(a.P) else 0.0,
                apareceram_pct=float(100 * apareceram.mean()) if len(b.P) else 0.0,
                regioes_sumiram=regioes(a.P[sumiram]), regioes_apareceram=regioes(b.P[apareceram])), sumiram, apareceram


def componentes_detalhe(P, voxel=0.5, top=10):
    from scipy import ndimage
    base = _componentes(P, voxel)
    if not len(P):
        return base
    k = np.floor((P - P.min(0)) / base["voxel_mm"]).astype(np.int64)
    G = np.zeros(k.max(0) + 1, bool)
    G[tuple(k.T)] = True
    lab, n = ndimage.label(G, structure=np.ones((3, 3, 3)))
    lp = lab[tuple(k.T)]
    tam = np.bincount(lp, minlength=n + 1)
    ordem = np.argsort(-tam[1:])[:top] + 1
    base["maiores"] = [dict(id=int(i), pontos=int(tam[i]), caixa=_caixa_json(P[lp == i])) for i in ordem]
    return base


def vista_html(fontes, saida, max_pontos=200_000):
    """HTML autonomo (plotly embutido): reduz so para navegar; uma cor por arquivo/scan."""
    from plotly.offline import get_plotlyjs
    total = sum(len(f.P) for f in fontes)
    traces = []
    for i, f in enumerate(fontes):
        n = max(1, int(max_pontos * len(f.P) / max(1, total)))
        Q = f.P if len(f.P) <= n else f.P[np.random.default_rng(i).choice(len(f.P), n, replace=False)]
        traces.append(dict(type="scatter3d", mode="markers", name=f"{f.nome} ({len(Q):,} de {len(f.P):,})",
                           x=Q[:, 0].round(3).tolist(), y=Q[:, 1].round(3).tolist(), z=Q[:, 2].round(3).tolist(),
                           marker=dict(size=1.5, color=CORES[i % len(CORES)])))
    layout = dict(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=30, b=0),
                  title="consulta3d: vista reduzida (decisoes usam as consultas, nao esta vista)")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Vista 3D</title><script>" + get_plotlyjs()
            + "</script></head><body style='margin:0;background:#fff'><div id='v' style='width:100vw;height:100vh'></div><script>"
            + f"Plotly.newPlot('v',{json.dumps(traces)},{json.dumps(layout)});</script></body></html>")
    with open(saida, "w", encoding="utf-8") as fh:
        fh.write(html)
    return dict(html=os.path.abspath(saida), pontos_exibidos=sum(len(t["x"]) for t in traces), pontos_totais=int(total))


# ---------------------------------------------------------------- registro de cobertura da inspecao
def _celulas(P, regiao):
    if not len(P):
        return set()
    return set(map(tuple, np.unique(np.floor(P / regiao).astype(np.int64), axis=0).tolist()))


def registrar(caminho, fontes, mascaras, regiao, consulta):
    reg = dict(regiao_mm=regiao, consultas=[], celulas_vistas=[])
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as fh:
            reg = json.load(fh)
        if reg.get("regiao_mm") != regiao:
            raise ValueError(f"registro usa regiao {reg.get('regiao_mm')} mm; nao misturar com {regiao}")
    vistas = set(map(tuple, reg["celulas_vistas"]))
    novas = set()
    for f, m in zip(fontes, mascaras):
        novas |= _celulas(f.P[m], regiao)
    reg["consultas"].append(dict(consulta, celulas=len(novas)))
    reg["celulas_vistas"] = sorted(vistas | novas)
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reg, fh)
    os.replace(tmp, caminho)
    return len(novas - vistas)


def cobertura_inspecao(fontes, caminho, regiao, top=10):
    com_dado = set()
    for f in fontes:
        com_dado |= _celulas(f.P, regiao)
    vistas = set()
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as fh:
            reg = json.load(fh)
        if reg.get("regiao_mm") != regiao:
            raise ValueError(f"registro usa regiao {reg.get('regiao_mm')} mm")
        vistas = set(map(tuple, reg["celulas_vistas"]))
    falta = sorted(com_dado - vistas)
    return dict(regiao_mm=regiao, celulas_com_dado=len(com_dado), celulas_inspecionadas=len(com_dado & vistas),
                fracao_inspecionada=len(com_dado & vistas) / max(1, len(com_dado)), faltam=len(falta),
                exemplos_faltando=[dict(celula=list(c), caixa_min=[x * regiao for x in c], caixa_max=[(x + 1) * regiao for x in c])
                                   for c in falta[:top]])


def _exportar(caminho, P):
    import open3d as o3d
    if not len(P):
        return None
    o3d.io.write_point_cloud(caminho, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float))))
    return os.path.abspath(caminho)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def novo(nome, multi=True):
        s = sub.add_parser(nome)
        s.add_argument("arquivos", nargs="+" if multi else 1)
        s.add_argument("--amostra", type=int, default=10, help="coordenadas de exemplo por arquivo")
        return s
    novo("info")
    s = novo("caixa"); s.add_argument("--min", nargs=3, type=float, required=True); s.add_argument("--max", nargs=3, type=float, required=True)
    s.add_argument("--exportar"); s.add_argument("--registro"); s.add_argument("--regiao", type=float, default=30.0)
    s = novo("esfera"); s.add_argument("--centro", nargs=3, type=float, required=True); s.add_argument("--raio", type=float, required=True)
    s.add_argument("--exportar"); s.add_argument("--registro"); s.add_argument("--regiao", type=float, default=30.0)
    s = novo("corte"); s.add_argument("--ponto", nargs=3, type=float, required=True); s.add_argument("--normal", nargs=3, type=float, required=True)
    s.add_argument("--espessura", type=float, default=0.5); s.add_argument("--png"); s.add_argument("--registro")
    s.add_argument("--regiao", type=float, default=30.0)
    s = novo("componentes"); s.add_argument("--voxel", type=float, default=0.5)
    s = sub.add_parser("comparar"); s.add_argument("antes"); s.add_argument("depois"); s.add_argument("--tol", type=float, default=0.1)
    s.add_argument("--regiao", type=float, default=30.0); s.add_argument("--exportar-sumiram"); s.add_argument("--exportar-apareceram")
    s = novo("vista"); s.add_argument("--saida", required=True); s.add_argument("--max-pontos", type=int, default=200_000)
    s = novo("cobertura_inspecao"); s.add_argument("--registro", required=True); s.add_argument("--regiao", type=float, default=30.0)
    a = ap.parse_args(argv)
    baixa_prioridade()
    orcamento_threads(1)

    if a.cmd == "comparar":
        A, B = Fonte(a.antes), Fonte(a.depois)
        r, sumiram, apareceram = comparar(A, B, a.tol, a.regiao)
        if a.exportar_sumiram:
            r["arquivo_sumiram"] = _exportar(a.exportar_sumiram, A.P[sumiram])
        if a.exportar_apareceram:
            r["arquivo_apareceram"] = _exportar(a.exportar_apareceram, B.P[apareceram])
    else:
        fontes = carregar(a.arquivos)
        if a.cmd == "info":
            r = []
            for f in fontes:
                esp = None
                if len(f.P) > 1:
                    Q = f.P[:: max(1, len(f.P) // 20000)]
                    esp = float(np.median(cKDTree(f.P).query(Q, k=2, workers=threads_atuais())[0][:, 1]))
                r.append(dict(arquivo=f.nome, caminho=f.caminho, pontos=int(len(f.P)), caixa=_caixa_json(f.P),
                              extensao_mm=np.ptp(f.P, axis=0).round(3).tolist() if len(f.P) else None,
                              espacamento_mediano_mm=esp, tem_origem=f.quadro is not None))
        elif a.cmd in ("caixa", "esfera"):
            ms = [sel_caixa(f.P, a.min, a.max) if a.cmd == "caixa" else sel_esfera(f.P, a.centro, a.raio) for f in fontes]
            r = dict(consulta=a.cmd, fontes=resumo_selecao(fontes, ms, a.amostra))
            if a.exportar:
                r["exportado"] = _exportar(a.exportar, np.vstack([f.P[m] for f, m in zip(fontes, ms)]))
            if a.registro:
                cons = dict(tipo=a.cmd, **({"min": a.min, "max": a.max} if a.cmd == "caixa" else {"centro": a.centro, "raio": a.raio}))
                r["celulas_novas_inspecionadas"] = registrar(a.registro, fontes, ms, a.regiao, cons)
        elif a.cmd == "corte":
            r = corte(fontes, a.ponto, a.normal, a.espessura, a.png, a.amostra)
            if a.registro:
                n, _, _ = base_plano(a.normal)
                ms = [np.abs((f.P - np.asarray(a.ponto)) @ n) <= a.espessura / 2 for f in fontes]
                r["celulas_novas_inspecionadas"] = registrar(a.registro, fontes, ms, a.regiao,
                                                             dict(tipo="corte", ponto=a.ponto, normal=a.normal, espessura=a.espessura))
        elif a.cmd == "componentes":
            r = [dict(arquivo=f.nome, **componentes_detalhe(f.P, a.voxel)) for f in fontes]
        elif a.cmd == "vista":
            r = vista_html(fontes, a.saida, a.max_pontos)
        elif a.cmd == "cobertura_inspecao":
            r = cobertura_inspecao(fontes, a.registro, a.regiao)
    print(json.dumps(r, indent=1, ensure_ascii=False, default=float))
    return r


if __name__ == "__main__":
    main()
