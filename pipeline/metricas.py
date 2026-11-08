"""Metricas padrao do pipeline num lugar so (as que o GUIA aceita como validas), com saida JSON.

    python pipeline/metricas.py nuvem A.ply [--referencia R.ply] [--rugosidade] [--componentes] [--saida m.json]
    python pipeline/metricas.py malha M.ply [--referencia R.ply] [--saida m.json]
    python pipeline/metricas.py consenso s1.ply s2.ply s3.ply [--tol 0.4] [--saida m.json]   (nuvens ja alinhadas)

- distancias nos DOIS sentidos (mediana, p90, p95, % acima de 0,3 mm) e cobertura da referencia a 0,3/0,5/1 mm;
- vies normal COM SINAL por celula de 30 mm (mediana por celula; unica metrica de alinhamento valida no guia);
- rugosidade local (RMS do plano em raio r, mediana/p90; a mesma de rugosidade_interna.py, para nuvens e malhas);
- componentes soltos (grade de 0,5 mm) e, para malha, o painel do MeshInspector via MeshLib;
- lixo sem confirmacao: fracao de cada scan sem ponto de OUTRO scan a `--tol` (Projeto 9: v3 <= 0,08%).
Nada disso e exatidao absoluta (exige referencia fisica). Distancia ponto-a-ponto pequena nao prova alinhamento.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recursos import baixa_prioridade, threads_atuais  # noqa: E402


def _w():
    return threads_atuais(12)


def ler_nuvem(caminho, normais=False):
    """(P, N|None) de .ply (nuvem ou malha: vertices) ou .npy."""
    if caminho.endswith(".npy"):
        X = np.load(caminho)
        return np.asarray(X[:, :3], float), (np.asarray(X[:, 3:6], float) if X.shape[1] >= 6 else None)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(caminho)
    P = np.asarray(pc.points, float)
    if not len(P):
        m = o3d.io.read_triangle_mesh(caminho)
        P = np.asarray(m.vertices, float)
        if normais:
            m.compute_vertex_normals()
            return P, np.asarray(m.vertex_normals, float)
        return P, None
    N = np.asarray(pc.normals, float) if pc.has_normals() else None
    if normais and N is None:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=40))
        N = np.asarray(pc.normals, float)
    return P, N


def estat(d, limiar=0.3):
    d = np.asarray(d, float)
    if not len(d):
        return dict(n=0)
    return dict(n=int(len(d)), mediana_mm=float(np.median(d)), p90_mm=float(np.percentile(d, 90)),
                p95_mm=float(np.percentile(d, 95)), acima_limiar_pct=float(100 * np.mean(d > limiar)), limiar_mm=limiar)


def distancias(A, B, limiar=0.3):
    """A->B e B->A (B = referencia) e cobertura de B por A a 0,3/0,5/1 mm."""
    dab, _ = cKDTree(B).query(A, workers=_w())
    dba, _ = cKDTree(A).query(B, workers=_w())
    return dict(a_para_ref=estat(dab, limiar), ref_para_a=estat(dba, limiar),
                cobertura_ref=dict((f"{t:g}mm", float(np.mean(dba <= t))) for t in (0.3, 0.5, 1.0)))


def vies_por_celula(A, B, NB, cel=30.0, tol=0.3, min_n=30):
    """Mediana COM SINAL do residuo normal (A sobre o plano local de B) por celula de `cel` mm.
    Celulas com < min_n pares nao contam. Viés pequeno sozinho pode esconder ruido de sinais opostos."""
    d, j = cKDTree(B).query(A, workers=_w())
    m = d < tol
    if not m.any():
        return dict(celulas=0)
    r = np.einsum("ij,ij->i", A[m] - B[j[m]], NB[j[m]])
    k = np.floor(A[m] / cel).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    ordem = np.argsort(inv, kind="stable")
    grupos = np.split(ordem, np.cumsum(cnt)[:-1])
    med = np.array([float(np.median(r[g])) for g, c in zip(grupos, cnt) if c >= min_n])
    if not len(med):
        return dict(celulas=0, pares=int(m.sum()))
    return dict(celula_mm=cel, tol_mm=tol, celulas=int(len(med)), pares=int(m.sum()),
                vies_abs_mediana_um=float(np.median(np.abs(med)) * 1000), vies_abs_p90_um=float(np.percentile(np.abs(med), 90) * 1000),
                vies_assinado_min_um=float(med.min() * 1000), vies_assinado_max_um=float(med.max() * 1000),
                residuo_abs_mediana_um=float(np.median(np.abs(r)) * 1000))


def rugosidade(P, raio=None, n=30000, min_viz=8, seed=0):
    """RMS (um) do plano local em `raio` mm numa amostra de `n` pontos. Raio padrao = 25x o espacamento
    mediano (2,5 mm no Onix em malha de ~0,1; nas pecas pequenas ~0,3 mm)."""
    if len(P) < min_viz:
        return dict(n=0)
    arv = cKDTree(P)
    if raio is None:
        esp = float(np.median(arv.query(P[:: max(1, len(P) // 20000)], k=2, workers=_w())[0][:, 1]))
        raio = max(0.25, 25 * esp)
    idx = np.random.default_rng(seed).choice(len(P), min(n, len(P)), replace=False)
    viz = arv.query_ball_point(P[idx], raio, workers=_w())
    r = []
    for lista in viz:
        if len(lista) < min_viz:
            continue
        Q = P[lista] - P[lista].mean(0)
        s = np.linalg.svd(Q, compute_uv=False)
        r.append(s[2] / np.sqrt(len(lista)))
    r = np.array(r) * 1000
    if not len(r):
        return dict(n=0, raio_mm=raio)
    return dict(n=int(len(r)), raio_mm=float(raio), mediana_um=float(np.median(r)), p90_um=float(np.percentile(r, 90)),
                p99_um=float(np.percentile(r, 99)))


def componentes(P, voxel=0.5):
    """Componentes 26-conexos em grade de `voxel` mm: total, fracao no maior e pontos fora do maior."""
    if not len(P):
        return dict(total=0)
    k = np.floor((P - P.min(0)) / voxel).astype(np.int64)
    dims = k.max(0) + 1
    if np.prod(dims.astype(float)) > 4e8:
        return componentes(P, voxel * 2)
    G = np.zeros(dims, bool)
    G[tuple(k.T)] = True
    lab, n = ndimage.label(G, structure=np.ones((3, 3, 3)))
    lp = lab[tuple(k.T)]
    tam = np.bincount(lp, minlength=n + 1)[1:]
    return dict(voxel_mm=voxel, total=int(n), fracao_maior=float(tam.max() / len(P)) if n else 0.0,
                pontos_fora_do_maior=int(len(P) - tam.max()) if n else 0)


def lixo_sem_confirmacao(nuvens, tol=0.4):
    """Para cada nuvem (ja alinhadas): fracao de pontos sem ponto de NENHUMA outra nuvem a <= tol."""
    arv = [cKDTree(X) for X in nuvens]
    out = []
    for i, X in enumerate(nuvens):
        sem = np.ones(len(X), bool)
        for j, t in enumerate(arv):
            if j == i or not sem.any():
                continue
            d, _ = t.query(X[sem], distance_upper_bound=tol, workers=_w())
            idx = np.flatnonzero(sem)
            sem[idx[np.isfinite(d)]] = False
        out.append(dict(pontos=int(len(X)), sem_confirmacao_pct=float(100 * sem.mean()) if len(X) else 0.0))
    return dict(tol_mm=tol, por_nuvem=out)


def metricas_malha(caminho):
    try:
        from malha_meshlib import metricas as mm_metricas
        from meshlib import mrmeshpy as mm
        return mm_metricas(mm.loadMesh(caminho))
    except Exception as e:          # meshlib ausente ou arquivo nao e malha
        return dict(erro=f"{type(e).__name__}: {e}")


def medir(caminho, referencia=None, rug=False, comp=False, malha=False):
    P, _ = ler_nuvem(caminho)
    out = dict(arquivo=os.path.abspath(caminho), pontos=int(len(P)),
               extensao_mm=np.ptp(P, axis=0).round(3).tolist() if len(P) else None)
    if malha:
        out["malha"] = metricas_malha(caminho)
    if rug:
        out["rugosidade"] = rugosidade(P)
    if comp:
        out["componentes"] = componentes(P)
    if referencia:
        R, NR = ler_nuvem(referencia, normais=True)
        out["referencia"] = os.path.abspath(referencia)
        out["distancias"] = distancias(P, R)
        out["vies"] = vies_por_celula(P, R, NR)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nome in ("nuvem", "malha"):
        s = sub.add_parser(nome)
        s.add_argument("arquivo")
        s.add_argument("--referencia")
        s.add_argument("--rugosidade", action="store_true")
        s.add_argument("--componentes", action="store_true")
        s.add_argument("--saida")
    s = sub.add_parser("consenso")
    s.add_argument("nuvens", nargs="+")
    s.add_argument("--tol", type=float, default=0.4)
    s.add_argument("--saida")
    a = ap.parse_args()
    baixa_prioridade()
    if a.cmd == "consenso":
        r = lixo_sem_confirmacao([ler_nuvem(c)[0] for c in a.nuvens], a.tol)
        for c, x in zip(a.nuvens, r["por_nuvem"]):
            x["arquivo"] = os.path.abspath(c)
    else:
        r = medir(a.arquivo, a.referencia, a.rugosidade or a.cmd == "malha", a.componentes, malha=a.cmd == "malha")
    txt = json.dumps(r, indent=1, ensure_ascii=False, default=float)
    print(txt)
    if a.saida:
        with open(a.saida, "w", encoding="utf-8") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
