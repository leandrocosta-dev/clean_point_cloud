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

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import gpu  # noqa: E402
import numpy as np
from recursos import baixa_prioridade  # noqa: E402

# Todo o calculo numerico roda na GPU (gpu.py). Fica na CPU so: leitura/gravacao de .ply (Open3D) e
# compute_vertex_normals de malha (operacao de malha do Open3D, sem equivalente na GPU).


def normais_hibridas(P: Any, raio: float, max_nn: int, orientar_para: Any = "centroide") -> np.ndarray:
    """Normais (n, 3) numpy como o KDTreeSearchParamHybrid do Open3D: PCA dos ate `max_nn` vizinhos mais proximos
    DENTRO de `raio` (kNN exato na GPU). O sinal do Open3D era arbitrario; aqui a normal aponta para fora do
    centroide da nuvem. `orientar_para`: "centroide" (padrao), um ponto de vista (3,), None (sinal arbitrario) ou
    normais anteriores (n, 3): o Open3D alinha a normal nova com a que ja existia, aqui tambem.
    Vizinhanca < 3 pontos -> normal (0, 0, 1).

        N = normais_hibridas(P, 4.0, 30)
        N = normais_hibridas(P, 4.0, 30, orientar_para=N_do_arquivo)   # mantem o lado "para fora" ja gravado
    """
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    N = m.zeros((len(Pg), 3), m.float64)
    N[:, 2] = 1.0
    if len(Pg) >= 3:
        k = min(int(max_nn), 64, len(Pg))
        for a, b, d, j in gpu._knn_iter(Pg, Pg, k):  # kNN exato; raio = descartar vizinhos mais longe que ele
            j = m.where(d <= raio, j, -1)
            ok = j >= 0
            cont = ok.sum(1)
            wv = ok[:, :, None].astype(m.float64)
            Y = Pg[m.maximum(j, 0)]
            med = (Y * wv).sum(1) / m.maximum(cont, 1)[:, None]
            Y = (Y - med[:, None, :]) * wv
            M = m.einsum("nki,nkj->nij", Y, Y)
            bom = cont >= 3
            _, v = m.linalg.eigh(m.where(bom[:, None, None], M, m.eye(3)))
            N[a:b] = m.where(bom[:, None], v[:, :, 0], N[a:b])
    if orientar_para is not None:
        if isinstance(orientar_para, str):
            ref = Pg - Pg.mean(0)  # para fora do centroide
        else:
            o = gpu.para_gpu(orientar_para, m.float64)
            ref = o if o.ndim == 2 else Pg - o  # normais anteriores | afastar-se do ponto de vista
        N = m.where(((N * ref).sum(1) < 0)[:, None], -N, N)
    return gpu.para_cpu(N)


def normais_da_nuvem(pc: Any, raio: float, max_nn: int) -> Any:
    """Reestima as normais de uma o3d.PointCloud na GPU (`normais_hibridas`), mantendo o lado das normais que ela ja
    tinha (como o estimate_normals do Open3D) ou, sem normais, apontando para fora do centroide. Grava em `pc.normals`."""
    import open3d as o3d

    P = np.asarray(pc.points, float)
    ant = np.asarray(pc.normals, float) if pc.has_normals() and len(pc.normals) == len(P) else None
    pc.normals = o3d.utility.Vector3dVector(normais_hibridas(P, raio, max_nn, "centroide" if ant is None else ant))
    return pc


def reduzir_com_normais(P: Any, N: Any, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """(centroides, normais medias unitarias) por voxel de `voxel` mm, na GPU (numpy in/out)."""
    m = gpu.xp()
    Pg, Ng = gpu.para_gpu(P, m.float64), gpu.para_gpu(N, m.float64)
    _, inv, cnt = gpu.agrupar(gpu.chaves_voxel(Pg, voxel))
    C = gpu.somar_por_grupo(Pg - Pg.min(0), inv, len(cnt)) / cnt[:, None] + Pg.min(0)
    Nm = gpu.somar_por_grupo(Ng, inv, len(cnt))
    Nm = Nm / m.maximum(m.linalg.norm(Nm, axis=1), 1e-12)[:, None]
    return gpu.para_cpu(C), gpu.para_cpu(Nm)


def icp_robusto(
    origem: Any,
    alvo: Any,
    normais_alvo: Any,
    T0: Any = None,
    tols: tuple[float, ...] = (8, 4, 2, 1.2),
    iteracoes: int = 40,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """ICP ponto-plano com perda de Tukey (escala tol/3) em varias escalas na GPU, no lugar de
    o3d.pipelines.registration.registration_icp. Em cada escala origem e alvo sao decimados a voxel = tol/12
    (a grade de busca tem celula = tol; sem decimar ficaria lenta com nuvens densas). Devolve (T 4x4, info) como
    gpu.icp_ponto_plano; `info["fitness"]` conta so pares com peso > 0 (o Open3D contava todos a <= tol: sai menor)."""
    T = np.eye(4) if T0 is None else np.array(T0, float)
    info = None
    for tol in tols:
        vox = tol / 12.0
        S = reduzir_com_normais(origem, np.zeros((len(origem), 3)), vox)[0] if len(origem) > 2 else origem
        A, NA = reduzir_com_normais(alvo, normais_alvo, vox)
        T, info = gpu.icp_ponto_plano(S, A, NA, T0=T, dist_max=tol, iteracoes=iteracoes, perda="tukey", c=tol / 3.0)
    return T, info


def componentes_grafo(n: int, a: Any, b: Any) -> tuple[int, np.ndarray]:
    """(n_componentes, rotulos (n,) numpy) do grafo nao-direcionado de `n` nos e arestas a[i]-b[i].
    GPU: hook-and-compress (cada raiz maior se liga a menor raiz vizinha com scatter_min, depois salto de ponteiros);
    o csgraph do cupyx exigiria pylibcugraph, ausente aqui."""
    m = gpu.xp()
    A, B = gpu.para_gpu(a, m.int64), gpu.para_gpu(b, m.int64)
    lab = m.arange(n, dtype=m.int64)
    while len(A):
        la, lb = lab[A], lab[B]
        dif = la != lb
        if not bool(dif.any()):
            break
        hi, lo = m.maximum(la, lb)[dif], m.minimum(la, lb)[dif]
        if gpu.em_gpu():
            import cupyx

            cupyx.scatter_min(lab, hi, lo)
        else:
            np.minimum.at(lab, hi, lo)
        while True:  # compressao: todos apontam para a raiz
            nl = lab[lab]
            if bool((nl == lab).all()):
                break
            lab = nl
    _, rot = m.unique(lab, return_inverse=True)
    return int(rot.max()) + 1 if n else 0, gpu.para_cpu(rot.reshape(-1))


def ler_nuvem(caminho: str, normais: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """(P, N|None) de .ply (nuvem ou malha: vertices) ou .npy. Leitura na CPU; normais ausentes: PCA na GPU."""
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
            m.compute_vertex_normals()  # CPU: operacao de malha do Open3D
            return P, np.asarray(m.vertex_normals, float)
        return P, None
    N = np.asarray(pc.normals, float) if pc.has_normals() else None
    if normais and N is None:
        N = normais_hibridas(P, 0.5, 40)
    return P, N


def estat(d: Any, limiar: float = 0.3) -> dict[str, Any]:
    m = gpu.xp()
    d = gpu.para_gpu(d, m.float64)
    if not len(d):
        return dict(n=0)
    return dict(
        n=int(len(d)),
        mediana_mm=float(m.median(d)),
        p90_mm=float(m.percentile(d, 90)),
        p95_mm=float(m.percentile(d, 95)),
        acima_limiar_pct=float(100 * m.mean(d > limiar)),
        limiar_mm=limiar,
    )


def distancias(A: Any, B: Any, limiar: float = 0.3) -> dict[str, Any]:
    """A->B e B->A (B = referencia) e cobertura de B por A a 0,3/0,5/1 mm."""
    m = gpu.xp()
    Ag, Bg = gpu.para_gpu(A, m.float64), gpu.para_gpu(B, m.float64)
    dab, _ = gpu.vizinho_mais_proximo(Bg, Ag)
    dba, _ = gpu.vizinho_mais_proximo(Ag, Bg)
    return dict(
        a_para_ref=estat(dab, limiar),
        ref_para_a=estat(dba, limiar),
        cobertura_ref=dict((f"{t:g}mm", float(m.mean(dba <= t))) for t in (0.3, 0.5, 1.0)),
    )


def _mediana_por_grupo(valores, inverso, n_grupos):
    """Mediana de `valores` em cada grupo de `inverso` (vazio: nan); par: media dos dois centrais (como np.median)."""
    m = gpu.xp()
    ordem = m.lexsort(m.stack([valores, inverso]))
    v = valores[ordem]
    cnt = m.bincount(inverso, minlength=n_grupos)
    ini = m.cumsum(cnt) - cnt
    lo = ini + (cnt - 1) // 2
    hi = ini + cnt // 2
    ult = len(v) - 1
    return m.where(cnt > 0, 0.5 * (v[m.clip(lo, 0, ult)] + v[m.clip(hi, 0, ult)]), m.nan)


def vies_por_celula(A: Any, B: Any, NB: Any, cel: float = 30.0, tol: float = 0.3, min_n: int = 30) -> dict[str, Any]:
    """Mediana COM SINAL do residuo normal (A sobre o plano local de B) por celula de `cel` mm.
    Celulas com < min_n pares nao contam. Viés pequeno sozinho pode esconder ruido de sinais opostos."""
    m = gpu.xp()
    Ag, Bg, NBg = (gpu.para_gpu(x, m.float64) for x in (A, B, NB))
    d, j = gpu.vizinho_mais_proximo(Bg, Ag)
    ok = d < tol
    if not bool(ok.any()):
        return dict(celulas=0)
    Am, jm = Ag[ok], j[ok]
    r = ((Am - Bg[jm]) * NBg[jm]).sum(1)
    org = m.floor(Am.min(0) / cel) * cel  # celulas alinhadas a multiplos de `cel` (como floor(A / cel))
    _, inv, cnt = gpu.agrupar(gpu.chaves_voxel(Am, cel, org))
    med = _mediana_por_grupo(r, inv, len(cnt))[cnt >= min_n]
    pares = int(ok.sum())
    if not len(med):
        return dict(celulas=0, pares=pares)
    ma = m.abs(med)
    return dict(
        celula_mm=cel,
        tol_mm=tol,
        celulas=int(len(med)),
        pares=pares,
        vies_abs_mediana_um=float(m.median(ma) * 1000),
        vies_abs_p90_um=float(m.percentile(ma, 90) * 1000),
        vies_assinado_min_um=float(med.min() * 1000),
        vies_assinado_max_um=float(med.max() * 1000),
        residuo_abs_mediana_um=float(m.median(m.abs(r)) * 1000),
    )


def rugosidade(P: Any, raio: float | None = None, n: int = 30000, min_viz: int = 8, seed: int = 0) -> dict[str, Any]:
    """RMS (um) do plano local em `raio` mm numa amostra de `n` pontos. Raio padrao = 25x o espacamento
    mediano (2,5 mm no Onix em malha de ~0,1; nas pecas pequenas ~0,3 mm).
    Em lote na GPU: covariancia da vizinhanca por soma dos vizinhos no raio (Grade.somas) e autovalores de todas
    as matrizes 3x3 de uma vez; RMS = sqrt(menor autovalor / n_vizinhos) (= s[2]/sqrt(n) do SVD)."""
    if len(P) < min_viz:
        return dict(n=0)
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    if raio is None:
        d, _ = gpu.knn(Pg, Pg[:: max(1, len(Pg) // 20000)], 2)
        raio = max(0.25, 25 * float(m.median(d[:, 1])))
    idx = gpu.para_gpu(np.random.default_rng(seed).choice(len(Pg), min(n, len(Pg)), replace=False))
    Q = Pg[idx]
    g = gpu.Grade(Pg, raio)
    r = []
    for a, b in gpu.blocos(len(Q), bytes_por_item=10 * 4 + 9 * 8 * 4):
        S = g.somas(Q[a:b], gpu=True).astype(m.float64)
        s0 = S[:, 0]
        ok = s0 >= min_viz
        med = S[:, 1:4] / m.maximum(s0, 1)[:, None]
        xx, yy, zz, xy, xz, yz = (S[:, c] for c in range(4, 10))
        C = m.empty((b - a, 3, 3), m.float64)
        C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = xx, yy, zz
        C[:, 0, 1] = C[:, 1, 0] = xy
        C[:, 0, 2] = C[:, 2, 0] = xz
        C[:, 1, 2] = C[:, 2, 1] = yz
        C -= s0[:, None, None] * (med[:, :, None] * med[:, None, :])
        w = m.linalg.eigvalsh(m.where(ok[:, None, None], C, m.eye(3)))[:, 0]
        r.append(m.sqrt(m.maximum(w, 0) / m.maximum(s0, 1))[ok])
    r = m.concatenate(r) * 1000
    if not len(r):
        return dict(n=0, raio_mm=raio)
    return dict(
        n=int(len(r)),
        raio_mm=float(raio),
        mediana_um=float(m.median(r)),
        p90_um=float(m.percentile(r, 90)),
        p99_um=float(m.percentile(r, 99)),
    )


def componentes(P: Any, voxel: float = 0.5) -> dict[str, Any]:
    """Componentes 26-conexos em grade de `voxel` mm: total, fracao no maior e pontos fora do maior (GPU, esparso)."""
    if not len(P):
        return dict(total=0)
    _, tam = gpu.componentes_conexos_voxel(P, voxel, 26)
    n = int(len(tam))
    return dict(
        voxel_mm=voxel,
        total=n,
        fracao_maior=float(tam.max() / len(P)) if n else 0.0,
        pontos_fora_do_maior=int(len(P) - tam.max()) if n else 0,
    )


def lixo_sem_confirmacao(nuvens: Any, tol: float = 0.4) -> dict[str, Any]:
    """Para cada nuvem (ja alinhadas): fracao de pontos sem ponto de NENHUMA outra nuvem a <= tol."""
    m = gpu.xp()
    gp = [gpu.para_gpu(X, m.float64) for X in nuvens]
    ind = [gpu.Indice(X, tol) if len(X) else None for X in gp]
    out = []
    for i, X in enumerate(gp):
        sem = m.ones(len(X), bool)
        for j, t in enumerate(ind):
            if j == i or t is None or not bool(sem.any()):
                continue
            pos = m.flatnonzero(sem)
            _, jj = t.mais_proximo(X[pos])
            sem[pos[jj >= 0]] = False
        out.append(dict(pontos=int(len(X)), sem_confirmacao_pct=float(100 * sem.mean()) if len(X) else 0.0))
    return dict(tol_mm=tol, por_nuvem=out)


def metricas_malha(caminho: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        from malha_meshlib import metricas as mm_metricas
        from meshlib import mrmeshpy as mm

        return mm_metricas(mm.loadMesh(caminho))
    except (ImportError, OSError, RuntimeError, ValueError) as e:  # meshlib ausente ou arquivo nao e malha
        return dict(erro=f"{type(e).__name__}: {e}")


def medir(
    caminho: str | os.PathLike[str], referencia: Any = None, rug: bool = False, comp: bool = False, malha: bool = False
) -> dict[str, Any]:
    P, _ = ler_nuvem(caminho)
    out = dict(
        arquivo=os.path.abspath(caminho),
        pontos=int(len(P)),
        extensao_mm=np.ptp(P, axis=0).round(3).tolist() if len(P) else None,
    )
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


def main() -> None:
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
