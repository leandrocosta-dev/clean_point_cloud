"""Alinhamento em nivel de frames (segmentos) de varios scans da mesma peca.

    python pipeline/alinhar_segmentos.py <projeto_limpo> --saida <pasta> --poses <poses_iniciais.json>

Por que: um scan feito a mao deriva ao longo da trajetoria; mesmo com o melhor alinhamento RIGIDO de scan
inteiro, dois scans da mesma superficie ficam a ~0,3 mm (mediana) um do outro. Aqui cada scan e dividido em
segmentos de quadros consecutivos (~50), cada segmento e um corpo rigido e entra num grafo de poses:
  - aresta de odometria entre segmentos consecutivos do mesmo scan (relativa = identidade, forte: o
    rastreador e preciso localmente; so a deriva acumulada e corrigida);
  - arestas de laco entre segmentos sobrepostos (de scans diferentes ou nao vizinhos) por ICP ponto-plano.
Otimiza o grafo (Levenberg-Marquardt), refaz os lacos com as poses novas e repete; a cada rodada mostra a
distancia mediana ao plano nas arestas (a medida que importa para engenharia reversa).
Saida: segmentos.json com, por scan, [{k0, k1, T}] = indices (na ordem dos quadros) e a 4x4 que leva o
segmento ao referencial comum. Nada e escrito no projeto de origem.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan
from alinhar_varios import _pcd, _amostra, isolado, reg, baixa_prioridade

ESC_LACO = [(4.0, 12.0), (2.0, 5.0), (1.0, 2.0)]    # (voxel, distancia maxima) em mm


# ---------------------------------------------------------------- segmentos
def centroides(P, v, minimo=3):
    """Centroide de cada voxel de v mm com pelo menos `minimo` pontos: promedia o ruido de medicao (os pontos
    brutos de um frame tem ~0,3 mm de ruido; um voxel de 1 mm de um segmento tem dezenas de pontos)."""
    P = np.asarray(P, float)
    mn = P.min(0)
    k = np.floor((P - mn) / v).astype(np.int64)
    d = k.max(0) + 1
    key = (k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    C = np.stack([np.bincount(inv, weights=P[:, a]) for a in range(3)], 1) / cnt[:, None]
    return C[cnt >= minimo]


def _segmentos_scan(args):
    proj, g, destino, quadros, voxel = args[:5]
    faixas = args[5] if len(args) > 5 else None        # [{seg,k0,k1}] explicitas (apos apagar quadros)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        s = RevoScan(os.path.join(proj, "data", g), tmp)
        try:
            for m in sorted(set(int(x) for x, p in zip(s.modo, s.tem_prof) if p)):
                s.K[m], _ = s.calibrar(m)
            ks = [k for k in range(len(s.ids)) if s.tem_prof[k]]
            if faixas:
                grupos = [np.array([k for k in range(f["k0"], f["k1"] + 1) if k < len(s.ids) and s.tem_prof[k]]) for f in faixas]
            else:
                nseg = max(1, int(round(len(ks) / quadros)))
                grupos = np.array_split(np.array(ks), nseg)
            for n, gk in enumerate(grupos):
                if faixas:
                    k0, k1 = faixas[n]["k0"], faixas[n]["k1"]
                    if len(gk) == 0:
                        continue
                else:
                    k0 = 0 if n == 0 else int(gk[0])
                    k1 = len(s.ids) - 1 if n + 1 == len(grupos) else int(grupos[n + 1][0]) - 1
                P = np.vstack([s.pontos(int(k), passo=1)[0] for k in gk])      # densidade total
                C = centroides(P, voxel)
                arq = os.path.join(destino, f"{g}_{n:03d}.npy")
                np.save(arq, C.astype(np.float32))
                out.append(dict(scan=g, seg=(faixas[n]["seg"] if faixas else n), k0=k0, k1=k1, arquivo=arq, n=len(C), n_brutos=len(P)))
        finally:
            s.c.close()
    return g, out


def segmentos(proj, scans, work, quadros=50, voxel=1.0, jobs=4, faixas=None):
    d = os.path.join(work, "segmentos")
    os.makedirs(d, exist_ok=True)
    cache = os.path.join(work, "segmentos_indice.json")
    d = os.path.join(work, "segmentos")
    if os.path.exists(cache):
        return json.load(open(cache))
    os.environ.setdefault("REVO_THREADS", "4")
    por_scan = {}
    with ProcessPoolExecutor(jobs, initializer=baixa_prioridade) as ex:
        for g, out in ex.map(_segmentos_scan, [(proj, g, d, quadros, voxel, (faixas or {}).get(g)) for g in scans]):
            por_scan[g] = out
            print(f"  {g}: {len(out)} segmentos ({sum(x['n'] for x in out):,} centroides de {sum(x['n_brutos'] for x in out):,} pontos)", flush=True)
    json.dump(por_scan, open(cache, "w"), indent=1)
    return por_scan


def vizinho(arvore, B, n):
    """cKDTree.query validado: ja devolveu indice > n (134 milhoes numa nuvem de 1,2 milhao). Se o resultado e
    incoerente, repete com 1 thread; se persistir, levanta erro (o processo isolado e repetido por `isolado`)."""
    for workers in (4, 1, 1):
        d, i = arvore.query(B, workers=workers)
        if i.size == 0 or (int(i.max()) < n and bool(np.all(np.isfinite(d)))):
            return d, i
    raise RuntimeError("cKDTree devolveu indices invalidos tres vezes")


# ---------------------------------------------------------------- medida: vies por celula
def _normais(C, raio=3.0):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(C, float)))
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=raio, max_nn=40))
    return p


def residuo_plano(pa, pb, T, tol=2.0, cel=30.0, min_n=100):
    """Medida de ALINHAMENTO insensivel ao ruido: (cobertura, vies mediano |.| por celula de 30 mm, p90 dos vieses,
    mediana de |r|). O residuo de cada ponto (distancia ao plano local do vizinho) tem ruido de superficie; a
    MEDIANA COM SINAL por celula so sobra se as nuvens estiverem deslocadas uma da outra."""
    A = np.asarray(pa.points)
    N = np.asarray(pa.normals)
    B = np.asarray(pb.points) @ T[:3, :3].T + T[:3, 3]
    d, idx = vizinho(cKDTree(A), B, len(A))
    m = d < tol
    if m.sum() < 200:
        return 0.0, None, None, None
    Bm = B[m]
    r = np.einsum("ij,ij->i", Bm - A[idx[m]], N[idx[m]])
    k = np.floor(Bm / cel).astype(np.int64)
    u, inv = np.unique(k, axis=0, return_inverse=True)
    cnt = np.bincount(inv)
    ordem = np.argsort(inv, kind="stable")
    cortes = np.cumsum(cnt)[:-1]
    vies = [abs(float(np.median(r[g]))) for g, c in zip(np.split(ordem, cortes), cnt) if c >= min_n]
    if not vies:
        return float(m.mean()), None, None, float(np.median(np.abs(r)))
    return float(m.mean()), float(np.median(vies)), float(np.percentile(vies, 90)), float(np.median(np.abs(r)))


def _icp_laco(args):
    """ICP robusto (Tukey) multiescala do segmento j sobre o i, partindo de Trel (j -> i)."""
    i, j, ci, cj, Trel = args
    Pi, Pj = np.load(ci), np.load(cj)
    pa0, pb0 = _normais(Pi), _normais(Pj)
    antes = residuo_plano(pa0, pb0, Trel)
    T = Trel
    loss = reg.TukeyLoss(k=0.6)
    for v, lim in ESC_LACO:
        pa = pa0 if v <= 1.0 else _pcd(Pi, v)
        pb = pb0 if v <= 1.0 else _pcd(Pj, v)
        pb = _amostra(pb, 250_000)
        ic = reg.registration_icp(pb, pa, lim, T, reg.TransformationEstimationPointToPlane(loss),
                                  reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=40))
        T = ic.transformation
    depois = residuo_plano(pa0, pb0, T)
    info = reg.get_information_matrix_from_point_clouds(_amostra(pb0, 250_000), pa0, 1.5, T)
    return i, j, T, np.array(info), antes, depois


# ---------------------------------------------------------------- grafo
def candidatos(nos, T, k_por_no=4, sobreposicao_min=0.2, tol=6.0):
    """Pares de segmentos sobrepostos (scans diferentes, ou do mesmo scan mas nao vizinhos), no referencial atual."""
    peq = []
    arv = []
    for n, no in enumerate(nos):
        P = _pcd(np.load(no["arquivo"]), 4.0, normais=False)
        P.transform(T[n])
        X = np.asarray(P.points)
        peq.append(X)
        arv.append(cKDTree(X))
    caixa = [(X.min(0) - tol, X.max(0) + tol) for X in peq]
    cand = []
    for i in range(len(nos)):
        for j in range(i + 1, len(nos)):
            if nos[i]["scan"] == nos[j]["scan"] and abs(nos[i]["seg"] - nos[j]["seg"]) <= 1:
                continue
            if np.any(caixa[i][1] < caixa[j][0]) or np.any(caixa[j][1] < caixa[i][0]):
                continue
            d1 = arv[i].query(peq[j], distance_upper_bound=tol, workers=4)[0]
            d2 = arv[j].query(peq[i], distance_upper_bound=tol, workers=4)[0]
            f = max(float(np.isfinite(d1).mean()), float(np.isfinite(d2).mean()))
            if f >= sobreposicao_min:
                cand.append((f, i, j))
    cand.sort(reverse=True)
    cont = [0] * len(nos)
    sel = []
    for f, i, j in cand:
        if cont[i] < k_por_no or cont[j] < k_por_no:
            sel.append((i, j))
            cont[i] += 1
            cont[j] += 1
    return sel


def arestas_odometria(nos, peso=1.0):
    out = []
    for n in range(len(nos) - 1):
        if nos[n]["scan"] == nos[n + 1]["scan"]:
            pa, pb = _pcd(np.load(nos[n]["arquivo"]), 1.0), _amostra(_pcd(np.load(nos[n + 1]["arquivo"]), 1.0), 100_000)
            info = np.array(reg.get_information_matrix_from_point_clouds(pb, pa, 0.8, np.eye(4))) * peso
            out.append((n, n + 1, np.eye(4), info))
    return out


def consistencia(nos, T, lacos, odo):
    """Quanto o grafo otimizado ainda discorda das medidas: deslocamento RMS (mm) de pontos do segmento j entre a
    pose relativa otimizada e a medida pelo ICP, por laco; e o 'passo' de deriva entre segmentos consecutivos."""
    def desloc(i, j, Tmedido, amostras):
        X = amostras[j]
        Tr = np.linalg.inv(T[i]) @ T[j]
        D = np.linalg.inv(Tmedido) @ Tr
        d = X @ (D[:3, :3] - np.eye(3)).T + D[:3, 3]
        return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    amostras = {}
    for n, no in enumerate(nos):
        X = np.load(no["arquivo"])
        amostras[n] = X[np.random.default_rng(n).choice(len(X), min(3000, len(X)), replace=False)].astype(float)
    dl = [desloc(i, j, Tm, amostras) for (i, j, Tm, info, *_) in lacos]
    do = [desloc(i, j, Tm, amostras) for (i, j, Tm, info) in odo]
    return dict(laco_rms_med=float(np.median(dl)), laco_rms_p90=float(np.percentile(dl, 90)),
                odometria_deriva_med=float(np.median(do)) if do else None,
                odometria_deriva_p90=float(np.percentile(do, 90)) if do else None)


def otimizar(nos, T, odo, lacos, ref):
    """Otimiza o grafo de poses. O Open3D exige que as arestas CERTAS formem um grafo conexo (sem isso o
    resultado e arbitrario: o aviso "Certain-edge subset of PoseGraph is not connected"). A odometria so liga
    segmentos do mesmo scan, entao os melhores lacos (maior cobertura, menor vies) entram como certos ate
    conectar todos os scans; os demais ficam incertos e podem ser podados como outliers."""
    n = len(nos)
    pai = list(range(n))

    def achar(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x
    for (i, j, _, _) in odo:
        pai[achar(i)] = achar(j)
    certos = set()
    ordem = sorted(range(len(lacos)), key=lambda k: -(lacos[k][5][0] - 2.0 * (lacos[k][5][1] or 1.0)))
    for k in ordem:
        i, j = lacos[k][0], lacos[k][1]
        if achar(i) != achar(j):
            pai[achar(i)] = achar(j)
            certos.add(k)
    if len({achar(x) for x in range(n)}) > 1:
        print("  AVISO: o grafo de lacos nao liga todos os segmentos", flush=True)
    pg = reg.PoseGraph()
    for k in range(n):
        pg.nodes.append(reg.PoseGraphNode(T[k]))
    for (i, j, Tr, info) in odo:                      # j -> i (i = n, j = n+1)
        pg.edges.append(reg.PoseGraphEdge(j, i, Tr, info, uncertain=False))
    for k, (i, j, Tr, info, _, _) in enumerate(lacos):
        pg.edges.append(reg.PoseGraphEdge(j, i, Tr, info, uncertain=k not in certos))
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
                            reg.GlobalOptimizationOption(max_correspondence_distance=0.8, edge_prune_threshold=0.25,
                                                         preference_loop_closure=2.0, reference_node=ref))
    return [np.array(nd.pose) for nd in pg.nodes], sum(1 for e in pg.edges if e.uncertain)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--poses", required=True, help="poses_iniciais.json de alinhar_varios.py")
    ap.add_argument("--quadros", type=int, default=50)
    ap.add_argument("--voxel", type=float, default=1.0)
    ap.add_argument("--rodadas", type=int, default=3)
    ap.add_argument("--simultaneos", type=int, default=4)
    ap.add_argument("--peso-odometria", type=float, default=0.3, help="forca da restricao entre segmentos consecutivos (menor = mais deriva permitida)")
    ap.add_argument("--reusar-lacos", action="store_true", help="usa lacos_rodadaN.pkl se existir (so reotimiza)")
    ap.add_argument("--poses-segmentos", help="segmentos.json de uma execucao anterior: comeca dessas poses")
    ap.add_argument("--scans", nargs="+")
    a = ap.parse_args()
    proj = os.path.abspath(a.projeto)
    os.makedirs(a.saida, exist_ok=True)
    ini = json.load(open(a.poses))
    scans = a.scans or sorted(ini["poses_iniciais"])
    ref_scan = ini["referencia"]
    baixa_prioridade()
    os.environ["OMP_NUM_THREADS"] = "3"          # 4 processos x 3 threads = 12 das 24; o resto fica para o sistema
    por_scan = segmentos(proj, scans, a.saida, quadros=a.quadros, voxel=a.voxel)
    nos = [s for g in scans for s in por_scan[g]]
    T = [np.array(ini["poses_iniciais"][s["scan"]]) for s in nos]
    ref = next(n for n, s in enumerate(nos) if s["scan"] == ref_scan and s["seg"] == 0)
    print(f"{len(nos)} segmentos em {len(scans)} scans; referencia {ref_scan} seg 0", flush=True)
    if a.poses_segmentos:
        prev = json.load(open(a.poses_segmentos))["segmentos"]
        T = [np.array(next(x for x in prev[s_["scan"]] if x["seg"] == s_["seg"])["T"]) for s_ in nos]
    odo = arestas_odometria(nos, a.peso_odometria)
    log = []
    for rodada in range(a.rodadas):
        pkl = os.path.join(a.saida, f"lacos_rodada{rodada+1}.pkl")
        if a.reusar_lacos and os.path.exists(pkl):
            import pickle
            lacos = pickle.load(open(pkl, "rb"))
            print(f"rodada {rodada+1}: {len(lacos)} lacos reaproveitados de {pkl}", flush=True)
            T, _ = otimizar(nos, T, odo, lacos, ref)
            print("  consistencia do grafo:", json.dumps(consistencia(nos, T, lacos, odo)), flush=True)
            continue
        par = candidatos(nos, T)
        print(f"rodada {rodada+1}: {len(par)} pares de lacos candidatos", flush=True)
        tarefas = [(i, j, nos[i]["arquivo"], nos[j]["arquivo"], np.linalg.inv(T[i]) @ T[j]) for i, j in par]
        lacos = []
        resultados = []
        t0 = time.time()
        with ThreadPoolExecutor(a.simultaneos) as ex:
            for k, r in enumerate(ex.map(lambda t: isolado(_icp_laco, t), tarefas), 1):
                resultados.append(r)
                if r is not None:
                    i, j, Tr, info, antes, depois = r
                    if depois[1] is not None and depois[0] >= 0.2 and depois[1] <= 0.4:
                        lacos.append(r)
                if k % 25 == 0:
                    print(f"   {k}/{len(tarefas)} ({time.time()-t0:.0f}s), {len(lacos)} lacos aceitos", flush=True)
        # diagnostico de TODOS os pares (inclusive recusados)
        diag = []
        for t, r in zip(tarefas, resultados):
            if r is None:
                diag.append(dict(i=t[0], j=t[1], erro="processo caiu"))
                continue
            i, j, Tr, info, antes, depois = r
            diag.append(dict(a=f"{nos[i]['scan'][-6:]}#{nos[i]['seg']}", b=f"{nos[j]['scan'][-6:]}#{nos[j]['seg']}",
                             antes=antes, depois=depois, aceito=bool(depois[1] is not None and depois[0] >= 0.2 and depois[1] <= 0.4)))
        json.dump(diag, open(os.path.join(a.saida, f"lacos_rodada{rodada+1}.json"), "w"), indent=1)
        ant = [r[4][1] for r in lacos if r[4][1] is not None]
        dep = [r[5][1] for r in lacos if r[5][1] is not None]
        print(f"  lacos aceitos {len(lacos)}/{len(tarefas)} | vies por celula (mediana dos pares): poses atuais {np.median(ant)*1000:.0f} um -> apos ICP dos pares {np.median(dep)*1000:.0f} um", flush=True)
        import pickle
        pickle.dump(lacos, open(os.path.join(a.saida, f"lacos_rodada{rodada+1}.pkl"), "wb"))
        T, n_inc = otimizar(nos, T, odo, lacos, ref)
        cons = consistencia(nos, T, lacos, odo)
        print("  consistencia do grafo:", json.dumps(cons), flush=True)
        print(f"  grafo otimizado ({len(odo)} odometrias + {len(lacos)} lacos)", flush=True)
        log.append(dict(rodada=rodada + 1, lacos=len(lacos), candidatos=len(tarefas), mediana_antes_um=float(np.median(ant) * 1000),
                        mediana_pares_um=float(np.median(dep) * 1000)))
        saida = {}
        for n, s in enumerate(nos):
            saida.setdefault(s["scan"], []).append(dict(k0=s["k0"], k1=s["k1"], seg=s["seg"], T=T[n].tolist()))
        json.dump(dict(referencia=ref_scan, segmentos=saida, log=log), open(os.path.join(a.saida, "segmentos.json"), "w"), indent=1)
    print("fim. log:", json.dumps(log))


if __name__ == "__main__":
    main()
