"""Alinha varios scans da MESMA peca (cada um no proprio referencial) num referencial unico.

    python pipeline/alinhar_varios.py <projeto_limpo> --saida <pasta_de_trabalho> [--scans a b c ...]

1. Nuvem de cada scan a partir dos proprios quadros limpos (cache em <saida>/nuvens/*.npy).
2. Para cada par: FPFH + RANSAC (varias sementes) + ICP em escalas; o par so vira aresta se o encaixe e o
   erro forem bons nos DOIS sentidos.
3. Grafo de poses (Open3D) com otimizacao global; arestas incoerentes com o resto sao podadas.
4. Refino de cada scan contra a uniao de TODOS os outros (algumas rodadas).
5. Metricas por scan (resíduo contra os demais, fracao encaixada a tolerancias de 0,1/0,3/1 mm) e lista de
   scans sem ligacao confiavel. Grava alinhamento.json (poses 4x4 de cada scan -> referencia) e relatorio.
Nao escreve nada no projeto de origem.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan

reg = o3d.pipelines.registration


def baixa_prioridade():
    """Processo pesado em prioridade ABAIXO do normal: sob carga total o VS Code/Claude Code ficam sem CPU e
    caem (e esta maquina ja mostrou instabilidade de hardware sob carga continua em todos os nucleos)."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


# ---------------------------------------------------------------- nuvens
def _nuvem_scan(args):
    proj, g, destino, alvo = args
    from uniao_scans import nuvem_dos_frames
    with tempfile.TemporaryDirectory() as tmp:
        s = RevoScan(os.path.join(proj, "data", g), tmp)
        try:
            P = nuvem_dos_frames(s, alvo)
        finally:
            s.c.close()
    np.save(os.path.join(destino, g + ".npy"), P.astype(np.float32))
    return g, len(P)


def nuvens(proj, scans, work, alvo=2_500_000, jobs=4):
    d = os.path.join(work, "nuvens")
    os.makedirs(d, exist_ok=True)
    faltam = [g for g in scans if not os.path.exists(os.path.join(d, g + ".npy"))]
    if faltam:
        os.environ.setdefault("REVO_THREADS", "4")
        with ProcessPoolExecutor(jobs, initializer=baixa_prioridade) as ex:
            for g, n in ex.map(_nuvem_scan, [(proj, g, d, alvo) for g in faltam]):
                print(f"  nuvem {g}: {n:,} pontos", flush=True)
    return {g: os.path.join(d, g + ".npy") for g in scans}


def _pcd(P, voxel, normais=True):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    p = p.voxel_down_sample(voxel)
    if normais:
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=50))
    return p


# ---------------------------------------------------------------- par a par
ESCALAS = [(5.0, 15.0), (2.0, 4.0), (1.0, 2.0)]    # (voxel, distancia maxima) em mm; a precisao fina fica para `refinar`


def _par(args):
    """Registra o scan j no referencial do scan i. Devolve os melhores candidatos refinados."""
    i, j, ci, cj, sementes = args[:5]
    semente0 = args[5] if len(args) > 5 else 0
    Pi, Pj = np.load(ci), np.load(cj)
    v0 = 6.0
    a, b = _pcd(Pi, v0), _pcd(Pj, v0)
    feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=v0 * 5, max_nn=150))
    fa, fb = feat(a), feat(b)
    nuv = {v: (_pcd(Pi, v), _pcd(Pj, v)) for v, _ in ESCALAS}
    cands = []
    for s in range(semente0, semente0 + sementes):
        o3d.utility.random.seed(s)
        r = reg.registration_ransac_based_on_feature_matching(
            b, a, fb, fa, True, v0 * 2.5, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(v0 * 2.5)],
            reg.RANSACConvergenceCriteria(800_000, 0.999))
        T = r.transformation
        for v, lim in ESCALAS:
            pa, pb = nuv[v]
            ic = reg.registration_icp(pb, pa, lim, T, reg.TransformationEstimationPointToPlane(),
                                      reg.ICPConvergenceCriteria(max_iteration=60))
            T = ic.transformation
        if any(np.linalg.norm(np.linalg.inv(T) @ c["T"] - np.eye(4)) < 0.02 for c in cands):
            continue
        # encaixe nos dois sentidos a 1 mm (o scan menor pode estar todo dentro do maior)
        pa, pb = nuv[1.0]
        e1 = reg.evaluate_registration(pb, pa, 1.0, T)
        e2 = reg.evaluate_registration(pa, pb, 1.0, np.linalg.inv(T))
        cands.append(dict(T=T, fit_j_em_i=float(e1.fitness), fit_i_em_j=float(e2.fitness), rmse=float(e1.inlier_rmse),
                          n_corr=len(e1.correspondence_set)))
    cands.sort(key=lambda c: -max(c["fit_j_em_i"], c["fit_i_em_j"]))
    return i, j, cands


def boa_aresta(c, fit_min=0.25, rmse_max=1.0):
    return max(c["fit_j_em_i"], c["fit_i_em_j"]) >= fit_min and c["rmse"] <= rmse_max


def _executa(modulo, nome, args, saida):
    baixa_prioridade()
    import importlib
    import pickle
    mod = sys.modules.get(modulo) or importlib.import_module(modulo)
    r = getattr(mod, nome)(args)
    with open(saida, "wb") as h:
        pickle.dump(r, h)


def isolado(fn, args, tentativas=3, timeout=1800):
    """Roda fn(args) num processo proprio: uma queda nativa do Open3D (ja vimos segfault) so perde esta tentativa."""
    import multiprocessing as mp
    import pickle
    ctx = mp.get_context("spawn")
    for t in range(tentativas):
        saida = tempfile.mktemp(suffix=".pkl")
        a = tuple(args) + (t * 100,) if fn.__name__ == "_par" and len(args) == 5 else tuple(args)
        p = ctx.Process(target=_executa, args=(fn.__module__, fn.__name__, a, saida))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
        if p.exitcode == 0 and os.path.exists(saida):
            with open(saida, "rb") as h:
                r = pickle.load(h)
            os.remove(saida)
            return r
        print(f"  processo caiu (codigo {p.exitcode}); tentativa {t + 2}/{tentativas}", flush=True)
    return None


def _extensao(caminho):
    P = np.load(caminho)[::20].astype(float)
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    e = np.ptp((P - c) @ V[:, ::-1], 0)
    return float(e[0] * e[1])


def _dist_pose(T1, T2):
    D = np.linalg.inv(T1) @ T2
    return float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))), float(np.linalg.norm(D[:3, 3]))


def crescer(scans, caminhos, ref=None, ancoras=2, fit_min=0.25, rmse_max=1.0, simultaneos=2, cache=None):
    """Poses iniciais por crescimento: o scan de referencia fica fixo e cada outro scan e registrado contra ate
    `ancoras` scans JA alinhados (os maiores primeiro). Se as ancoras concordam (<0,5 grau, <2 mm) a pose e
    confiavel; se discordam, e marcada ambigua. Quem nao encaixa em nenhuma ancora volta numa passada seguinte.
    Devolve (ref, T, info) com T[k] = pose do scan k no referencial da referencia."""
    from concurrent.futures import ThreadPoolExecutor
    n = len(scans)
    areas = [_extensao(caminhos[g]) for g in scans]
    ref = int(np.argmax(areas)) if ref is None else ref
    tam = np.array(areas)
    T = {ref: np.eye(4)}
    info = {ref: dict(ancoras=[], ambigua=False)}
    usados = {}                                    # (i, j) -> candidatos, reaproveita entre passadas
    if cache and os.path.exists(cache):
        usados = {tuple(map(int, k.split(","))): [dict(c, T=np.array(c["T"])) for c in v]
                  for k, v in json.load(open(cache)).items()}

    def tenta(i, j):
        if (i, j) in usados:
            return usados[(i, j)]
        r = isolado(_par, (i, j, caminhos[scans[i]], caminhos[scans[j]], 1))
        usados[(i, j)] = [] if r is None else r[2]
        if cache:
            json.dump({f"{a},{b}": [dict(c, T=c["T"].tolist()) for c in v] for (a, b), v in usados.items()}, open(cache, "w"))
        return usados[(i, j)]

    for passada in range(3):
        pendentes = [k for k in range(n) if k not in T]
        if not pendentes:
            break
        progresso = False
        # os maiores (mais pontos de sobreposicao provavel) primeiro
        for j in sorted(pendentes, key=lambda k: -tam[k]):
            reg_ord = sorted(T, key=lambda k: -tam[k])
            achados = []
            lote = reg_ord[:]
            while lote and len(achados) < ancoras:
                grupo = [lote.pop(0) for _ in range(min(simultaneos, len(lote)))]
                with ThreadPoolExecutor(len(grupo)) as ex:
                    res = list(ex.map(lambda a: (a, tenta(*sorted((a, j)))), grupo))
                for a, c in res:
                    if not c:
                        continue
                    # candidato: transformacao do scan j para o referencial do scan a
                    i_, j_ = sorted((a, j))
                    cs = c if (i_, j_) == (a, j) else [dict(x, T=np.linalg.inv(x["T"])) for x in c]
                    # em `usados` (i<j) T leva j->i; se a > j precisamos de j->a = inv(T de a->j)
                    for x in cs:
                        if boa_aresta(x, fit_min, rmse_max):
                            achados.append((a, x))
                            break
            print(f"  {scans[j][-6:]}: {len(achados)} ancora(s) {[scans[a][-6:] for a, _ in achados]}", flush=True)
            if achados:
                poses = [T[a] @ x["T"] for a, x in achados]
                amb = False
                if len(poses) > 1:
                    ang, mm = _dist_pose(poses[0], poses[1])
                    amb = ang > 0.5 or mm > 2.0
                    print(f"     concordancia entre ancoras: {ang:.2f} graus, {mm:.2f} mm{'  <-- DISCORDAM' if amb else ''}", flush=True)
                T[j] = poses[0]
                info[j] = dict(ancoras=[scans[a] for a, _ in achados], ambigua=amb,
                               fitness=[max(x["fit_j_em_i"], x["fit_i_em_j"]) for _, x in achados])
                progresso = True
        if not progresso:
            break
    return ref, T, info


# ---------------------------------------------------------------- refino global
FINO = [(1.0, 2.0), (0.5, 1.0), (0.3, 0.5)]     # (voxel, distancia maxima) em mm


def _amostra(p, n, semente=0):
    """No maximo n pontos (aleatorio uniforme) mantendo as normais."""
    if len(p.points) <= n:
        return p
    idx = np.random.default_rng(semente).choice(len(p.points), n, replace=False)
    return p.select_by_index(idx.tolist())


def _refinar_par(args):
    """ICP fino do par (i, j) partindo da pose relativa atual. A fonte e uma amostra (o custo do ICP e
    proporcional a ela); o alvo fica inteiro. Devolve (i, j, T_j->i, info, fitness, rmse) ou None."""
    i, j, ci, cj, Tij = args
    Pi, Pj = np.load(ci), np.load(cj)
    nuv = {v: (_pcd(Pi, v), _pcd(Pj, v)) for v, _ in FINO}
    pa, pb = nuv[1.0]
    ev = reg.evaluate_registration(_amostra(pb, 200_000), pa, 2.0, Tij)
    if ev.fitness < 0.12 and reg.evaluate_registration(_amostra(pa, 200_000), pb, 2.0, np.linalg.inv(Tij)).fitness < 0.12:
        return None
    T = Tij
    for v, lim in FINO:
        pa, pb = nuv[v]
        ic = reg.registration_icp(_amostra(pb, 250_000), pa, lim, T, reg.TransformationEstimationPointToPlane(),
                                  reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=25))
        T = ic.transformation
    pa, pb = nuv[0.5]
    pbs = _amostra(pb, 250_000)
    info = reg.get_information_matrix_from_point_clouds(pbs, pa, 1.0, T)
    e = reg.evaluate_registration(pbs, pa, 1.0, T)
    return i, j, T, np.array(info), float(e.fitness), float(e.inlier_rmse)


def _pares_sobrepostos(scans, caminhos, T, ks, fit_min=0.12, max_por_scan=8):
    """Pre-selecao barata (nuvens de 4 mm): pares cuja sobreposicao ja e visivel com as poses atuais."""
    peq = {}
    for k in ks:
        p = _pcd(np.load(caminhos[scans[k]]), 4.0, normais=False)
        p.transform(T[k])
        peq[k] = p
    cand = []
    for a, i in enumerate(ks):
        for j in ks[a + 1:]:
            d1 = np.asarray(peq[j].compute_point_cloud_distance(peq[i]))
            d2 = np.asarray(peq[i].compute_point_cloud_distance(peq[j]))
            f = max(float((d1 < 6.0).mean()), float((d2 < 6.0).mean()))
            if f >= fit_min:
                cand.append((f, i, j))
    cand.sort(reverse=True)
    cont = {k: 0 for k in ks}
    sel = []
    for f, i, j in cand:                       # no maximo `max_por_scan` parceiros por scan (os de maior sobreposicao)
        if cont[i] < max_por_scan or cont[j] < max_por_scan:
            sel.append((i, j))
            cont[i] += 1
            cont[j] += 1
    return sel


def refinar(scans, caminhos, T, simultaneos=2, iteracoes=2):
    """Refaz os pares sobrepostos em escala fina a partir das poses atuais e reotimiza o grafo de poses."""
    from concurrent.futures import ThreadPoolExecutor
    ks = sorted(T)
    ref = ks[0]
    for it in range(iteracoes):
        pares = _pares_sobrepostos(scans, caminhos, T, ks)
        print(f"  refino {it+1}: {len(pares)} pares sobrepostos", flush=True)
        tarefas = [(i, j, caminhos[scans[i]], caminhos[scans[j]], np.linalg.inv(T[i]) @ T[j]) for i, j in pares]
        arestas = []
        with ThreadPoolExecutor(simultaneos) as ex:
            for t, r in zip(tarefas, ex.map(lambda t: isolado(_refinar_par, t), tarefas)):
                if r is not None and r[4] >= 0.12 and r[5] <= 0.35:
                    arestas.append(r)
                    print(f"    {scans[r[0]][-6:]}-{scans[r[1]][-6:]}: encaixe {r[4]:.2f} rmse {r[5]:.3f} mm", flush=True)
        pg = reg.PoseGraph()
        idx = {k: m for m, k in enumerate(ks)}
        for k in ks:
            pg.nodes.append(reg.PoseGraphNode(T[k]))
        for (i, j, Tij, info, fit, rmse) in arestas:
            pg.edges.append(reg.PoseGraphEdge(idx[j], idx[i], Tij, info, uncertain=True))
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
                                reg.GlobalOptimizationOption(max_correspondence_distance=1.0, edge_prune_threshold=0.25,
                                                             reference_node=idx[ref]))
        antes = {k: T[k].copy() for k in ks}
        for k in ks:
            T[k] = np.array(pg.nodes[idx[k]].pose)
        mov = [float(np.linalg.norm((T[k] @ np.linalg.inv(antes[k]))[:3, 3])) for k in ks]
        print(f"  refino {it+1}: {len(arestas)} arestas finas; deslocamento medio {np.mean(mov):.3f} mm (max {np.max(mov):.3f})", flush=True)
    return T


def metricas(scans, caminhos, T, voxel=0.6):
    """Por scan: distancia dos seus pontos a uniao dos outros, so onde ha sobreposicao (< 2 mm)."""
    ks = sorted(T)
    nuv = {k: _pcd(np.load(caminhos[scans[k]]), voxel, normais=False) for k in ks}
    for k in ks:
        nuv[k].transform(T[k])
    out = {}
    for k in ks:
        outros = o3d.geometry.PointCloud()
        for o in ks:
            if o != k:
                outros += nuv[o]
        d = np.asarray(nuv[k].compute_point_cloud_distance(outros))
        ov = d < 2.0
        out[scans[k]] = dict(
            sobreposicao=float(ov.mean()),
            mediana_mm=float(np.median(d[ov])) if ov.any() else None,
            p95_mm=float(np.percentile(d[ov], 95)) if ov.any() else None,
            encaixado_01mm=float((d[ov] < 0.1).mean()) if ov.any() else None,
            encaixado_03mm=float((d[ov] < 0.3).mean()) if ov.any() else None,
            pontos=len(d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--scans", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--ancoras", type=int, default=2)
    a = ap.parse_args()
    proj = os.path.abspath(a.projeto)
    os.makedirs(a.saida, exist_ok=True)
    todos = sorted(os.listdir(os.path.join(proj, "data")))
    scans = a.scans or todos
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    print(f"{len(scans)} scans", flush=True)
    cam = nuvens(proj, scans, a.saida, jobs=4)
    ref, T, info = crescer(scans, caminhos=cam, ref=scans.index(a.ref) if a.ref else None, ancoras=a.ancoras,
                           cache=os.path.join(a.saida, "pares.json"))
    soltos = [scans[k] for k in range(len(scans)) if k not in T]
    print("referencia:", scans[ref], "| sem ligacao:", [g[-6:] for g in soltos], flush=True)
    json.dump(dict(referencia=scans[ref], sem_ligacao=soltos, poses_iniciais={scans[k]: T[k].tolist() for k in T},
                   info={scans[k]: v for k, v in info.items()}), open(os.path.join(a.saida, "poses_iniciais.json"), "w"), indent=1)
    m0 = metricas(scans, cam, dict(T), voxel=1.5)
    T = refinar(scans, cam, T)
    m1 = metricas(scans, cam, T)
    rel = dict(referencia=scans[ref], sem_ligacao=soltos, poses={scans[k]: T[k].tolist() for k in T},
               info={scans[k]: v for k, v in info.items()}, metricas_inicial=m0, metricas_final=m1)
    json.dump(rel, open(os.path.join(a.saida, "alinhamento.json"), "w"), indent=1)
    print(f"{'scan':16s} {'sobrep':>7s} {'mediana':>8s} {'p95':>6s} {'<0,1mm':>7s} {'<0,3mm':>7s}  (mediana antes>depois do refino)")
    for g in m1:
        x, y = m0[g], m1[g]
        f = lambda v: "   -  " if v is None else f"{v:6.3f}"
        print(f"{g:16s} {y['sobreposicao']:7.2f} {f(x['mediana_mm'])}>{f(y['mediana_mm'])} {f(y['p95_mm'])} {f(y['encaixado_01mm'])} {f(y['encaixado_03mm'])}")


if __name__ == "__main__":
    main()
