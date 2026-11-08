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
from concurrent.futures import ProcessPoolExecutor

import gpu
import numpy as np
from comum import configurar_log
from registro_gpu import avaliar_encaixe, fracao_proxima, icp, matriz_informacao, nuvem_voxel
from revo_projeto import RevoScan

# Open3D so onde nao ha equivalente na GPU: FPFH + RANSAC por features (_par) e PoseGraph global_optimization (refinar).
# Importado sob demanda (`_reg()`): o processo principal e os de ICP na GPU nao precisam dele.


_LOGGER = configurar_log(__name__)


def _reg():
    import open3d as o3d

    return o3d, o3d.pipelines.registration


from recursos import baixa_prioridade  # noqa: E402,F401  (reexportado: outros scripts importam daqui)


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
                _LOGGER.info("%s", f"  nuvem {g}: {n:,} pontos")
    return {g: os.path.join(d, g + ".npy") for g in scans}


def _pcd(P, voxel, normais=True):
    """Centroides de voxel (+ normais PCA no raio 3 x voxel) na GPU: `Nuvem` (antes: voxel_down_sample + estimate_normals)."""
    return nuvem_voxel(P, voxel, normais=normais)


# ---------------------------------------------------------------- par a par
ESCALAS = [(5.0, 15.0), (2.0, 4.0), (1.0, 2.0)]  # (voxel, distancia maxima) em mm; a precisao fina fica para `refinar`


def escala():
    """Fator das grades grossas (FPFH 6 mm, ESCALAS, pre-selecao 4 mm). 1 = peca grande (Onix).

    Vem de ALINHAR_ESCALA (herdada pelos processos isolados). `--escala auto` calcula pela diagonal da peca:
    peca de ~50 mm (Projeto 8) -> ~0,2 (FPFH ~1,2 mm, o valor que funcionou); voxel de 6 mm nela falha.
    """
    try:
        return min(1.0, max(0.1, float(os.environ.get("ALINHAR_ESCALA", "1"))))
    except ValueError:
        return 1.0


def escala_auto(caminhos):
    """clip(diagonal / 350 mm, 0,15, 1) com a maior diagonal (percentis 1-99) entre os scans."""
    diag = 0.0
    xp_ = gpu.xp()
    for c in caminhos:
        P = np.load(c)[::10].astype(float)
        if len(P) < 10:
            continue
        Pg = gpu.para_gpu(P)
        lo, hi = gpu.para_cpu(xp_.percentile(Pg, 1, axis=0)), gpu.para_cpu(xp_.percentile(Pg, 99, axis=0))
        diag = max(diag, float(np.linalg.norm(hi - lo)))
    return float(np.clip(diag / 350.0, 0.15, 1.0)), diag


def _par(args):
    """Registra o scan j no referencial do scan i. Devolve os melhores candidatos refinados."""
    i, j, ci, cj, sementes = args[:5]
    semente0 = args[5] if len(args) > 5 else 0
    o3d, reg = _reg()
    Pi, Pj = np.load(ci), np.load(cj)
    esc = escala()
    v0 = 6.0 * esc
    escalas = [(v * esc, d * esc) for v, d in ESCALAS]
    a, b = _pcd(Pi, v0), _pcd(Pj, v0)
    feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=v0 * 5, max_nn=150))
    a3, b3 = a.para_o3d(), b.para_o3d()  # FPFH + RANSAC: Open3D (CPU, sem equivalente na GPU)
    fa, fb = feat(a3), feat(b3)
    nuv = {v: (_pcd(Pi, v), _pcd(Pj, v)) for v, _ in escalas}
    cands = []
    for s in range(semente0, semente0 + sementes):
        o3d.utility.random.seed(s)
        r = reg.registration_ransac_based_on_feature_matching(
            b3,
            a3,
            fb,
            fa,
            True,
            v0 * 2.5,
            reg.TransformationEstimationPointToPoint(False),
            3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(v0 * 2.5)],
            reg.RANSACConvergenceCriteria(800_000, 0.999),
        )
        T = r.transformation
        for v, lim in escalas:
            pa, pb = nuv[v]
            T, _ = icp(pb, pa, T, lim, 60)
        if any(np.linalg.norm(np.linalg.inv(T) @ c["T"] - np.eye(4)) < 0.02 for c in cands):
            continue
        # encaixe nos dois sentidos a 1 mm x escala (o scan menor pode estar todo dentro do maior)
        pa, pb = nuv[escalas[-1][0]]
        tol_enc = max(0.3, esc)
        f1, rmse1, n1 = avaliar_encaixe(pb, pa, tol_enc, T)
        f2, _, _ = avaliar_encaixe(pa, pb, tol_enc, np.linalg.inv(T))
        cands.append(dict(T=T, fit_j_em_i=float(f1), fit_i_em_j=float(f2), rmse=float(rmse1), n_corr=int(n1)))
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
        _LOGGER.info("%s", f"  processo caiu (codigo {p.exitcode}); tentativa {t + 2}/{tentativas}")
    return None


def direto(fn, args, tentativas=3):
    """Roda fn(args) neste processo (trabalho 100% GPU, sem Open3D: nao precisa de isolamento); repete se falhar."""
    for t in range(tentativas):
        try:
            return fn(tuple(args))
        except Exception as e:  # noqa: BLE001 (memoria da GPU, I/O)
            gpu.liberar_memoria()
            _LOGGER.info("%s", f"  {fn.__name__} falhou ({type(e).__name__}: {e}); tentativa {t + 2}/{tentativas}")
    return None


def _extensao(caminho):
    """Area do retangulo dos dois maiores eixos principais (ordem de crescimento: os maiores primeiro)."""
    xp = gpu.xp()
    P = gpu.para_gpu(np.load(caminho)[::20], xp.float64)
    c = P.mean(0)
    Y = P - c
    w, V = np.linalg.eigh(gpu.para_cpu(Y.T @ Y / max(len(P) - 1, 1)))  # covariancia 3x3: autovetores na CPU
    e = gpu.para_cpu(xp.ptp(Y @ xp.asarray(V[:, ::-1]), 0))
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
    usados = {}  # (i, j) -> candidatos, reaproveita entre passadas
    if cache and os.path.exists(cache):
        usados = {
            tuple(map(int, k.split(","))): [dict(c, T=np.array(c["T"])) for c in v]
            for k, v in json.load(open(cache)).items()
        }

    def tenta(i, j):
        if (i, j) in usados:
            return usados[(i, j)]
        r = isolado(_par, (i, j, caminhos[scans[i]], caminhos[scans[j]], 1))
        usados[(i, j)] = [] if r is None else r[2]
        if cache:
            json.dump(
                {f"{a},{b}": [dict(c, T=c["T"].tolist()) for c in v] for (a, b), v in usados.items()}, open(cache, "w")
            )
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
            _LOGGER.info("%s", f"  {scans[j][-6:]}: {len(achados)} ancora(s) {[scans[a][-6:] for a, _ in achados]}")
            if achados:
                poses = [T[a] @ x["T"] for a, x in achados]
                amb = False
                if len(poses) > 1:
                    ang, mm = _dist_pose(poses[0], poses[1])
                    amb = ang > 0.5 or mm > 2.0
                    _LOGGER.info(
                        "%s",
                        f"     concordancia entre ancoras: {ang:.2f} graus, {mm:.2f} mm{('  <-- DISCORDAM' if amb else '')}",
                    )
                T[j] = poses[0]
                info[j] = dict(
                    ancoras=[scans[a] for a, _ in achados],
                    ambigua=amb,
                    fitness=[max(x["fit_j_em_i"], x["fit_i_em_j"]) for _, x in achados],
                )
                progresso = True
        if not progresso:
            break
    return ref, T, info


# ---------------------------------------------------------------- refino global
FINO = [(1.0, 2.0), (0.5, 1.0), (0.3, 0.5)]  # (voxel, distancia maxima) em mm


def _amostra(p, n, semente=0):
    """No maximo n pontos (aleatorio uniforme) mantendo as normais."""
    return p.amostra(n, semente)


def _refinar_par(args):
    """ICP fino do par (i, j) partindo da pose relativa atual. A fonte e uma amostra (o custo do ICP e
    proporcional a ela); o alvo fica inteiro. Devolve (i, j, T_j->i, info, fitness, rmse) ou None."""
    i, j, ci, cj, Tij = args
    Pi, Pj = np.load(ci), np.load(cj)
    nuv = {v: (_pcd(Pi, v), _pcd(Pj, v)) for v, _ in FINO}
    pa, pb = nuv[1.0]
    fit = avaliar_encaixe(_amostra(pb, 200_000), pa, 2.0, Tij)[0]
    if fit < 0.12 and avaliar_encaixe(_amostra(pa, 200_000), pb, 2.0, np.linalg.inv(Tij))[0] < 0.12:
        return None
    T = Tij
    for v, lim in FINO:
        pa, pb = nuv[v]
        T, _ = icp(_amostra(pb, 250_000), pa, T, lim, 25)
    pa, pb = nuv[0.5]
    pbs = _amostra(pb, 250_000)
    info = matriz_informacao(pbs, pa, 1.0, T)
    fit, rmse, _ = avaliar_encaixe(pbs, pa, 1.0, T)
    return i, j, T, np.array(info), float(fit), float(rmse)


def _pares_sobrepostos(scans, caminhos, T, ks, fit_min=0.12, max_por_scan=8):
    """Pre-selecao barata (nuvens de 4 mm): pares cuja sobreposicao ja e visivel com as poses atuais."""
    peq = {}
    for k in ks:
        peq[k] = _pcd(np.load(caminhos[scans[k]]), 4.0 * escala(), normais=False).transformada(T[k])
    cand = []
    for a, i in enumerate(ks):
        for j in ks[a + 1 :]:
            f = max(fracao_proxima(peq[i], peq[j], 6.0 * escala()), fracao_proxima(peq[j], peq[i], 6.0 * escala()))
            if f >= fit_min:
                cand.append((f, i, j))
    cand.sort(reverse=True)
    cont = {k: 0 for k in ks}
    sel = []
    for f, i, j in cand:  # no maximo `max_por_scan` parceiros por scan (os de maior sobreposicao)
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
        _LOGGER.info("%s", f"  refino {it + 1}: {len(pares)} pares sobrepostos")
        tarefas = [(i, j, caminhos[scans[i]], caminhos[scans[j]], np.linalg.inv(T[i]) @ T[j]) for i, j in pares]
        arestas = []
        with ThreadPoolExecutor(simultaneos) as ex:
            for t, r in zip(tarefas, ex.map(lambda t: direto(_refinar_par, t), tarefas)):
                if r is not None and r[4] >= 0.12 and r[5] <= 0.35:
                    arestas.append(r)
                    _LOGGER.info(
                        "%s", f"    {scans[r[0]][-6:]}-{scans[r[1]][-6:]}: encaixe {r[4]:.2f} rmse {r[5]:.3f} mm"
                    )
        o3d, reg = _reg()  # PoseGraph + global_optimization: Open3D (CPU, sem equivalente na GPU)
        pg = reg.PoseGraph()
        idx = {k: m for m, k in enumerate(ks)}
        for k in ks:
            pg.nodes.append(reg.PoseGraphNode(T[k]))
        for i, j, Tij, info, fit, rmse in arestas:
            pg.edges.append(reg.PoseGraphEdge(idx[j], idx[i], Tij, info, uncertain=True))
        reg.global_optimization(
            pg,
            reg.GlobalOptimizationLevenbergMarquardt(),
            reg.GlobalOptimizationConvergenceCriteria(),
            reg.GlobalOptimizationOption(
                max_correspondence_distance=1.0, edge_prune_threshold=0.25, reference_node=idx[ref]
            ),
        )
        antes = {k: T[k].copy() for k in ks}
        for k in ks:
            T[k] = np.array(pg.nodes[idx[k]].pose)
        mov = [float(np.linalg.norm((T[k] @ np.linalg.inv(antes[k]))[:3, 3])) for k in ks]
        _LOGGER.info(
            "%s",
            f"  refino {it + 1}: {len(arestas)} arestas finas; deslocamento medio {np.mean(mov):.3f} mm (max {np.max(mov):.3f})",
        )
    return T


def metricas(scans, caminhos, T, voxel=0.6):
    """Por scan: distancia dos seus pontos a uniao dos outros, so onde ha sobreposicao (< 2 mm)."""
    xp = gpu.xp()
    ks = sorted(T)
    nuv = {k: _pcd(np.load(caminhos[scans[k]]), voxel, normais=False).transformada(T[k]) for k in ks}
    out = {}
    for k in ks:
        outros = xp.concatenate([nuv[o].pts for o in ks if o != k])
        d, j = gpu.vizinho_mais_proximo(outros, nuv[k].pts, dist_max=2.0)
        ov = j >= 0
        dov = gpu.para_cpu(d[ov])
        out[scans[k]] = dict(
            sobreposicao=float(ov.mean()),
            mediana_mm=float(np.median(dov)) if len(dov) else None,
            p95_mm=float(np.percentile(dov, 95)) if len(dov) else None,
            encaixado_01mm=float((dov < 0.1).mean()) if len(dov) else None,
            encaixado_03mm=float((dov < 0.3).mean()) if len(dov) else None,
            pontos=len(d),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--scans", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--ancoras", type=int, default=2)
    ap.add_argument(
        "--escala",
        default="auto",
        help="fator das grades grossas (1 = peca grande; auto = diagonal/350 mm entre 0,15 e 1)",
    )
    a = ap.parse_args()
    proj = os.path.abspath(a.projeto)
    os.makedirs(a.saida, exist_ok=True)
    todos = sorted(os.listdir(os.path.join(proj, "data")))
    scans = a.scans or todos
    from recursos import orcamento_threads

    orcamento_threads(4)
    _LOGGER.info("%s", f"{len(scans)} scans")
    cam = nuvens(proj, scans, a.saida, jobs=4)
    if a.escala == "auto":
        esc, diag = escala_auto([cam[g] for g in scans])
        _LOGGER.info("%s", f"escala automatica {esc:.2f} (diagonal da peca {diag:.0f} mm; FPFH {6 * esc:.2f} mm)")
    else:
        esc = float(a.escala)
    os.environ["ALINHAR_ESCALA"] = str(esc)
    ref, T, info = crescer(
        scans,
        caminhos=cam,
        ref=scans.index(a.ref) if a.ref else None,
        ancoras=a.ancoras,
        cache=os.path.join(a.saida, "pares.json"),
    )
    soltos = [scans[k] for k in range(len(scans)) if k not in T]
    _LOGGER.info("%s %s %s %s", "referencia:", scans[ref], "| sem ligacao:", [g[-6:] for g in soltos])
    json.dump(
        dict(
            referencia=scans[ref],
            sem_ligacao=soltos,
            poses_iniciais={scans[k]: T[k].tolist() for k in T},
            info={scans[k]: v for k, v in info.items()},
        ),
        open(os.path.join(a.saida, "poses_iniciais.json"), "w"),
        indent=1,
    )
    m0 = metricas(scans, cam, dict(T), voxel=1.5)
    T = refinar(scans, cam, T)
    m1 = metricas(scans, cam, T)
    rel = dict(
        referencia=scans[ref],
        sem_ligacao=soltos,
        poses={scans[k]: T[k].tolist() for k in T},
        info={scans[k]: v for k, v in info.items()},
        metricas_inicial=m0,
        metricas_final=m1,
    )
    json.dump(rel, open(os.path.join(a.saida, "alinhamento.json"), "w"), indent=1)
    _LOGGER.info(
        "%s",
        f"{'scan':16s} {'sobrep':>7s} {'mediana':>8s} {'p95':>6s} {'<0,1mm':>7s} {'<0,3mm':>7s}  (mediana antes>depois do refino)",
    )
    for g in m1:
        x, y = m0[g], m1[g]
        f = lambda v: "   -  " if v is None else f"{v:6.3f}"
        _LOGGER.info(
            "%s",
            f"{g:16s} {y['sobreposicao']:7.2f} {f(x['mediana_mm'])}>{f(y['mediana_mm'])} {f(y['p95_mm'])} {f(y['encaixado_01mm'])} {f(y['encaixado_03mm'])}",
        )


if __name__ == "__main__":
    main()
