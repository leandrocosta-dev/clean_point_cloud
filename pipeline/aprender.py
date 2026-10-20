"""Aprende com TODOS os projetos: extrai uma amostra leve de cada scan e avalia variantes de pipeline.

Fase 1 - extrair:  python pipeline/aprender.py extrair [--raiz D:/Projects] [--jobs 6]
    Para cada scan com quadros grava aprendizado/cache/<projeto>__<scan>.npz com a nuvem amostrada
    no mundo (A: usada para entender a cena; E: amostra independente para avaliar), as miras da
    camera, a calibracao e o fuse.ply do Revo alinhado aos quadros (gabarito quando alinha bem).
    Pula o que ja existe no cache.

Fase 2 - avaliar:  python pipeline/aprender.py avaliar [--variantes v1 v2 ...] [--saida arquivo.json]
    Roda cada variante de pipeline sobre as amostras e mede contra o gabarito (fuse.ply alinhado):
      recall  = parte do fuse.ply que continua nos pontos mantidos (perder isso estraga a malha)
      lixo    = parte dos pontos mantidos a mais de 3 mm do fuse.ply
    Nao gera nenhum .revo.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "revo_frames"))
CACHE = os.path.join(os.path.dirname(AQUI), "aprendizado", "cache")
FITNESS_GT = 0.85       # fuse.ply alinhado aos quadros com isso ou mais vira gabarito


def listar_scans(raiz):
    for proj in sorted(os.listdir(raiz)):
        d = os.path.join(raiz, proj)
        revos = [f for f in os.listdir(d) if f.endswith(".revo")] if os.path.isdir(d) else []
        if not revos:
            continue
        for n in json.load(open(os.path.join(d, revos[0]), encoding="utf-8"))["nodes"]:
            pasta = os.path.join(d, "data", n["guid"])
            if os.path.isfile(os.path.join(pasta, "cache", "frames.dataset")) and \
                    os.path.isfile(os.path.join(pasta, "raw_preview.ply")):
                yield proj, n["guid"], pasta


def _limitar(X, n):
    """Subamostra deterministica para no maximo n pontos."""
    if len(X) <= n:
        return X
    return X[np.random.default_rng(0).choice(len(X), n, replace=False)]


def _extrair_um(args):
    proj, sid, pasta = args
    saida = os.path.join(CACHE, f"{proj}__{sid}.npz")
    if os.path.exists(saida):
        return proj, sid, "cache"
    os.environ.setdefault("REVO_THREADS", "4")
    from revo_projeto import RevoScan, ler_ply
    import open3d as o3d
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            sc = RevoScan(pasta, tmp)
            erros = {}
            for m in sorted(set(int(x) for x, p in zip(sc.modo, sc.tem_prof) if p)):
                _, e = sc.calibrar(m)
                erros[m] = e
            ks = [k for k in range(len(sc.ids)) if sc.tem_prof[k]]
            sc_ok = ks[:: max(1, len(ks) // 150)]
            A, miras = [], []
            for k in sc_ok:
                P, _ = sc.pontos(k, passo=6)
                if P is None or len(P) < 100:
                    continue
                A.append(P)
                T = sc.T[k]
                dcam = np.median(np.linalg.norm(P - T[:3, 3], axis=1))
                miras.append(T[:3, 3] + T[:3, :3] @ np.array([0, 0, -1.0]) * dcam)
            A = _limitar(np.vstack(A).astype(np.float32), 1_000_000)
            # amostra de avaliacao: outros quadros (deslocados), mais densa
            ke = ks[1:: max(1, len(ks) // 80)]
            E = _limitar(np.vstack([p for p in (sc.pontos(k, passo=4)[0] for k in ke) if p is not None]).astype(np.float32), 1_200_000)
            _, prev = ler_ply(os.path.join(pasta, "raw_preview.ply"))
            X = prev[:, :3].astype(float)
            esp = float(np.median(cKDTree(X).query(X, k=2, workers=-1)[0][:, 1]))
            fit, rmse, F = 0.0, 0.0, np.zeros((0, 3), np.float32)
            fp = os.path.join(pasta, "fuse.ply")
            if os.path.isfile(fp):
                reg = o3d.pipelines.registration
                Fp = o3d.io.read_point_cloud(fp)
                src = Fp.voxel_down_sample(0.8)
                dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(E.astype(float))).voxel_down_sample(0.8)
                src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.4, max_nn=30))
                dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.4, max_nn=30))
                pp = reg.TransformationEstimationPointToPlane()
                ic = reg.registration_icp(src, dst, 2.0, np.eye(4), pp)
                ic = reg.registration_icp(src, dst, 0.8, ic.transformation, pp)
                fit, rmse = float(ic.fitness), float(ic.inlier_rmse)
                Fa = Fp.voxel_down_sample(0.8).transform(ic.transformation)
                F = _limitar(np.asarray(Fa.points, np.float32), 400_000)
            sc.c.close()
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(saida, A=A, E=E, miras=np.array(miras, np.float32), F=F,
                            meta=json.dumps(dict(proj=proj, scan=sid, erro_calib=max(erros.values()), esp=esp,
                                                 fitness=fit, rmse=rmse, quadros=len(ks), t=round(time.time() - t0))))
        return proj, sid, f"ok fitness {fit:.2f}"
    except Exception as e:
        return proj, sid, f"ERRO {type(e).__name__}: {e}"


def cmd_extrair(a):
    itens = list(listar_scans(a.raiz))
    print(f"{len(itens)} scans", flush=True)
    feitos = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for proj, sid, st in ex.map(_extrair_um, itens):
            feitos += 1
            print(f"[{feitos}/{len(itens)}] {proj}/{sid}: {st}", flush=True)


# ------------------------------------------------------------------ avaliacao
def carregar(path):
    z = np.load(path)
    return z["A"], z["E"], z["miras"], z["F"], json.loads(str(z["meta"]))


def variante_mascara(nome, A, miras, E):
    """Devolve (mascara sobre E, info). Uma variante e uma estrategia de pipeline.

    Sintaxe: <base>[,chave=valor...]   base = plano | sem_base | auto_<limite de cruzamento>
    As chaves sao parametros de segmentar_peca (ex.: margem_mm=3, min_tam_frac=0.02, faixa_base=2).
    """
    from revo_projeto import segmentar_peca, mascara_manter
    base, *extras = nome.split(",")
    kw = {k: (v if k == "selecao" else float(v)) for k, v in (x.split("=") for x in extras)}
    if base == "plano":
        modo, kw["cruza_max"] = "plano", 9.0
    elif base == "sem_base":
        modo = "sem"
    else:
        modo, kw["cruza_max"] = "auto", float(base.split("_")[1])
    seg = segmentar_peca(None, amostra=(A.astype(float), miras.astype(float)), transitorio=False,
                         log=lambda *x: None, modo_base=modo, **kw)
    return mascara_manter(E.astype(float), seg), dict(sem_base=seg["sem_base"], cruza=seg["cruza"])


def metricas(F, E, keep):
    arv = cKDTree(F)
    d_todos = arv.query(E, workers=-1)[0]
    # recall relativo: pontos do fuse cobertos por algum ponto mantido / cobertos por algum ponto qualquer
    cob_todos = np.zeros(len(F), bool)
    cob_keep = np.zeros(len(F), bool)
    arvE = cKDTree(E)
    d, _ = arvE.query(F, workers=-1)
    cob_todos = d < 0.6
    arvK = cKDTree(E[keep]) if keep.any() else None
    cob_keep = (arvK.query(F, workers=-1)[0] < 0.6) if arvK is not None else np.zeros(len(F), bool)
    recall = cob_keep.sum() / max(1, cob_todos.sum())
    lixo = float((d_todos[keep] > 3.0).mean()) if keep.any() else 0.0
    return float(recall), lixo, float(keep.mean())


def _aval_um(args):
    f, variantes = args
    A, E, miras, F, meta = carregar(os.path.join(CACHE, f))
    if meta["fitness"] < FITNESS_GT or len(F) == 0:
        return None
    out = {}
    for v in variantes:
        try:
            keep, info = variante_mascara(v, A, miras, E)
            rec, lixo, kf = metricas(F, E, keep)
            out[v] = dict(proj=meta["proj"], scan=meta["scan"], recall=rec, lixo=lixo, mantido=kf, **info)
        except Exception as e:
            out[v] = dict(proj=meta["proj"], scan=meta["scan"], erro=f"{type(e).__name__}: {e}")
    return out


def cmd_avaliar(a):
    arqs = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
    res = {v: [] for v in a.variantes}
    with ProcessPoolExecutor(a.jobs) as ex:
        for out in ex.map(_aval_um, [(f, a.variantes) for f in arqs]):
            if out:
                for v in a.variantes:
                    res[v].append(out[v])
    print(f"scans com gabarito: {len(res[a.variantes[0]])} de {len(arqs)}")
    print(f"{'variante':<46}{'recall med':>11}{'p5':>7}{'<98%':>6}{'<90%':>6}{'lixo':>7}{'err':>5}")
    for v, r in res.items():
        ok = [x for x in r if "recall" in x]
        rc = np.array([x["recall"] for x in ok])
        print(f"{v:<46}{rc.mean()*100:10.2f}%{np.percentile(rc,5)*100:6.0f}%{(rc<0.98).sum():6d}{(rc<0.90).sum():6d}"
              f"{np.mean([x['lixo'] for x in ok])*100:6.2f}%{len(r)-len(ok):5d}")
    if a.saida:
        json.dump(res, open(a.saida, "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extrair")
    e.add_argument("--raiz", default="D:/Projects")
    e.add_argument("--jobs", type=int, default=6)
    v = sub.add_parser("avaliar")
    v.add_argument("--variantes", nargs="+", default=["plano", "sem_base", "auto_0.10"])
    v.add_argument("--saida")
    v.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    {"extrair": cmd_extrair, "avaliar": cmd_avaliar}[a.cmd](a)


if __name__ == "__main__":
    main()
