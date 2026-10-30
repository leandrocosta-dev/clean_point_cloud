"""Limpeza dos quadros por CONSENSO entre scans alinhados (antes da fusao).

    python pipeline/limpar_por_consenso.py <projeto_limpo> <pasta_segmentos> <segmentos.json> --saida <projeto_novo>

Com todos os scans no mesmo referencial (poses por segmento de `alinhar_direto.py`), cada pixel de cada quadro
vira um ponto 3D comum e so fica se:
  (a) pelo menos `--apoio-outros` OUTROS scans tem pontos no mesmo lugar (tolerancia ~1 mm): fantasmas, reflexos e
      cenario que so um ou dois scans veem saem; e
  (b) esta a ate `--raio` mm do NUCLEO da peca (pontos vistos por >= `--nucleo` scans): chapas, suportes e hastes
      que ficam fixos em relacao a peca (e por isso passam em (a)) mas fora do casco saem.
Quadros que ficam com menos de `--min-pontos` pontos sao apagados. As poses nao mudam; calibracao e
raw_preview.ply sao os mesmos (a calibracao foi estimada com eles).
O projeto de origem nunca e alterado; o destino nao pode existir.
"""
import argparse
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan, SIG_SQL
from alinhar_varios import baixa_prioridade

VOX = 2.0
OFF = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])           # duas grades deslocadas de 1 mm: tolerancia ~1 mm
POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def chaves(P, o, mn):
    k = np.floor((P - mn + o) / VOX).astype(np.int64)
    return (k[:, 0] * 400003 + k[:, 1]) * 400031 + k[:, 2]


def popcount(m):
    return POP[m & 0xFFFF] + POP[(m >> 16) & 0xFFFF] + POP[(m >> 32) & 0xFFFF]


def construir(pasta, segjson, scans, nucleo, raio, saida_npz, vox_nucleo=3.0):
    ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
    pj = json.load(open(segjson))["segmentos"]
    todos = []
    for g in scans:
        X = []
        for s in ind[g]:
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            C = np.load(s["arquivo"]).astype(np.float64)
            X.append(C @ T[:3, :3].T + T[:3, 3])
        todos.append(np.vstack(X))
    mn = np.vstack([x.min(0, keepdims=True) for x in todos]).min(0) - 40
    out = dict(mn=mn)
    for gi, o in enumerate(OFF):
        ks, bs = [], []
        for bit, X in enumerate(todos):
            k = np.unique(chaves(X, o, mn))
            ks.append(k)
            bs.append(np.full(len(k), 1 << bit, dtype=np.int64))
        k = np.concatenate(ks)
        b = np.concatenate(bs)
        ordem = np.argsort(k, kind="stable")
        k, b = k[ordem], b[ordem]
        u, ini = np.unique(k, return_index=True)
        out[f"u{gi}"] = u
        out[f"m{gi}"] = np.bitwise_or.reduceat(b, ini)
    # nucleo: voxels vistos por >= `nucleo` scans, num grid de 3 mm, dilatado por `raio`
    u, m = out["u0"], out["m0"]
    forte = u[popcount(m) >= nucleo]
    kz = forte % 400031
    t = forte // 400031
    ky = t % 400003
    kx = t // 400003
    cen = (np.stack([kx, ky, kz], 1) + 0.5) * VOX + mn
    lo = mn
    dims = np.ceil((np.vstack(todos).max(0) + 40 - lo) / vox_nucleo).astype(int) + 2
    G = np.zeros(dims, bool)
    idx = np.floor((cen - lo) / vox_nucleo).astype(int)
    G[tuple(idx.T)] = True
    G = ndimage.binary_dilation(G, structure=ndimage.generate_binary_structure(3, 3), iterations=max(1, int(round(raio / vox_nucleo))))
    np.savez(saida_npz, mn=mn, u0=out["u0"], m0=out["m0"], u1=out["u1"], m1=out["m1"], lo=lo, vox_nucleo=vox_nucleo,
             nucleo_shape=np.array(G.shape))
    np.save(saida_npz + ".nucleo.npy", np.packbits(G.ravel()))
    return len(scans), int(G.sum()), G.shape


def _carregar(saida_npz):
    z = np.load(saida_npz)
    shp = tuple(z["nucleo_shape"])
    G = np.unpackbits(np.load(saida_npz + ".nucleo.npy"))[: int(np.prod(shp))].astype(bool).reshape(shp)
    return z, G


def _limpar_scan(args):
    proj, destino, g, bit, segs, npz, apoio_outros, min_pontos = args
    baixa_prioridade()
    z, G = _carregar(npz)
    mn, lo, vn = z["mn"], z["lo"], float(z["vox_nucleo"])
    masc = [(z["u0"], z["m0"]), (z["u1"], z["m1"])]
    outros = ~np.int64(1 << bit)
    stats = dict(scan=g, antes=0, depois=0, quadros=0, apagados=0)
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(os.path.join(proj, "data", g), tmp)
        try:
            for m in sorted(set(int(x) for x, p in zip(scan.modo, scan.tem_prof) if p)):
                scan.K[m], _ = scan.calibrar(m)
            db = os.path.join(tmp, g + ".limpo")
            scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
            shutil.copy(scan.db, db)
            cn = sqlite3.connect(db)
            for k, i in enumerate(scan.ids):
                if not scan.tem_prof[k]:
                    continue
                seg = next((s for s in segs if s["k0"] <= k <= s["k1"]), None)
                if seg is None:
                    continue
                T = np.array(seg["T"])
                w, h, mask, zz, quatro = scan.profundidade(i)
                P, pix = scan.pontos(k)
                Pg = P @ T[:3, :3].T + T[:3, 3]
                sup = np.zeros(len(Pg), dtype=np.int64)
                for o, (u, mk) in zip(OFF, masc):
                    kk = chaves(Pg, o, mn)
                    j = np.searchsorted(u, kk)
                    j[j >= len(u)] = len(u) - 1
                    sup |= np.where(u[j] == kk, mk[j], 0)
                ok = popcount(sup & outros) >= apoio_outros
                v = np.floor((Pg - lo) / vn).astype(int)
                dentro = np.all((v >= 0) & (v < np.array(G.shape)), axis=1)
                perto = np.zeros(len(Pg), bool)
                perto[dentro] = G[tuple(v[dentro].T)]
                ok &= perto
                stats["quadros"] += 1
                stats["antes"] += len(pix)
                if ok.sum() < min_pontos:
                    cn.execute("delete from FrameInfo where id=?", (i,))
                    cn.execute("delete from FrameRGBD where id=?", (i,))
                    stats["apagados"] += 1
                    continue
                nova = np.zeros(w * h, bool)
                nova[pix[ok]] = True
                dados = (struct.pack("<III", w, h, int(ok.sum())) + np.packbits(nova, bitorder="little").tobytes()
                         + quatro + zz[ok].astype("<f4").tobytes())
                cn.execute("update FrameRGBD set data=? where id=?", (dados, i))
                stats["depois"] += int(ok.sum())
            cn.commit()
            cn.execute("pragma wal_checkpoint(TRUNCATE)")
            integ = cn.execute("pragma integrity_check").fetchone()[0]
            cn.close()
            for ext in ("-wal", "-shm", "-journal"):
                if os.path.exists(db + ext):
                    os.remove(db + ext)
            with open(db, "r+b") as f:
                f.write(scan.sig)
            os.makedirs(os.path.join(destino, "data", g, "cache"), exist_ok=True)
            shutil.move(db, os.path.join(destino, "data", g, "cache", "frames.dataset"))
            stats["integridade"] = integ
        finally:
            scan.c.close()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto"); ap.add_argument("pasta_segmentos"); ap.add_argument("segmentos")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--apoio-outros", type=int, default=2)
    ap.add_argument("--nucleo", type=int, default=6)
    ap.add_argument("--raio", type=float, default=15.0)
    ap.add_argument("--min-pontos", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    baixa_prioridade()
    proj, saida = os.path.abspath(a.projeto), os.path.abspath(a.saida)
    if os.path.exists(saida) and (not os.path.isdir(saida) or os.listdir(saida)):   # pasta vazia pode ser reaproveitada
        sys.exit(f"destino ja existe: {saida}")
    pj = json.load(open(a.segmentos))
    scans = sorted(g for g in json.load(open(os.path.join(a.pasta_segmentos, "segmentos_indice.json"))) if g in pj["segmentos"])
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ.setdefault("REVO_THREADS", "3")
    t0 = time.time()
    # copia tudo menos o dataset (que e reescrito por scan)
    for g in os.listdir(os.path.join(proj, "data")):
        shutil.copytree(os.path.join(proj, "data", g), os.path.join(saida, "data", g), ignore=shutil.ignore_patterns("frames.dataset"))
    for f in os.listdir(proj):
        if os.path.isfile(os.path.join(proj, f)):
            shutil.copy2(os.path.join(proj, f), os.path.join(saida, f))
    npz = os.path.join(saida, "consenso.npz")
    n, nvox, shp = construir(a.pasta_segmentos, a.segmentos, scans, a.nucleo, a.raio, npz)
    print(f"consenso de {n} scans; nucleo (>= {a.nucleo} scans) dilatado {a.raio:g} mm: {nvox:,} voxels de 3 mm em grade {shp} ({time.time()-t0:.0f}s)", flush=True)
    tarefas = [(proj, saida, g, scans.index(g), pj["segmentos"][g], npz, a.apoio_outros, a.min_pontos) for g in scans]
    rel = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for st in ex.map(_limpar_scan, tarefas):
            rel.append(st)
            print(f"  {st['scan']}: {st['antes']:,} -> {st['depois']:,} pontos ({st['depois']/max(1,st['antes'])*100:.0f}% mantidos), "
                  f"{st['apagados']} de {st['quadros']} quadros apagados, integridade {st.get('integridade')}", flush=True)
    for g in os.listdir(os.path.join(proj, "data")):          # scans fora do consenso (ex.: outra calibracao) vao como estavam
        if g not in scans and not os.path.exists(os.path.join(saida, "data", g, "cache", "frames.dataset")):
            os.makedirs(os.path.join(saida, "data", g, "cache"), exist_ok=True)
            shutil.copy2(os.path.join(proj, "data", g, "cache", "frames.dataset"), os.path.join(saida, "data", g, "cache", "frames.dataset"))
    json.dump(dict(parametros=vars(a), scans=rel), open(os.path.join(saida, "limpeza_consenso.json"), "w"), indent=1)
    tot_a, tot_d = sum(s["antes"] for s in rel), sum(s["depois"] for s in rel)
    print(f"total: {tot_a:,} -> {tot_d:,} pontos ({tot_d/tot_a*100:.0f}% mantidos) em {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
