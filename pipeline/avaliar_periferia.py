"""Avalia a regra de periferia da mesa em todos os scans do cache de aprender.py.

Uso: python pipeline/avaliar_periferia.py [--jobs 6] [--saida arquivo.json] [--periferia-mm 5 --banda 12 --alargue 10]

Compara, sobre a amostra E de cada scan, a limpeza padrao (sem base) com a mesma limpeza + periferia.
Nao usa o fuse.ply (ele inclui a mesa). Por scan mede:
  ativa      : a periferia ficou ligada (a trava de cruzamento do plano nao desligou)
  mesa_tirada: fracao a mais removida pela periferia
  corpo_perdido: pontos a mais removidos com altura > 15 mm acima do plano / pontos mantidos pelo padrao nessa altura
  corpo_longe: dos pontos de corpo perdidos, fracao a mais de 1 mm da superficie que ficou (corte real, nao duplicata)
O criterio de seguranca e corpo_perdido ~ 0 e corpo_longe ~ 0.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "revo_frames"))
from aprender import CACHE, carregar


def um(args):
    f, per = args
    from revo_projeto import segmentar_peca, mascara_manter
    A, E, miras, F, meta = carregar(os.path.join(CACHE, f))
    E = E.astype(float)
    nul = lambda *x: None
    try:
        kw = dict(amostra=(A.astype(float), miras.astype(float)), transitorio=False, log=nul, modo_base="sem")
        a = mascara_manter(E, segmentar_peca(None, **kw))
        sb = segmentar_peca(None, periferia_mm=per[0], per_banda=per[1], per_alargue=per[2], **kw)
        b = mascara_manter(E, sb)
        pp = sb["periferia"]
        if pp is None:                       # trava desligou a regra: a mascara e identica ao padrao
            return dict(proj=meta["proj"], scan=meta["scan"], ativa=False, cruza=float(sb["cruza"]),
                        mant_padrao=float(a.mean()), mant_per=float(b.mean()), mesa_tirada=float((a & ~b).mean()),
                        corpo_perdido=0.0, corpo_perdido_n=0, corpo_longe=0.0)
        h = E @ pp["n"] + pp["d"]            # altura sobre o plano da mesa (d_ref, o mesmo que a regra usa)
        perdido = a & ~b
        corpo = a & (h > 15)
        cp = perdido & (h > 15)
        # perdidos dentro da projecao vertical do corpo mantido (h > 15): e corte de verdade, nao mesa em volta
        u = np.cross(pp["n"], [1.0, 0, 0]); u /= np.linalg.norm(u); v = np.cross(pp["n"], u)
        uv = np.c_[E @ u, E @ v]
        ij = np.floor((uv - (uv.min(0) - 5)) / 2.0).astype(int)
        fp = np.zeros(ij.max(0) + 3, bool)
        fp[tuple(ij[b & (h > 15)].T)] = True
        fp = ndimage.binary_closing(fp, iterations=3)
        dentro = (a & ~b) & fp[tuple(ij.T)]
        hp = np.percentile(h[perdido], [5, 50, 95]).round(0).tolist() if perdido.any() else None
        hk = np.percentile(h[a], [5, 50, 95]).round(0).tolist()
        longe = 0.0
        if cp.any():
            dd = cKDTree(E[b]).query(E[cp], workers=-1)[0]
            longe = float((dd > 1.0).mean())
        return dict(proj=meta["proj"], scan=meta["scan"], ativa=sb["periferia"] is not None, cruza=float(sb["cruza"]),
                    mant_padrao=float(a.mean()), mant_per=float(b.mean()), mesa_tirada=float(perdido.mean()),
                    corpo_perdido=float(cp.sum() / max(1, corpo.sum())), corpo_perdido_n=int(cp.sum()), corpo_longe=longe,
                    dentro_n=int(dentro.sum()), dentro_frac=float(dentro.sum() / max(1, (b & (h > 15)).sum())), h_max=float(h[b].max()), ext_padrao=[round(float(x)) for x in np.ptp(E[a], 0)], h_perdido=hp, h_mantido=hk, frac_plano=float(sb['frac_plano']), ext_per=[round(float(x)) for x in np.ptp(E[b], 0)] if b.any() else None)
    except Exception as e:
        return dict(proj=meta["proj"], scan=meta["scan"], erro=f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--saida", default=os.path.join(os.path.dirname(CACHE), "periferia_01.json"))
    ap.add_argument("--periferia-mm", type=float, default=5.0)
    ap.add_argument("--banda", type=float, default=12.0)
    ap.add_argument("--alargue", type=float, default=10.0)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--so", help="so os scans cujo nome de arquivo contem este texto")
    a = ap.parse_args()
    arqs = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
    if a.so:
        arqs = [f for f in arqs if a.so in f]
    if a.limite:
        arqs = arqs[:: max(1, len(arqs) // a.limite)]
    per = (a.periferia_mm, a.banda, a.alargue)
    res = []
    with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:
        for i, r in enumerate(ex.map(um, [(f, per) for f in arqs]), 1):
            res.append(r)
            print(i, len(arqs), r.get("proj"), r.get("scan"), r.get("erro") or f"ativa={r['ativa']} mesa_tirada={r['mesa_tirada']:.2f} dentro={r.get('dentro_n', 0)} ({r.get('dentro_frac', 0)*100:.3f}%) mant={r['mant_per']*100:.0f}% ext {r.get('ext_padrao')}->{r.get('ext_per')}", flush=True)
    json.dump(res, open(a.saida, "w"), indent=1)
    ok = [r for r in res if "erro" in r is False or "erro" not in r]
    at = [r for r in ok if r["ativa"]]
    print(f"\n{len(ok)} scans ok, {len(res) - len(ok)} com erro; periferia ativa em {len(at)}")
    if at:
        cp = np.array([r["dentro_frac"] for r in at])
        print(f"nos ativos: mesa tirada media {np.mean([r['mesa_tirada'] for r in at])*100:.1f}% | perdido dentro do corpo mediana {np.median(cp)*100:.3f}% p95 {np.percentile(cp,95)*100:.3f}% max {cp.max()*100:.3f}% | com dentro > 0,5%: {(cp > 0.005).sum()}")


if __name__ == "__main__":
    main()
