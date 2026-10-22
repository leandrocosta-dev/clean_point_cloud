"""Remapeia k0/k1 dos segmentos quando a limpeza apagou quadros (comparando os ids dos dois datasets) e
reconstrói os centroides dos segmentos sobre os quadros limpos.

    python pipeline/remapear_segmentos.py <projeto_antigo> <projeto_novo> <segmentos.json> --saida <pasta>
"""
import argparse, json, os, shutil, sqlite3, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alinhar_segmentos as S
from alinhar_varios import baixa_prioridade


def ids(proj, g, tmp):
    src = os.path.join(proj, "data", g, "cache", "frames.dataset"); d = os.path.join(tmp, g + ".sqlite"); shutil.copy(src, d)
    with open(d, "r+b") as h: h.write(b"SQLite format 3\x00")
    c = sqlite3.connect(d); r = sorted(x[0] for x in c.execute("select id from FrameInfo")); c.close(); os.remove(d); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("antigo"); ap.add_argument("novo"); ap.add_argument("segmentos"); ap.add_argument("--saida", required=True)
    a = ap.parse_args(); baixa_prioridade(); os.environ["OMP_NUM_THREADS"] = "3"
    pj = json.load(open(a.segmentos)); novo = {}; faixas = {}
    with tempfile.TemporaryDirectory() as tmp:
        for g, segs in pj["segmentos"].items():
            A, B = ids(a.antigo, g, tmp), ids(a.novo, g, tmp)
            apagados = np.array([k for k, i in enumerate(A) if i not in set(B)], dtype=int)
            nsegs = []
            for s in sorted(segs, key=lambda x: x["seg"]):
                k0n = s["k0"] - int((apagados < s["k0"]).sum()); k1n = s["k1"] - int((apagados <= s["k1"]).sum())
                nsegs.append(dict(s, k0=k0n, k1=k1n))
            novo[g] = nsegs; faixas[g] = [dict(seg=s["seg"], k0=s["k0"], k1=s["k1"]) for s in nsegs]
            if len(apagados): print(f"  {g}: {len(apagados)} quadros apagados; indices remapeados", flush=True)
    os.makedirs(a.saida, exist_ok=True)
    json.dump(dict(referencia=pj["referencia"], segmentos=novo), open(os.path.join(a.saida, "segmentos_remapeados.json"), "w"), indent=1)
    ind = S.segmentos(os.path.abspath(a.novo), sorted(novo), a.saida, voxel=1.0, faixas=faixas)
    print("segmentos reconstruidos:", sum(len(v) for v in ind.values()))


if __name__ == "__main__":
    main()
