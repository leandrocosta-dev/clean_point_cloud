"""Segunda passada de limpeza: mantem so os pixels dos quadros perto do fuse.ply ORIGINAL do scan.

    python pipeline/limpar_por_fuse.py <projeto_original> <projeto_limpo> --saida <pasta_nova> [--tol 3]

Quando o fuse.ply do Revo (filtrado pelo proprio Revo) so tem a peca, mas `limpar_raw` deixou
restos (haste de marcadores colada, mao/pinca que segura a peca), tudo que fica a mais de --tol mm
do fuse.ply e apagado dos quadros do projeto limpo. Quadros que ficam com < --min-pix-quadro pixels
perdem a profundidade (como em limpar_por_placa). O original e o limpo nao sao tocados.
O fuse.ply e a referencia: se ele nao tiver a peca inteira, esta passada corta superficie real.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "revo_frames"))
from revo_projeto import RevoScan, ler_ply, gravar_revo


def baixa_prioridade():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def processar(orig, limpo, saida, s, tol, min_pix):
    pasta_o = os.path.join(orig, "data", s)
    pasta_l = os.path.join(limpo, "data", s)
    dest = os.path.join(saida, "data", s)
    fuse = os.path.join(pasta_o, "fuse.ply")
    if not (os.path.isfile(os.path.join(pasta_l, "cache", "frames.dataset")) and os.path.isfile(fuse)):
        shutil.copytree(pasta_l, dest)
        return f"{s}: sem frames/fuse.ply, copiado sem alteracao"
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "o"))
        so = RevoScan(pasta_o, os.path.join(tmp, "o"))
        modos = sorted(set(int(m) for m, p in zip(so.modo, so.tem_prof) if p))
        K = {m: so.calibrar(m)[0] for m in modos}
        so.c.close()
        os.makedirs(os.path.join(tmp, "l"))
        sl = RevoScan(pasta_l, os.path.join(tmp, "l"))
        sl.K = K
        _, F = ler_ply(fuse)
        arv = cKDTree(F[:, :3].astype(float))
        import sqlite3
        db = sl.db + ".novo"
        sl.c.execute("pragma wal_checkpoint(TRUNCATE)")
        shutil.copy(sl.db, db)
        cn = sqlite3.connect(db)
        antes = depois = vazios = 0
        for k, i in enumerate(sl.ids):
            if not sl.tem_prof[k]:
                continue
            w, h, mask, z, quatro = sl.profundidade(i)
            P, pix = sl.pontos(k)
            d = arv.query(P, distance_upper_bound=tol, workers=4)[0]
            ok = np.isfinite(d)
            antes += len(pix)
            if ok.sum() < min_pix:
                cn.execute("delete from FrameRGBD where id=?", (i,))
                vazios += 1
                continue
            nova = np.zeros(w * h, bool)
            nova[pix[ok]] = True
            dados = (struct.pack("<III", w, h, int(ok.sum()))
                     + np.packbits(nova, bitorder="little").tobytes() + quatro
                     + z[ok].astype("<f4").tobytes())
            cn.execute("update FrameRGBD set data=? where id=?", (dados, i))
            depois += int(ok.sum())
        cn.commit()
        integ = cn.execute("pragma integrity_check").fetchone()[0]
        cn.execute("pragma wal_checkpoint(TRUNCATE)")
        cn.close()
        sl.c.close()
        with open(db, "r+b") as f:
            f.write(sl.sig)
        shutil.copytree(pasta_l, dest, ignore=shutil.ignore_patterns("cache"))
        os.makedirs(os.path.join(dest, "cache"))
        shutil.copyfile(db, os.path.join(dest, "cache", "frames.dataset"))
    return (f"{s}: {antes:,} -> {depois:,} pontos ({depois / max(1, antes) * 100:.1f}% mantidos) | "
            f"{vazios} quadros sem peca (profundidade removida) | integridade {integ}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("limpo")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--tol", type=float, default=3.0, help="mm maximos ate o fuse.ply")
    ap.add_argument("--min-pix-quadro", type=int, default=200)
    ap.add_argument("--scans", nargs="+")
    a = ap.parse_args()
    baixa_prioridade()
    orig, limpo, saida = (os.path.abspath(x) for x in (a.original, a.limpo, a.saida))
    if os.path.exists(saida) and os.listdir(saida):
        sys.exit("destino ja existe: " + saida)
    revo = [f for f in os.listdir(limpo) if f.endswith(".revo")][0]
    nos = json.load(open(os.path.join(limpo, revo), encoding="utf-8"))["nodes"]
    todos = [n["guid"] for n in nos]
    scans = a.scans or [n["guid"] for n in nos if n.get("type") == 2]
    os.makedirs(os.path.join(saida, "data"), exist_ok=True)
    for s in todos:
        if s in scans:
            print(processar(orig, limpo, saida, s, a.tol, a.min_pix_quadro), flush=True)
        else:
            shutil.copytree(os.path.join(limpo, "data", s), os.path.join(saida, "data", s))
    nome = os.path.basename(saida)
    gravar_revo(os.path.join(limpo, revo), saida, nome, todos)
    print("saida:", saida)


if __name__ == "__main__":
    main()
