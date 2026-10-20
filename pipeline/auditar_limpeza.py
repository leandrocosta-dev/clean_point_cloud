"""Audita uma limpeza de raw: compara, quadro a quadro, o scan original com o limpo.

Uso:
    python pipeline/auditar_limpeza.py <projeto_original> <projeto_limpo> <scan_id> [--passo 6] [--altura 15]

Mede o que a limpeza REMOVEU, por altura acima do plano da mesa (RANSAC no original):
  - removidos acima de `altura` mm  -> corpo da peca: o esperado e ZERO (e a garantia de nao cortar);
  - removidos entre 4 e `altura` mm -> base/pes da peca ou relevo da mesa: conferir;
  - removidos ate 4 mm             -> mesa.
Tambem mostra a extensao do que ficou e o maior pedaco removido acima de 4 mm (para ver se foi
uma peca de apoio inteira ou uma fatia da peca).
Compara so pixels: nao depende do fuse.ply do Revo.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan, plano_ransac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("limpo")
    ap.add_argument("scan")
    ap.add_argument("--passo", type=int, default=8, help="usa 1 a cada N quadros (os pixels sao todos comparados: passo de pixel desloca a amostra apos a limpeza)")
    ap.add_argument("--altura", type=float, default=15.0)
    ap.add_argument("--referencia", help="projeto limpo de referencia (ex.: limpeza padrao sem remover mesa): conta o que o novo tirou a mais")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t1, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t2, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t3:
        so = RevoScan(os.path.join(a.original, "data", a.scan), t1)
        sl = RevoScan(os.path.join(a.limpo, "data", a.scan), t2)
        assert list(so.ids) == list(sl.ids), "os dois scans tem quadros diferentes"
        for m in sorted(set(int(m) for m, p in zip(so.modo, so.tem_prof) if p)):
            so.calibrar(m)
        sl.K = so.K          # mesma calibracao nos dois lados: a comparacao e pixel a pixel
        ks = [k for k in range(len(so.ids)) if so.tem_prof[k]][:: a.passo]
        sr = None
        if a.referencia:
            sr = RevoScan(os.path.join(a.referencia, "data", a.scan), t3)
            sr.K = so.K
        orig, kept, kref = [], [], []
        for k in ks:
            P, pix = so.pontos(k)
            _, pixl = sl.pontos(k)
            if P is None:
                continue
            orig.append(P)
            if sr is not None:
                _, pr = sr.pontos(k)
                kref.append(np.isin(pix, pr) if pr is not None else np.zeros(len(P), bool))
            kept.append(np.isin(pix, pixl) if pixl is not None else np.zeros(len(P), bool))
        P = np.vstack(orig)
        kept = np.concatenate(kept)
        n, d, frac = plano_ransac(P[:: max(1, len(P) // 60000)], thr=1.0, it=800)
        h = P @ n + d
        rem = ~kept
        print(f"quadros {len(ks)} | pontos {len(P):,} | mantidos {kept.mean()*100:.1f}% | plano da mesa: {frac*100:.0f}% dos pontos")
        for nome, m in ((f"acima de {a.altura:g} mm (corpo)", h > a.altura),
                        (f"4 a {a.altura:g} mm", (h > 4) & (h <= a.altura)),
                        ("ate 4 mm (mesa)", h <= 4)):
            print(f"  {nome:28s} total {m.sum():>10,} | removidos {(m & rem).sum():>10,} ({(m & rem).sum() / max(1, m.sum()) * 100:5.1f}%)")
        k_ = P[kept]
        print("extensao do que ficou (mm):", np.round(np.ptp(k_, 0), 0), "| altura max acima do plano:", round(float((k_ @ n + d).max()), 1))
        # Garantia de nao cortar: pontos removidos que caem DENTRO da projecao vertical do corpo mantido
        u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
        uv = np.c_[P @ u, P @ v]
        lo = uv.min(0) - 5
        ij = np.floor((uv - lo) / 2.0).astype(int)
        fp = np.zeros(ij.max(0) + 3, bool)
        fp[tuple(ij[kept & (h > a.altura)].T)] = True
        fp = ndimage.binary_closing(fp, iterations=3)
        dentro = fp[tuple(ij.T)]
        corte = rem & (h > a.altura) & dentro
        print(f"removidos acima de {a.altura:g} mm DENTRO da projecao do corpo mantido: {corte.sum():,} "
              f"({corte.sum() / max(1, (kept & (h > a.altura)).sum()) * 100:.3f}% do corpo mantido)")
        if sr is not None:
            kr = np.concatenate(kref)
            mais = kr & ~kept
            print(f"vs referencia: tirou a mais {mais.sum():,} pts ({mais.sum() / max(1, kr.sum()) * 100:.1f}% do que a referencia mantinha)")
            for nome, m in ((f"acima de {a.altura:g} mm", h > a.altura), (f"4 a {a.altura:g} mm", (h > 4) & (h <= a.altura)), ("ate 4 mm", h <= 4)):
                print(f"   a mais {nome:20s}: {(mais & m).sum():>10,} | dentro da projecao do corpo: {(mais & m & dentro).sum():>9,}")
        # Os removidos dentro da coluna do corpo sao pixels da propria peca ou de outro objeto? Distancia 3D
        # ate a superficie mantida: <= 1 mm = pixel da peca apagado (corte); longe = objeto separado.
        arv = cKDTree(P[kept][:: max(1, kept.sum() // 3_000_000)])
        def hist(m, rotulo):
            if m.sum() == 0:
                return
            dd = arv.query(P[m][:: max(1, m.sum() // 400_000)], workers=-1)[0]
            print(f"   {rotulo}: n={m.sum():,} | dist. ate o mantido: <=1 mm {np.mean(dd <= 1) * 100:.1f}% | 1-3 mm {np.mean((dd > 1) & (dd <= 3)) * 100:.1f}% | 3-10 mm {np.mean((dd > 3) & (dd <= 10)) * 100:.1f}% | >10 mm {np.mean(dd > 10) * 100:.1f}%")
        hist(corte, "removidos na coluna do corpo")
        if sr is not None:
            hist(mais & (h > a.altura), "a mais que a referencia (acima de 15 mm)")
            hist(mais & (h > 4) & (h <= a.altura), "a mais que a referencia (4 a 15 mm)")
        alto = rem & (h > 4)
        if alto.any():
            R = P[alto]
            print(f"removidos acima de 4 mm: {alto.sum():,} pts | altura media {h[alto].mean():.1f} mm | p99 {np.percentile(h[alto], 99):.1f} mm | max {h[alto].max():.1f} mm")
        so.c.close()
        sl.c.close()


if __name__ == "__main__":
    main()
