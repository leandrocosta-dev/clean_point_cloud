"""Limpa os .dph de scans de um objeto sobre mesa (formato antigo): tira o plano da mesa e o que nao e o objeto.

Uso: python pipeline/revo_antigo/limpar_mesa.py <projeto> --saida <pasta nova> --scans id1 id2 [--sem-plano id3] [--altura 2.5]
Por scan: poses (registrada ou .inf, a que concorda com o fuse), nuvem de mundo, plano da mesa por RANSAC (1 mm; so se > 15% dos
pontos); por quadro: pixel com altura > --altura mm sobre o plano, separado por salto de profundidade (2 mm), componentes
dilatadas (3 px) e ficam a maior e as com >= 20% dela. --sem-plano: scans ja segmentados na captura (so as componentes).
Grava os .dph na copia; guarda plano e nuvem limpa em <saida>/_limpeza/. O original nao e tocado.
"""
import argparse
import os
import shutil
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, pontos, profundidade
from mundo_scan import nuvem_mundo, plano_ransac, poses_do_scan


def componentes(d, M, salto=2.0, dilat=3, frac=0.2, escala=0.1):
    z = d.astype(np.float32) * escala
    g = np.zeros_like(M)
    for dy, dx in ((0, 1), (1, 0)):
        q = np.abs(z - np.roll(z, (dy, dx), (0, 1))) > salto
        g |= q | np.roll(q, (-dy, -dx), (0, 1))
    lab, k = ndi.label(ndi.binary_dilation(M & ~g, iterations=dilat) & M)
    if not k:
        return M & False
    sz = ndi.sum(M, lab, range(1, k + 1))
    return np.isin(lab, [j + 1 for j in range(k) if sz[j] >= frac * sz.max()])


def limpar_por_grupo(p, K, po, a, s):
    """Cada grupo de quadros (rastreio continuo) tem a mesa num referencial proprio: plano RANSAC por grupo (sinal: o lado com mais massa)."""
    fs = listar(p); grupos = sorted(set(f[-12:-9] for f in fs)); tot = kept = 0; nuv = []
    for g in grupos:
        fg = [f for f in fs if f[-12:-9] == g]; Wg = []
        for f in fg:
            pts, v, u = pontos(profundidade(f), K, passo=3)
            if len(pts):
                T = po[f][0]; Wg.append(pts @ T[:3, :3].T + T[:3, 3])
        Wg = np.concatenate(Wg); m, inl = plano_ransac(Wg, 1.0); nn = np.linalg.norm(m[:3]); n = m[:3] / nn; off = m[3] / nn
        h = Wg @ n + off
        if np.percentile(h, 95) < -np.percentile(h, 5):
            n, off = -n, -off
        print(f"{s} grupo {g}: {len(fg)} quadros, plano {100*len(inl)/len(Wg):.0f}% dos pontos, altura p5/p50/p95 {np.percentile(Wg@n+off,[5,50,95]).round(0)}", flush=True)
        np.save(os.path.join(a.saida, "_limpeza", f"plano_{s}_{g}.npy"), np.append(n, off))
        for f in fg:
            d = profundidade(f); pts, v, u = pontos(d, K)
            if not len(pts):
                continue
            T = po[f][0]; W = pts @ T[:3, :3].T + T[:3, 3]; ok = (W @ n + off) > a.altura
            M = np.zeros(d.shape, bool); M[v[ok], u[ok]] = True; Mk = componentes(d, M, escala=K[4]); kk = Mk[v, u]
            novo = np.zeros_like(d); novo[v[kk], u[kk]] = d[v[kk], u[kk]]; novo.tofile(f)
            tot += len(v); kept += int(kk.sum()); nuv.append((g, W[kk][::4]))
    np.savez(os.path.join(a.saida, "_limpeza", f"nuvem_{s}.npz"), **{g: np.concatenate([w for gg, w in nuv if gg == g]).astype(np.float32) for g in grupos})
    print(f"{s}: {kept}/{tot} pixels mantidos ({100*kept/tot:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projeto"); ap.add_argument("--saida", required=True)
    ap.add_argument("--scans", nargs="*", default=[]); ap.add_argument("--sem-plano", nargs="*", default=[])
    ap.add_argument("--por-grupo", nargs="*", default=[], help="scans sem pose registrada: plano de mesa por grupo (frame_GGG_*), poses .inf")
    ap.add_argument("--altura", type=float, default=2.5)
    a = ap.parse_args()
    if os.path.exists(a.saida):
        sys.exit("destino ja existe")
    shutil.copytree(a.projeto, a.saida)
    os.makedirs(os.path.join(a.saida, "_limpeza"))
    for s in a.scans + a.sem_plano + a.por_grupo:
        p = os.path.join(a.saida, "data", s); K = intrinsecos(p); po = poses_do_scan(p)
        n = off = None
        if s in a.por_grupo:
            limpar_por_grupo(p, K, po, a, s)
            continue
        if s in a.scans:
            W, _ = nuvem_mundo(p, po, passo=3)
            m, inl = plano_ransac(W, 1.0)
            if len(inl) / len(W) < 0.15:
                sys.exit(f"{s}: plano de mesa nao encontrado ({100*len(inl)/len(W):.1f}% dos pontos)")
            nn = np.linalg.norm(m[:3]); n = m[:3] / nn; off = m[3] / nn
            print(f"{s}: plano {m.round(3)} ({100*len(inl)/len(W):.1f}% dos pontos)", flush=True)
            np.save(os.path.join(a.saida, "_limpeza", f"plano_{s}.npy"), np.append(n, off))
        tot = kept = 0; nuv = []
        for f in listar(p):
            d = profundidade(f); pts, v, u = pontos(d, K)
            if not len(pts):
                continue
            T = po[f][0]; W = pts @ T[:3, :3].T + T[:3, 3]
            ok = np.ones(len(pts), bool) if n is None else (W @ n + off) > a.altura
            M = np.zeros(d.shape, bool); M[v[ok], u[ok]] = True
            Mk = componentes(d, M, escala=K[4]); kk = Mk[v, u]
            novo = np.zeros_like(d); novo[v[kk], u[kk]] = d[v[kk], u[kk]]; novo.tofile(f)
            tot += len(v); kept += int(kk.sum()); nuv.append(W[kk][::4])
        np.save(os.path.join(a.saida, "_limpeza", f"nuvem_{s}.npy"), np.concatenate(nuv).astype(np.float32))
        print(f"{s}: {kept}/{tot} pixels mantidos ({100*kept/tot:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
