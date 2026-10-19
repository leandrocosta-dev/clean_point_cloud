"""Gera uma nuvem sintetica que imita uma captura com problemas:
base (mesa giratoria), ruido fino, outliers, ilhas flutuantes e um trecho
deslocado por perda de rastreamento. Tambem salva a peca "verdade" para
medir o resultado.
"""
import sys
import numpy as np
import open3d as o3d

rng = np.random.default_rng(7)
S = 0.3  # espacamento alvo em mm


def grid_rect(a, b, step):
    u = np.arange(0, a, step)
    v = np.arange(0, b, step)
    U, V = np.meshgrid(u, v)
    return np.c_[U.ravel(), V.ravel()]


def box_surface(L, W, H):
    """Caixa apoiada no plano z=0, sem a face de baixo (nao e vista no scan)."""
    faces = []
    t = grid_rect(L, W, S); faces.append(np.c_[t[:, 0], t[:, 1], np.full(len(t), H)])
    f = grid_rect(L, H, S); faces.append(np.c_[f[:, 0], np.zeros(len(f)), f[:, 1]])
    faces.append(np.c_[f[:, 0], np.full(len(f), W), f[:, 1]])
    s = grid_rect(W, H, S); faces.append(np.c_[np.zeros(len(s)), s[:, 0], s[:, 1]])
    faces.append(np.c_[np.full(len(s), L), s[:, 0], s[:, 1]])
    return np.vstack(faces)


def cylinder(cx, cy, z0, r, h):
    th = np.arange(0, 2 * np.pi, S / r)
    z = np.arange(0, h, S)
    T, Z = np.meshgrid(th, z)
    side = np.c_[cx + r * np.cos(T.ravel()), cy + r * np.sin(T.ravel()), z0 + Z.ravel()]
    g = grid_rect(2 * r, 2 * r, S) - r
    g = g[np.linalg.norm(g, axis=1) <= r]
    top = np.c_[cx + g[:, 0], cy + g[:, 1], np.full(len(g), z0 + h)]
    return np.vstack([side, top])


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _true_T(frag, R, t):
    """Transformacao que leva o trecho deslocado de volta ao lugar certo."""
    c = frag.mean(0)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = c - (c + t) @ R
    return T


def main(out="testes/sintetico.ply", cenario="detalhe"):
    L, W, H = 60, 40, 25
    peca = np.vstack([box_surface(L, W, H), cylinder(20, 20, H, 8, 15)])
    # Ressalto/rebaixo para dar geometria assimetrica (registro precisa disso)
    peca = np.vstack([peca, cylinder(48, 30, H, 4, 6)])
    peca -= [L / 2, W / 2, 0]  # centraliza em XY
    verdade = peca.copy()

    # Base: disco r=70 no plano z=0, fora da pegada da peca
    g = grid_rect(140, 140, S) - 70
    g = g[np.linalg.norm(g, axis=1) <= 70]
    g = g[~((np.abs(g[:, 0]) < L / 2) & (np.abs(g[:, 1]) < W / 2))]
    base = np.c_[g, np.zeros(len(g))]

    # Perda de rastreamento: parte da peca foi capturada deslocada.
    # cenario "detalhe": sobreposicao 10 a 22 mm contem o ressalto cilindrico
    # cenario "liso": sobreposicao 0 a 10 mm so tem paredes retas (ambiguo)
    corte_a, corte_b = (22, 10) if cenario == "detalhe" else (10, 0)
    main_part = peca[peca[:, 0] < corte_a]
    frag = peca[peca[:, 0] > corte_b]
    np.save(out.replace(".ply", "_T_verdade.npy"), _true_T(frag, rot_z(6.0), np.array([25.0, 45.0, 18.0])))
    R = rot_z(6.0)
    t = np.array([25.0, 45.0, 18.0])
    c = frag.mean(0)
    frag_desl = (frag - c) @ R.T + c + t

    pts = np.vstack([main_part, frag_desl, base])
    pts += rng.normal(0, 0.03, pts.shape)  # ruido do sensor

    # Outliers espalhados e ilhas flutuantes
    lo, hi = pts.min(0) - 10, pts.max(0) + 10
    outl = rng.uniform(lo, hi, (2500, 3))
    ilhas = np.vstack([rng.normal(rng.uniform(lo, hi), 0.6, (90, 3)) for _ in range(6)])
    pts = np.vstack([pts, outl, ilhas])
    pts = pts[rng.permutation(len(pts))]

    o3d.io.write_point_cloud(out, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))
    o3d.io.write_point_cloud(out.replace(".ply", "_verdade.ply"),
                             o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verdade)))
    print(f"{len(pts):,} pontos | peca {len(verdade):,} | base {len(base):,} | trecho deslocado {len(frag):,}")


if __name__ == "__main__":  # uso: gerar_sintetico.py saida.ply [detalhe|liso]
    main(*sys.argv[1:])
