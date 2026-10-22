"""Orienta para FORA as normais da nuvem nos pontos visiveis de fora da peca (corrige trechos "de verso" na malha).

    python pipeline/orientar_normais.py <nuvem.ply (com normais)> --saida <nuvem_orientada.ply> [--celula 1.0] [--tol 3.0] [--cone 65]

As normais da fusao apontam para o lado de onde a camera viu a superficie: onde so um scan visto POR DENTRO mediu a parede,
o Poisson gera uma folha de verso (amarelo no MeshInspector, branco no 3Dconnexion). Aqui, de varias direcoes de fora (cone ao
redor do eixo menor da peca, PCA) um z-buffer ortografico marca os pontos a ate --tol mm do mais alto da celula: esses
sao visiveis de fora e tem a normal virada para o observador. O resto fica como estava.
"""
import argparse
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--celula", type=float, default=1.0); ap.add_argument("--tol", type=float, default=0.8); ap.add_argument("--cone", type=float, default=65.0)
a = ap.parse_args()
pc = o3d.io.read_point_cloud(a.nuvem); P = np.asarray(pc.points); N = np.asarray(pc.normals).copy()
c = P[::50].mean(0); w, E = np.linalg.eigh(np.cov((P[::50] - c).T)); V = E[:, ::-1]
Q = ((P - c) @ V).astype(np.float32); Nq = (N @ V).astype(np.float32)
dirs = [np.array([0, 0, 1.0])]
for ang in np.radians(np.arange(25, a.cone + 1, 20)):
    for az in np.radians(np.arange(0, 360, 45)):
        dirs.append(np.array([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az), np.cos(ang)]))
visto = np.zeros(len(P), bool); flip = np.zeros(len(P), bool)
for k, dr in enumerate(dirs):
    dr = dr.astype(np.float32); u = np.cross(dr, [0, 0, 1.0] if abs(dr[2]) < 0.99 else [1.0, 0, 0]); u /= np.linalg.norm(u); v = np.cross(dr, u)
    s = Q @ dr; gx = np.floor((Q @ u) / a.celula).astype(np.int64); gy = np.floor((Q @ v) / a.celula).astype(np.int64)
    key = (gx - gx.min()) * 100003 + (gy - gy.min()); _, inv = np.unique(key, return_inverse=True)
    mx = np.full(inv.max() + 1, -1e9, np.float32); np.maximum.at(mx, inv, s)
    vis = s >= mx[inv] - a.tol
    ruim = vis & (Nq @ dr < 0)                      # visivel de fora, normal apontando para dentro
    flip |= ruim & ~visto if False else ruim
    visto |= vis
    print(f"  direcao {k+1}/{len(dirs)}: visiveis {visto.mean()*100:.1f}% | normais p/ dentro acumuladas {flip.mean()*100:.2f}%", flush=True)
# um ponto visivel de uma direcao com normal para dentro pode estar com normal correta para outra direcao: so vira se NENHUMA direcao que o ve a aprova
bom = np.zeros(len(P), bool)
for k, dr in enumerate(dirs):
    dr = dr.astype(np.float32); u = np.cross(dr, [0, 0, 1.0] if abs(dr[2]) < 0.99 else [1.0, 0, 0]); u /= np.linalg.norm(u); v = np.cross(dr, u)
    s = Q @ dr; gx = np.floor((Q @ u) / a.celula).astype(np.int64); gy = np.floor((Q @ v) / a.celula).astype(np.int64)
    key = (gx - gx.min()) * 100003 + (gy - gy.min()); _, inv = np.unique(key, return_inverse=True)
    mx = np.full(inv.max() + 1, -1e9, np.float32); np.maximum.at(mx, inv, s)
    bom |= (s >= mx[inv] - a.tol) & (Nq @ dr > 0)
flip = visto & ~bom
Nq[flip] *= -1
print(f"visiveis de fora: {visto.mean()*100:.1f}% | normais viradas p/ fora: {flip.sum():,} ({flip.mean()*100:.2f}%)")
pc.normals = o3d.utility.Vector3dVector((Nq @ V.T).astype(np.float64))
o3d.io.write_point_cloud(a.saida, pc, write_ascii=False)
