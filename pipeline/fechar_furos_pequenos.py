"""Fecha so furos PEQUENOS de borda lisa (furinhos no meio da superficie), preservando aberturas reais da peca.

    python pipeline/fechar_furos_pequenos.py malha.ply nuvem_externa.ply --saida out.ply [--perim-max 300] [--ang-max 20] [--plano-max 0.05] [--tol 3.0]

Criterios (todos): perimetro <= --perim-max mm; p90 do desvio das normais das faces da borda <= --ang-max graus (borda lisa:
uma abertura real tem parede/quina e normais variadas); anel quase plano (menor/maior autovalor <= --plano-max); remendo a <= --tol mm da nuvem; com --frac-min, >= essa fracao de amostras do remendo com >=3 pontos da nuvem a 1 mm (dado MEDIDO dentro do furo: abertura real de pequeno porte tem ~0).
Para furinhos em dobras (normais variadas) use --ang-max 90 --plano-max 1 --perim-max 200 --frac-min 0.3.
Aberturas da grade (perimetros 400-700 mm) e bordas da casca ficam de fora. Conferir SEMPRE o resultado contra a versao anterior lado a lado.
"""
import argparse
import numpy as np
import open3d as o3d
import meshlib.mrmeshpy as mr

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("malha"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--perim-max", type=float, default=300); ap.add_argument("--perim-min", type=float, default=20)
ap.add_argument("--ang-max", type=float, default=20); ap.add_argument("--plano-max", type=float, default=0.05)
ap.add_argument("--tol", type=float, default=3.0); ap.add_argument("--frac-min", type=float, default=0.0)
a = ap.parse_args()
m = mr.loadMesh(a.malha)
t = o3d.geometry.KDTreeFlann(o3d.io.read_point_cloud(a.nuvem))


def avaliar(e):
    ring = m.topology.getLeftRing(e)
    B = np.array([[m.orgPnt(x).x, m.orgPnt(x).y, m.orgPnt(x).z] for x in ring])
    N = []
    for x in ring:
        f = m.topology.right(x)
        if f.valid():
            n = m.dirDblArea(f); l = np.linalg.norm([n.x, n.y, n.z]); N.append([n.x / l, n.y / l, n.z / l])
    N = np.array(N); mn = N.mean(0); mn /= np.linalg.norm(mn) + 1e-12
    ang = float(np.percentile(np.degrees(np.arccos(np.clip(N @ mn, -1, 1))), 90))
    s = np.linalg.svd(B - B.mean(0), compute_uv=False)
    return ang, float(s[2] / s[0]), B.mean(0)


def p90_remendo(e):
    c = mr.Mesh(m); f0 = c.topology.faceSize(); mr.fillHole(c, e, mr.FillHoleParams())
    Q = np.array([[c.triCenter(mr.FaceId(f)).x, c.triCenter(mr.FaceId(f)).y, c.triCenter(mr.FaceId(f)).z]
                  for f in range(f0, c.topology.faceSize())])
    if not len(Q):
        return 1e9, 0.0
    S = Q[::max(1, len(Q) // 100)]
    frac = float(np.mean([t.search_radius_vector_3d(q, 1.0)[0] >= 3 for q in S]))
    return float(np.percentile([np.sqrt(t.search_knn_vector_3d(q, 1)[2][0]) for q in S], 90)), frac


feitos, recusados = [], set()
while True:
    achou = False
    for e in m.topology.findHoleRepresentiveEdges():
        p = m.orgPnt(e); k = (round(p.x, 1), round(p.y, 1), round(p.z, 1)); per = m.holePerimeter(e)
        if k in recusados or not a.perim_min <= per <= a.perim_max:
            continue
        ang, plano, c = avaliar(e)
        p90, frac = p90_remendo(e) if (ang <= a.ang_max and plano <= a.plano_max) else (1e9, 0.0)
        if p90 <= a.tol and frac >= a.frac_min:
            s = mr.FillHoleNicelySettings(); s.triangulateOnly = False
            mr.fillHoleNicely(m, e, s); feitos.append((round(per), c.round().tolist())); achou = True; break
        recusados.add(k)
    if not achou:
        break
print("fechados", len(feitos), feitos, "restantes", len(m.topology.findHoleRepresentiveEdges()))
mr.saveMesh(m, a.saida)
