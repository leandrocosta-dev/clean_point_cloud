"""Fecha furos da malha onde a nuvem de scans JA TEM superficie (o aparo/Poisson apagou), deixando as aberturas reais.

    python pipeline/fechar_furos_com_dado.py malha.ply nuvem_externa.ply --saida v15.ply [--tol 5.5] [--perim-min 25] [--perim-max 4000]

Para cada furo: simula fillHole numa copia e mede a distancia (p90) dos centros dos triangulos do remendo a nuvem. O furo so e fechado se o remendo tem p90 <= --tol mm
da nuvem E >= --frac-min das amostras tem >=3 pontos da nuvem a 1 mm (superficie MEDIDA dentro do furo). Abertura real (remendo no ar, ou
so encostado nas bordas de uma fenda estreita) e mantida. Teste so por distancia (p90 < 5,5 mm) fechou aberturas reais: errado.
Nao inverter faces do remendo uma a uma: quebra a topologia (milhares de furos).
"""
import argparse
import numpy as np
import open3d as o3d
import meshlib.mrmeshpy as mr

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("malha"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--tol", type=float, default=2.5); ap.add_argument("--frac-min", type=float, default=0.4); ap.add_argument("--perim-min", type=float, default=25)
ap.add_argument("--perim-max", type=float, default=4000)
a = ap.parse_args()
m = mr.loadMesh(a.malha)
t = o3d.geometry.KDTreeFlann(o3d.io.read_point_cloud(a.nuvem))


def p90_remendo(e):
    c = mr.Mesh(m); f0 = c.topology.faceSize(); mr.fillHole(c, e, mr.FillHoleParams())
    Q = np.array([[c.triCenter(mr.FaceId(f)).x, c.triCenter(mr.FaceId(f)).y, c.triCenter(mr.FaceId(f)).z]
                  for f in range(f0, c.topology.faceSize())])
    if not len(Q):
        return 1e9, 0.0
    S = Q[::max(1, len(Q) // 200)]
    p90 = float(np.percentile([np.sqrt(t.search_knn_vector_3d(q, 1)[2][0]) for q in S], 90))
    frac = float(np.mean([t.search_radius_vector_3d(q, 1.0)[0] >= 3 for q in S]))
    return p90, frac


feitos, recusados = 0, set()
while True:
    achou = False
    for e in m.topology.findHoleRepresentiveEdges():
        p = m.orgPnt(e); k = (round(p.x, 1), round(p.y, 1), round(p.z, 1))
        if k in recusados or not a.perim_min < m.holePerimeter(e) < a.perim_max:
            continue
        p90, frac = p90_remendo(e)
        if p90 <= a.tol and frac >= a.frac_min:
            s = mr.FillHoleNicelySettings(); s.triangulateOnly = False
            mr.fillHoleNicely(m, e, s); feitos += 1; achou = True; break
        recusados.add(k)
    if not achou:
        break
print("fechados", feitos, "restantes", len(m.topology.findHoleRepresentiveEdges()))
mr.saveMesh(m, a.saida)
