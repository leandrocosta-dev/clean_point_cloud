"""Metricas e reparo de malha com MeshLib (o motor do MeshInspector): os numeros batem com o painel Information dele.

    python pipeline/malha_meshlib.py <malha.ply> [--saida reparada.ply] [--min-faces 2000] [--furo-max 20] [--orientar]

Sem --saida so mede. Com --saida: remove componentes com menos de --min-faces, corrige arestas multiplas (nao-manifold),
preenche furos de ate --furo-max mm (diametro aproximado da borda), opcionalmente reorienta faces, e mede de novo.
Calibrado: parachoque_malha_1M_v8.ply -> 980610 tri, 491002 vertices, 538 furos, 384 componentes (igual ao MeshInspector 2.5.9).
"""
import argparse
import sys

from meshlib import mrmeshpy as mm


def metricas(m):
    comps = mm.MeshComponents.getAllComponents(m)
    tam = sorted((c.count() for c in comps), reverse=True)
    return dict(triangulos=m.topology.numValidFaces(), vertices=m.topology.numValidVerts(),
                furos=len(m.topology.findHoleRepresentiveEdges()), componentes=len(comps),
                maior_componente=tam[0] if tam else 0)


def reparar(m, min_faces=2000, furo_max=20.0, orientar=False):
    comps = mm.MeshComponents.getAllComponents(m)
    manter = mm.FaceBitSet()
    for c in comps:
        if c.count() >= min_faces:
            manter |= c
    m.topology.deleteFaces(m.topology.getValidFaces() - manter)
    mm.fixMultipleEdges(m)
    p = mm.FillHoleParams()
    for e in m.topology.findHoleRepresentiveEdges():
        # comprimento da borda ~ pi * diametro; so preenche furos pequenos
        if m.holePerimeter(e) <= 3.1416 * furo_max:
            mm.fillHole(m, e, p)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("malha"); ap.add_argument("--saida")
    ap.add_argument("--min-faces", type=int, default=2000); ap.add_argument("--furo-max", type=float, default=20.0)
    ap.add_argument("--orientar", action="store_true")
    a = ap.parse_args()
    m = mm.loadMesh(a.malha)
    print("antes :", metricas(m))
    if a.saida:
        reparar(m, a.min_faces, a.furo_max, a.orientar)
        print("depois:", metricas(m))
        mm.saveMesh(m, a.saida)
        print("gravado", a.saida)


if __name__ == "__main__":
    main()
