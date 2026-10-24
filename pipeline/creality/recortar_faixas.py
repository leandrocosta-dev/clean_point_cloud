"""2a etapa do recorte (depois de recortar_base.py; malha ja nivelada, base em z~1,3-2,2 mm).
Uso: recortar_faixas.py <entrada.ply> <saida.ply>

Faixas finas da borda da parede desciam 1-2 mm abaixo da base e saiam como "faixa" no pe do gato:
1. triangulos com todos os vertices em z<2,2 que nao sao do maior componente (a base oval) saem;
2. triangulos com algum vertice em z<1,3 saem (abaixo da base, que comeca em z=1,4);
3. maior componente, malha apoiada em z=0, furos fechados com fillHoleNicely (MeshLib).
"""
import open3d as o3d, numpy as np, sys
from meshlib import mrmeshpy as mm
ent,sai=sys.argv[1:3]
M=o3d.io.read_triangle_mesh(ent); V=np.asarray(M.vertices); T=np.asarray(M.triangles)
low=(V[:,2]<2.2)[T].all(1); idx=np.where(low)[0]
S=o3d.geometry.TriangleMesh(M.vertices,o3d.utility.Vector3iVector(T[low])); tc,nt,_=S.cluster_connected_triangles(); tc=np.array(tc); nt=np.array(nt)
ap=np.zeros(len(T),bool); ap[idx[tc!=np.argmax(nt)]]=True; M.remove_triangles_by_mask(ap); M.remove_unreferenced_vertices()
V=np.asarray(M.vertices); T=np.asarray(M.triangles); M.remove_triangles_by_mask((V[:,2]<1.3)[T].any(1)); M.remove_unreferenced_vertices()
tc,nt,_=M.cluster_connected_triangles(); tc=np.array(tc); nt=np.array(nt); M.remove_triangles_by_mask(tc!=np.argmax(nt)); M.remove_unreferenced_vertices()
V=np.asarray(M.vertices).copy(); V[:,2]-=V[:,2].min(); M.vertices=o3d.utility.Vector3dVector(V); o3d.io.write_triangle_mesh(sai,M)
m=mm.loadMesh(sai); mm.fixMultipleEdges(m); p=mm.FillHoleNicelySettings(); p.triangulateOnly=False
for e in m.topology.findHoleRepresentiveEdges(): mm.fillHoleNicely(m,e,p)
print("furos",len(m.topology.findHoleRepresentiveEdges()),"comp",len(mm.MeshComponents.getAllComponents(m)),"faces",m.topology.numValidFaces(),"vol",round(m.volume()))
mm.saveMesh(m,sai)
