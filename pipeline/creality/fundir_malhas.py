"""Funde A + B (ambas do Creality Scan, B ja alinhada em A). Uso: fundir_malhas.py <A.ply> <B_alinhada.ply> <saida.ply> [--corte 1.5]

- Plano da mesa (RANSAC no scan A), gato em s<0. A malha do software em A traz um pedaco da mesa colado no plano: apaga os
  pontos de A com s > -8 mm cuja projecao no plano cai fora da pegada do gato (casco convexo dos pontos de B com s > -8,
  alargado 2 mm). A base plana do gato (oval com furo) vem de B, nao e mesa.
- De B entra so o que A (ja sem mesa) nao cobre (dist > corte). Poisson depth 10, corte a 0,3 mm.
"""
import open3d as o3d, numpy as np, sys, time
from scipy.spatial import ConvexHull, Delaunay
a,b,out=sys.argv[1:4]; corte=float(sys.argv[sys.argv.index("--corte")+1]) if "--corte" in sys.argv else 1.5
MA=o3d.io.read_triangle_mesh(a); MB=o3d.io.read_triangle_mesh(b)
for M in (MA,MB): M.compute_vertex_normals(); M.compute_triangle_normals()
A=MA.sample_points_uniformly(1500000,use_triangle_normal=True); B=MB.sample_points_uniformly(1000000,use_triangle_normal=True)
n=np.array([0.935,-0.064,0.349]); d0=-124.841/np.linalg.norm(n); n/=np.linalg.norm(n)
u=np.cross(n,[0,0,1.]); u/=np.linalg.norm(u); v=np.cross(n,u)
PA=np.asarray(A.points); PB=np.asarray(B.points); sA=PA@n+d0; sB=PB@n+d0
q=PB[sB>-8]; Q=np.c_[q@u,q@v]; h=ConvexHull(Q); H=Q[h.vertices]; c=H.mean(0); H=H+2.0*(H-c)/np.linalg.norm(H-c,axis=1,keepdims=True)
dentro=Delaunay(H).find_simplex(np.c_[PA@u,PA@v])>=0
mesa=(sA>-8)&~dentro; print("A: pontos de mesa removidos",mesa.sum(),"de",len(PA))
A=A.select_by_index(np.where(~mesa)[0])
dB=np.asarray(B.compute_point_cloud_distance(A)); Bn=B.select_by_index(np.where(dB>corte)[0]); print("B novo",len(Bn.points),"de",len(B.points))
F=A+Bn
t=time.time(); m,den=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(F,depth=10,width=0,scale=1.05,linear_fit=True); den=np.asarray(den); print("poisson",len(m.vertices),round(time.time()-t))
tr=o3d.geometry.KDTreeFlann(F); V=np.asarray(m.vertices)
dist=np.array([np.sqrt(tr.search_knn_vector_3d(x,1)[2][0]) for x in V])
keep=(dist<0.3)&(den>np.quantile(den,0.005)); m.remove_vertices_by_mask(~keep); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(out,m); print(m)
