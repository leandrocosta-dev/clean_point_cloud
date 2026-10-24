"""Funde A (inteira) + partes de B que A nao cobre (dist > --corte mm), Poisson depth 10, corte a 0,3 mm. Uso: fundir_malhas.py <A.ply> <B_alinhada.ply> <saida.ply> [--corte 1.5]"""
import open3d as o3d, numpy as np, sys, time
a,b,out=sys.argv[1:4]; corte=float(sys.argv[sys.argv.index("--corte")+1]) if "--corte" in sys.argv else 1.5
MA=o3d.io.read_triangle_mesh(a); MB=o3d.io.read_triangle_mesh(b)
for M in (MA,MB): M.compute_vertex_normals(); M.compute_triangle_normals()
A=MA.sample_points_poisson_disk(1500000,use_triangle_normal=True) if False else MA.sample_points_uniformly(1500000,use_triangle_normal=True)
B=MB.sample_points_uniformly(1000000,use_triangle_normal=True)
n=np.array([0.935,-0.064,0.349]); d0=-124.841/np.linalg.norm(n); n/=np.linalg.norm(n)  # plano da mesa (RANSAC no scan A), gato em s<0
sB=np.asarray(B.points)@n+d0; B=B.select_by_index(np.where(sB<-0.6)[0]); print("B sem mesa",len(B.points))
dB=np.asarray(B.compute_point_cloud_distance(A)); Bn=B.select_by_index(np.where(dB>corte)[0]); print("B novo",len(Bn.points),"de",len(B.points))
F=A+Bn
t=time.time(); m,den=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(F,depth=10,width=0,scale=1.05,linear_fit=True); den=np.asarray(den); print("poisson",len(m.vertices),round(time.time()-t))
tr=o3d.geometry.KDTreeFlann(F); V=np.asarray(m.vertices)
dist=np.array([np.sqrt(tr.search_knn_vector_3d(v,1)[2][0]) for v in V])
keep=(dist<0.3)&(den>np.quantile(den,0.005)); m.remove_vertices_by_mask(~keep); m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(out,m); print(m)
