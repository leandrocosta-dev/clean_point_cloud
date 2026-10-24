"""Alinha o scan B (outro lado) na nuvem limpa do scan A (Creality Otter). Uso: alinhar_lados.py <A_limpa.ply> <B_pc_before.ply> <saida_dir>"""
import open3d as o3d, numpy as np, sys, os, copy
A_path,B_path,out=sys.argv[1:4]; os.makedirs(out,exist_ok=True)
A=o3d.io.read_point_cloud(A_path)
B=o3d.io.read_point_cloud(B_path); B.scale(1000,center=(0,0,0))
sp=np.median(B.compute_nearest_neighbor_distance())
l=np.array(B.cluster_dbscan(eps=max(1.0,sp*4),min_points=10)); B=B.select_by_index(np.where(l==np.argmax(np.bincount(l[l>=0])))[0])
B,_=B.remove_statistical_outlier(30,3.0); print("B principal",len(B.points),"A",len(A.points))
o3d.io.write_point_cloud(os.path.join(out,"B_principal.ply"),B)
vox=1.5
def prep(p):
    q=p.voxel_down_sample(vox); q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox*2.5,max_nn=40))
    f=o3d.pipelines.registration.compute_fpfh_feature(q,o3d.geometry.KDTreeSearchParamHybrid(radius=vox*6,max_nn=120)); return q,f
qa,fa=prep(A); qb,fb=prep(B)
best=None
for seed in range(8):
    o3d.utility.random.seed(seed)
    r=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(qb,qa,fb,fa,True,vox*2.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vox*2.0)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000,0.999))
    ic=o3d.pipelines.registration.registration_icp(qb,qa,vox*1.5,r.transformation,o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(seed,"ransac",round(r.fitness,3),"icp fit",round(ic.fitness,3),"rmse",round(ic.inlier_rmse,3))
    if best is None or ic.fitness>best[0].fitness: best=(ic,seed)
np.save(os.path.join(out,"T_ransac_best.npy"),best[0].transformation); print("melhor",best[1],best[0].fitness)
