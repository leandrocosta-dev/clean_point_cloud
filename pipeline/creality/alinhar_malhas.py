"""Alinha a malha B na malha A (ambas exportadas do Creality Scan, mm). Uso: alinhar_malhas.py <A.ply> <B.ply> <saida_dir>"""
import open3d as o3d, numpy as np, sys, os
reg=o3d.pipelines.registration
A_path,B_path,out=sys.argv[1:4]; os.makedirs(out,exist_ok=True)
MA=o3d.io.read_triangle_mesh(A_path); MB=o3d.io.read_triangle_mesh(B_path)
for M in (MA,MB): M.compute_vertex_normals()
A=MA.sample_points_uniformly(800000); B=MB.sample_points_uniformly(600000)
vox=1.2
def prep(p):
    q=p.voxel_down_sample(vox); q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox*3,max_nn=40))
    return q,reg.compute_fpfh_feature(q,o3d.geometry.KDTreeSearchParamHybrid(radius=vox*7,max_nn=150))
qa,fa=prep(A); qb,fb=prep(B); res=[]
for seed in range(8):
    o3d.utility.random.seed(seed)
    r=reg.registration_ransac_based_on_feature_matching(qb,qa,fb,fa,True,vox*2,reg.TransformationEstimationPointToPoint(False),3,
      [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),reg.CorrespondenceCheckerBasedOnDistance(vox*2)],reg.RANSACConvergenceCriteria(3000000,0.999))
    T=r.transformation
    for thr in (2.0,1.0,0.6,0.4,0.25):
        ic=reg.registration_icp(B,A,thr,T,reg.TransformationEstimationPointToPlane(),reg.ICPConvergenceCriteria(1e-9,1e-9,60)); T=ic.transformation
    print(seed,"ransac",round(r.fitness,3),"icp",round(ic.fitness,3),"rmse",round(ic.inlier_rmse,3),flush=True); res.append((ic.fitness,T))
res.sort(key=lambda x:-x[0]); T=res[0][1]; np.save(os.path.join(out,"T_B_para_A.npy"),T)
B.transform(T); d=np.asarray(B.compute_point_cloud_distance(A)); print("B->A pcts 25/50/75/90",np.percentile(d,[25,50,75,90]).round(3),"frac<0.3",np.mean(d<0.3).round(3))
MB.transform(T); o3d.io.write_triangle_mesh(os.path.join(out,"B_alinhada.ply"),MB)
