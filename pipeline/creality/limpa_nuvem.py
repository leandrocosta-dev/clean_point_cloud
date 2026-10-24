import open3d as o3d, numpy as np, sys
R=r"C:\Users\pc\Downloads\Teste Gatinho\sGtXXt9_DOkFjr_xK2vpu_o0vNul"
def carregar(nome):
    p=o3d.io.read_point_cloud(R+"\\"+nome); p.scale(1000,center=(0,0,0)); return p
def objeto(p, ref_lado=None, dist=1.0):
    q=p.voxel_down_sample(0.5); pl,_=q.segment_plane(0.4,3,3000)
    n=pl[:3]/np.linalg.norm(pl[:3]); d=pl[3]/np.linalg.norm(pl[:3])
    P=np.asarray(p.points); s=P@n+d
    if ref_lado is None: ref_lado=-1 if np.median(s)>0 else 1  # decide below
    return n,d,s
def ruido(p,idx=None,k=40,raio=1.0,N=20000,seed=0):
    t=o3d.geometry.KDTreeFlann(p); P=np.asarray(p.points); rng=np.random.default_rng(seed); out=[]
    for i in rng.choice(len(P),N,replace=False):
        _,ix,_=t.search_radius_vector_3d(P[i],raio)
        if len(ix)<15: continue
        Q=P[ix]-P[ix].mean(0); w=np.linalg.eigvalsh(Q.T@Q/len(Q)); out.append(np.sqrt(max(w[0],0)))
    return np.median(out),np.mean(out)
if __name__=="__main__":
    for nome in ("pc_before.ply","pc_after.ply"):
        p=carregar(nome); n,d,s=objeto(p)
        # lado do gato = o lado que contem a maior massa de pontos fora do plano; usar Fused como referencia
        F=o3d.io.read_triangle_mesh(R+r"\result\Fused.ply"); sf=np.asarray(F.vertices)@n+d; lado=np.sign(np.median(sf))
        m=(s*lado)>1.0; o=p.select_by_index(np.where(m)[0])
        l=np.array(o.cluster_dbscan(eps=0.8,min_points=8)); c=np.bincount(l[l>=0]); O=o.select_by_index(np.where(l==np.argmax(c))[0])
        print(nome,len(O.points),"ruido mediana/media (mm)",ruido(O))
