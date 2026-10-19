import sys,sqlite3,struct,json,time,numpy as np,open3d as o3d
S,mode=sys.argv[1],sys.argv[2]
t0=time.time()
c=sqlite3.connect(S+'/frames.sqlite')
fx,fy,cx,cy=1140.4,1141.0,232.8,319.5
r=json.load(open('/home/claude/moviai_scan/saida_real/3cd01b6f-raw_preview_relatorio.json'))
n=np.array(r['base']['n']); d0=r['base']['d']
u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u); Bm=np.c_[u,v,n].T
cen=np.array([166.45,-28.30]); p0=Bm.T@(np.array([cen[0],cen[1],0.0])-np.array([0,0,d0]))
def Rw(deg):
    a=np.radians(deg); k=n; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Rm=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K; M=np.eye(4); M[:3,:3]=Rm; M[:3,3]=p0-Rm@p0; return M
fixes=np.load(S+'/fixes.npy') if mode=='corrigida' else None
rows=c.execute('select id,bin from FrameInfo order by id').fetchall()
acc=o3d.geometry.PointCloud()
for k,(i,b) in enumerate(rows):
    T=np.frombuffer(b[16:144],'<f8').reshape(4,4)
    if fixes is not None and fixes[k]: T=Rw(fixes[k])@T
    dd=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,nn=struct.unpack('<III',dd[:12]); mask=np.frombuffer(dd[12:12+w*h//8],np.uint8)
    z=np.frombuffer(dd[12+w*h//8+4:12+w*h//8+4+4*nn],'<f4').astype(float)
    vv,uu=np.nonzero(np.unpackbits(mask,bitorder='little').astype(bool).reshape(h,w))
    W=np.c_[(uu-cx)*z/fx,-(vv-cy)*z/fy,-z]@T[:3,:3].T+T[:3,3]
    keep=(W@n+d0)>35.0   # remove base e haste (peca fica entre 44 e 82 mm de altura)
    if keep.sum()<200: continue
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W[keep])).voxel_down_sample(0.2)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8,max_nn=20))
    pc.orient_normals_towards_camera_location(T[:3,3])
    acc+=pc
    if k%60==0: acc=acc.voxel_down_sample(0.2)
acc=acc.voxel_down_sample(0.2)
acc,_=acc.remove_statistical_outlier(20,2.0)
lab=np.array(acc.cluster_dbscan(1.2,10)); big=np.argmax(np.bincount(lab[lab>=0]))
acc=acc.select_by_index(np.where(lab==big)[0].tolist())
print(mode,'nuvem 0,2 mm:',len(acc.points),'pontos',f'{time.time()-t0:.0f}s',flush=True)
mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(acc,depth=9,scale=1.1)
dens=np.asarray(dens); mesh.remove_vertices_by_mask(dens<np.quantile(dens,0.06))
bb=acc.get_axis_aligned_bounding_box(); bb=bb.scale(1.02,bb.get_center()); mesh=mesh.crop(bb)
mesh.remove_degenerate_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_unreferenced_vertices()
tc,cn,_=mesh.cluster_connected_triangles(); tc=np.asarray(tc); cn=np.asarray(cn)
mesh.remove_triangles_by_mask(cn[tc]<0.02*cn.max()); mesh.remove_unreferenced_vertices()
L=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]; e=np.linalg.norm(L[:,0]-L[:,1],axis=1)
mesh.compute_triangle_normals(); mesh.compute_vertex_normals()
out=f'/home/claude/difusor_{mode}.stl'; o3d.io.write_triangle_mesh(out,mesh)
print(out,len(mesh.triangles),'triangulos | aresta mediana %.3f mm'%np.median(e),f'{time.time()-t0:.0f}s',flush=True)
