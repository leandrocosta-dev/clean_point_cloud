import sys,sqlite3,struct,json,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
S=sys.argv[1]
c=sqlite3.connect(S+'/frames.sqlite')
fx,fy,cx,cy=1140.4,1141.0,232.8,319.5
def frame(i):
    d=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,n=struct.unpack('<III',d[:12]); mask=np.frombuffer(d[12:12+w*h//8],np.uint8)
    vals=np.frombuffer(d[12+w*h//8+4:12+w*h//8+4+4*n],'<f4').astype(float)
    v,u=np.nonzero(np.unpackbits(mask,bitorder='little').astype(bool).reshape(h,w))
    return np.c_[(u-cx)*vals/fx,-(v-cy)*vals/fy,-vals]
r=json.load(open('C:/Users/leandro.soliddesigne/Music/IA/34.nuvem_pontos/moviai_scan/saida_atual/raw_preview_relatorio.json'))
n=np.array(r['base']['n']); d0=r['base']['d']; u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
Bm=np.c_[u,v,n].T  # mundo -> local
cen=np.array([166.45,-28.30]); R=22.91
rows=[(i,np.frombuffer(b[16:144],'<f8').reshape(4,4)) for i,b in c.execute('select id,bin from FrameInfo order by id')]
TL=np.load(S+'/timeline.npy')
def world(k,T=None):
    i,T0=rows[k]; T=T0 if T is None else T; P=frame(i)[::2]; return P@T[:3,:3].T+T[:3,3]
def loc(W): return np.c_[W@u,W@v,W@n+d0]
def plug(W):
    q=loc(W); rr=np.linalg.norm(q[:,:2]-cen,axis=1); return q[(rr>R+1.5)&(q[:,2]>46)]
A=np.vstack([plug(world(k)) for k in range(0,70)])
B=np.vstack([plug(world(k)) for k in range(148,248)])
A=np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A)).voxel_down_sample(0.3).points)
B=np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(B)).voxel_down_sample(0.3).points)
tA=cKDTree(A)
def rotz(p,deg):
    a=np.radians(deg); x=p[:,0]-cen[0]; y=p[:,1]-cen[1]
    return np.c_[cen[0]+np.cos(a)*x-np.sin(a)*y, cen[1]+np.sin(a)*x+np.cos(a)*y, p[:,2]]
sc=lambda t: np.mean(tA.query(rotz(B,t))[0]<0.5)
grid=[(t,sc(t)) for t in np.arange(-30,30.01,0.5)]
t0=max(grid,key=lambda x:x[1])[0]
fine=[(t,sc(t)) for t in np.arange(t0-1,t0+1.001,0.05)]
delta,best=max(fine,key=lambda x:x[1])
print(f'correcao B->A em torno do eixo: {delta:+.2f} gr | sobreposicao {best*100:.1f}% (sem correcao {sc(0)*100:.1f}%)')
d,_=tA.query(rotz(B,delta)); print('distancia mediana das paredes B corrigidas ate A: %.3f mm'%np.median(d[d<2]))
# Correcao: rotacao em torno do eixo (no mundo), linear no trecho cego #70..#147, constante de #148 a #247
axis_w=n; p0_loc=np.array([cen[0],cen[1],0.0])
p0_w=Bm.T@(p0_loc-np.array([0,0,d0]))
def Rw(deg):
    a=np.radians(deg); k=axis_w; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Rm=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K; M=np.eye(4); M[:3,:3]=Rm; M[:3,3]=p0_w-Rm@p0_w; return M
# confere o sentido: aplicar Rw(delta) nos pontos do mundo deve equivaler a rotz(+delta) no local
tst=world(200); e=np.median(np.linalg.norm(loc(tst@Rw(delta)[:3,:3].T+Rw(delta)[:3,3])-rotz(loc(tst),delta),axis=1))
print('consistencia do sentido da rotacao: %.4f mm'%e)
fixes={}
for k in range(len(rows)):
    if rows[k][0]>>32!=0: continue
    if 70<=k<148: frac=(k-69)/(148-69)
    elif k>=148: frac=1.0
    else: frac=0.0
    fixes[k]=frac*delta
np.save(S+'/fixes.npy',np.array([fixes.get(k,0.0) for k in range(len(rows))]))
def fuse(corr):
    acc=[]
    for k,(i,T) in enumerate(rows):
        P=frame(i); W=P@T[:3,:3].T+T[:3,3]
        if corr and fixes.get(k,0): M=Rw(fixes[k]); W=W@M[:3,:3].T+M[:3,3]
        acc.append(W)
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(acc))).voxel_down_sample(0.15)
    return pc
for corr,name in ((False,'fusao_original'),(True,'fusao_corrigida')):
    pc=fuse(corr); o3d.io.write_point_cloud(f'{S}/{name}.ply',pc); print(name,len(pc.points),'pontos')
json.dump({'delta_graus':delta,'trecho_cego':[70,147],'trecho_B':[148,247]},open(S+'/correcao.json','w'))
