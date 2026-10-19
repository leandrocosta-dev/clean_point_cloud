import sys,sqlite3,struct,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import minimize
S=sys.argv[1]
c=sqlite3.connect(S+'/frames.sqlite')
prev=np.asarray(o3d.io.read_point_cloud(S+'/revo_proj/Project20261015_204752_20261015_204805_Limpo/data/20261015_204805/raw_preview.ply').points)
tree=cKDTree(prev)
def frame(i,order):
    d=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,n=struct.unpack('<III',d[:12]); mask=np.frombuffer(d[12:12+w*h//8],np.uint8)
    vals=np.frombuffer(d[12+w*h//8+4:12+w*h//8+4+4*n],'<f4')
    m=np.unpackbits(mask,bitorder=order).astype(bool).reshape(h,w)
    v,u=np.nonzero(m); return u.astype(float),v.astype(float),vals.astype(float)
def score(p,u,v,z,sx):
    fx,fy,cx,cy=p
    P=np.c_[sx*(u-cx)*z/fx,-(v-cy)*z/fy,-z]
    d,_=tree.query(P[::10],workers=-1); return np.mean(np.minimum(d,3.0))
best=[]
for order in ('little','big'):
    u,v,z=frame(0,order)
    for sx,x0 in ((-1,[2284.3,2284.3,438.9,401.1]),(1,[1142.2,1142.2,200.6,319.4])):
        r=minimize(score,x0,args=(u,v,z,sx),method='Nelder-Mead',options=dict(maxiter=600,xatol=0.05,fatol=1e-4))
        fx,fy,cx,cy=r.x; P=np.c_[sx*(u-cx)*z/fx,-(v-cy)*z/fy,-z]; d,_=tree.query(P,workers=-1)
        best.append((np.median(d),order,sx,np.round(r.x,2)))
        print(f'bits={order} sx={sx:+d}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  mediana={np.median(d):.3f} mm  p90={np.percentile(d,90):.3f}')
