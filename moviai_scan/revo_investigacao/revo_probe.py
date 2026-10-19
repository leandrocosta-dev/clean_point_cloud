import sys,sqlite3,struct,numpy as np,open3d as o3d, itertools
from scipy.spatial import cKDTree
S=sys.argv[1]
c=sqlite3.connect(S+'/frames.sqlite')
prev=np.asarray(o3d.io.read_point_cloud(S+'/revo_proj/Project20261015_204752_20261015_204805_Limpo/data/20261015_204805/raw_preview.ply').points)
tree=cKDTree(prev)
def frame(i,order):
    d=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,n=struct.unpack('<III',d[:12]); mask=np.frombuffer(d[12:12+w*h//8],np.uint8)
    vals=np.frombuffer(d[12+w*h//8+4:12+w*h//8+4+4*n],'<f4')
    m=np.unpackbits(mask,bitorder=order).astype(bool).reshape(h,w)
    v,u=np.nonzero(m); return u.astype(float),v.astype(float),vals.astype(float),w,h
res=[]
for order in ('little','big'):
    u,v,z,w,h=frame(0,order)
    for (fx,cx,cy,tag) in [(2284.312,401.123,638.884,'Pl'),(2284.312/2,401.123/2,638.884/2,'Pl/2'),(2288.8,813.17/2,652.5/2,'mapL/2'),
                           (2284.312,638.884-200,401.123,'troca'),(1142.156,w/2,h/2,'centro f/2'),(2284.312,w/2,h/2,'centro f')]:
        X=(u-cx)*z/fx; Y=(v-cy)*z/fx
        for sx,sy,sz in itertools.product((1,-1),repeat=3):
            P=np.c_[sx*X,sy*Y,sz*z]
            d,_=tree.query(P[::20],workers=-1)
            res.append((np.median(d),order,tag,(sx,sy,sz)))
res.sort()
for r in res[:8]: print(f'{r[0]:8.3f} mm  bits={r[1]:6s} K={r[2]:10s} sinais={r[3]}')
