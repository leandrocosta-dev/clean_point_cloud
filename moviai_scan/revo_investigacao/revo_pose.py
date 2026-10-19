import sys,sqlite3,struct,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
S=sys.argv[1]
c=sqlite3.connect(S+'/frames.sqlite')
prev=np.asarray(o3d.io.read_point_cloud(S+'/revo_proj/Project20261015_204752_20261015_204805_Limpo/data/20261015_204805/raw_preview.ply').points)
tree=cKDTree(prev)
fx,fy,cx,cy=1140.4,1141.0,232.8,319.5
def frame(i):
    d=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,n=struct.unpack('<III',d[:12]); mask=np.frombuffer(d[12:12+w*h//8],np.uint8)
    vals=np.frombuffer(d[12+w*h//8+4:12+w*h//8+4+4*n],'<f4').astype(float)
    v,u=np.nonzero(np.unpackbits(mask,bitorder='little').astype(bool).reshape(h,w))
    return np.c_[(u-cx)*vals/fx,-(v-cy)*vals/fy,-vals]
def pose(i):
    b=c.execute('select bin from FrameInfo where id=?',(i,)).fetchone()[0]
    return np.frombuffer(b[16:144],'<f8').reshape(4,4)
ids=[r[0] for r in c.execute('select id from FrameInfo order by id')]
print('ids: primeiro',ids[:5],'ultimo',ids[-3:],'total',len(ids))
for i in (ids[50],ids[150],ids[250],ids[350]):
    P=frame(i); T=pose(i)
    out=[]
    for nome,M in (('T',T),('inv(T)',np.linalg.inv(T))):
        Q=P@M[:3,:3].T+M[:3,3]; d,_=tree.query(Q[::10],workers=-1); out.append(f'{nome}: {np.median(d):.3f} mm')
    ang=np.degrees(np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1)))
    print(f'quadro {i}: giro {ang:6.1f} gr | '+' | '.join(out))
