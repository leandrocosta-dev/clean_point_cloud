import sys,sqlite3,struct,numpy as np,open3d as o3d
S=sys.argv[1]
c=sqlite3.connect(S+'/frames.sqlite')
fx,fy,cx,cy=1140.4,1141.0,232.8,319.5
def frame(i):
    d=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,n=struct.unpack('<III',d[:12]); mask=np.frombuffer(d[12:12+w*h//8],np.uint8)
    vals=np.frombuffer(d[12+w*h//8+4:12+w*h//8+4+4*n],'<f4').astype(float)
    v,u=np.nonzero(np.unpackbits(mask,bitorder='little').astype(bool).reshape(h,w))
    return np.c_[(u-cx)*vals/fx,-(v-cy)*vals/fy,-vals]
def pose(b): return np.frombuffer(b[16:144],'<f8').reshape(4,4)
rows=c.execute('select id,bin from FrameInfo order by id').fetchall()
segs={}
for i,b in rows: segs.setdefault(i>>32,[]).append((i,pose(b),struct.unpack('<IIIi',b[:16])))
for s,L in segs.items():
    idx=[i&0xffffffff for i,_,_ in L]
    # giro relativo entre quadros consecutivos (mesa giratoria)
    steps=[]; 
    for (_,A,_),(_,B,_) in zip(L,L[1:]):
        R=np.linalg.inv(A)[:3,:3]@B[:3,:3]; steps.append(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
    steps=np.array(steps)
    print(f'segmento {s}: {len(L)} quadros, indices {idx[0]}..{idx[-1]} | giro por passo mediana {np.median(steps):.2f} gr, max {steps.max():.2f} gr | giro total aprox {steps.sum():.0f} gr')
    print('   campo hdr[3] (amostra):',[h[3] for _,_,h in L[:10]])
# nuvem por segmento (amostrada) para analise
for s,L in segs.items():
    acc=[]
    for i,T,_ in L:
        P=frame(i); P=P[::3]; acc.append(P@T[:3,:3].T+T[:3,3])
    A=np.vstack(acc); pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A)).voxel_down_sample(0.4)
    o3d.io.write_point_cloud(f'{S}/seg{s}.ply',pc); print('seg',s,'pontos',len(pc.points))
