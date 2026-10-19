import sys,json,numpy as np,open3d as o3d,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
S=sys.argv[1]
r=json.load(open('/home/claude/moviai_scan/saida_real/3cd01b6f-raw_preview_relatorio.json'))
n=np.array(r['base']['n']); d=r['base']['d']; u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
loc=lambda p: np.c_[p@u,p@v,p@n+d]
c=np.array([166.45,-28.30]); R=22.91
cores={0:'#1a5fd0',1:'#d0021b',2:'#1a9e5a'}
fig,ax=plt.subplots(1,4,figsize=(22,6.2),dpi=100)
for s in (0,1,2):
    q=loc(np.asarray(o3d.io.read_point_cloud(f'{S}/seg{s}.ply').points))
    rr=np.linalg.norm(q[:,:2]-c,axis=1)
    sl=q[(rr>R+2)&(q[:,2]>58)&(q[:,2]<70)]; cyl=q[(np.abs(rr-R)<1.2)&(q[:,2]>58)&(q[:,2]<70)]
    top=q[(rr<R)&(q[:,2]>78)]
    # direcoes das paredes
    ang=[]
    if len(sl)>50:
        t=cKDTree(sl[:,:2])
        for j in t.query_ball_point(sl[:,:2],2.0):
            if len(j)<8: continue
            P=sl[j,:2]-sl[j,:2].mean(0); ev,V=np.linalg.eigh(P.T@P)
            if ev[1]>8*ev[0]: ang.append(np.degrees(np.arctan2(V[1,1],V[0,1]))%90)
    h,e=np.histogram(ang,bins=90,range=(0,90))
    pk=e[np.argsort(-np.convolve(np.r_[h[-2:],h,h[:2]],np.ones(5)/5,'valid'))[:1]]+0.5 if len(ang) else []
    print(f'seg {s}: pontos no bloco (58-70mm) {len(sl)}, topo {len(top)}, direcao dominante das paredes {np.round(pk,1)} gr (mod 90)')
    for a,(nm) in ((ax[s],f'segmento {s}'),(ax[3],'todos')):
        a.scatter(cyl[:,0],cyl[:,1],s=0.3,c='#bbbbbb',linewidths=0)
        a.scatter(sl[:,0],sl[:,1],s=1.2,c=cores[s],linewidths=0,label=f'seg {s}')
    ax[s].set_title(f'Segmento {s}: fatia 58 a 70 mm')
for a in ax: a.set_aspect('equal'); a.set_xlim(110,195); a.set_ylim(-58,25)
ax[3].legend(markerscale=8); ax[3].set_title('Os 3 segmentos sobrepostos')
fig.tight_layout(); fig.savefig('/home/claude/difusor_segmentos.png')
