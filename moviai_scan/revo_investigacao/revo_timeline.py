import sys,sqlite3,struct,json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
n=np.array(r['base']['n']); d=r['base']['d']; u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
cen=np.array([166.45,-28.30]); R=22.91
rows=[(i,np.frombuffer(b[16:144],'<f8').reshape(4,4)) for i,b in c.execute('select id,bin from FrameInfo order by id')]
out=[]
for k,(i,T) in enumerate(rows):
    P=frame(i)[::2]; W=P@T[:3,:3].T+T[:3,3]; q=np.c_[W@u,W@v,W@n+d]
    rr=np.linalg.norm(q[:,:2]-cen,axis=1); sl=q[(rr>R+2)&(q[:,2]>48)&(q[:,2]<78)]
    fa=fb=0
    if len(sl)>80:
        t=cKDTree(sl[:,:2]); 
        for j in t.query_ball_point(sl[::2,:2],2.0):
            if len(j)<8: continue
            Q=sl[j,:2]-sl[j,:2].mean(0); ev,V=np.linalg.eigh(Q.T@Q)
            if ev[1]<8*ev[0]: continue
            a=np.degrees(np.arctan2(V[1,1],V[0,1]))%90
            if abs((a-48.5+45)%90-45)<4: fa+=1
            elif abs((a-33.5+45)%90-45)<4: fb+=1
    out.append((k,i>>32,i&0xffffffff,len(sl),fa,fb))
out=np.array(out)
np.save(S+'/timeline.npy',out)
fig,ax=plt.subplots(figsize=(16,4.5),dpi=100)
x=np.arange(len(out))
ax.bar(x,out[:,4],color='#d0021b',width=1,label='paredes a 48,5 gr (familia A)')
ax.bar(x,-out[:,5],color='#1a5fd0',width=1,label='paredes a 33,5 gr (familia B)')
for b in np.where(np.diff(out[:,1])!=0)[0]: ax.axvline(b+0.5,color='k',ls='--',lw=1)
ax.set_xlabel('quadro (ordem de captura; tracejado = novo segmento)'); ax.set_ylabel('trechos de parede'); ax.legend()
ax.set_title('Qual orientacao do plugue cada quadro enxerga')
fig.tight_layout(); fig.savefig(S+'/difusor_linha_tempo.png')
for k,s,idx,ns,fa,fb in out:
    if fa+fb>15: print(f'#{k:3d} seg{s} idx{idx:3d}: A={fa:4d} B={fb:4d} -> {"A" if fa>2*fb else ("B" if fb>2*fa else "misto")}')
