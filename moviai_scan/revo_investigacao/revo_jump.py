import sys,sqlite3,struct,json,numpy as np,open3d as o3d,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S=sys.argv[1]
c=sqlite3.connect(S+'/frames.sqlite')
fx,fy,cx,cy=1140.4,1141.0,232.8,319.5
def frame(i):
    d=c.execute('select data from FrameRGBD where id=?',(i,)).fetchone()[0]
    w,h,n=struct.unpack('<III',d[:12]); mask=np.frombuffer(d[12:12+w*h//8],np.uint8)
    vals=np.frombuffer(d[12+w*h//8+4:12+w*h//8+4+4*n],'<f4').astype(float)
    v,u=np.nonzero(np.unpackbits(mask,bitorder='little').astype(bool).reshape(h,w))
    return np.c_[(u-cx)*vals/fx,-(v-cy)*vals/fy,-vals]
rows=[(i,np.frombuffer(b[16:144],'<f8').reshape(4,4)) for i,b in c.execute('select id,bin from FrameInfo order by id') if (i>>32)==0]
r=json.load(open('/home/claude/moviai_scan/saida_real/3cd01b6f-raw_preview_relatorio.json'))
n=np.array(r['base']['n']); d=r['base']['d']; u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
cen=np.array([166.45,-28.30])
# giro em torno do eixo do cilindro (normal da base) e translacao por passo
info=[]
for k,((ia,A),(ib,B)) in enumerate(zip(rows,rows[1:])):
    Rr=B[:3,:3]@A[:3,:3].T
    ang=np.degrees(np.arccos(np.clip((np.trace(Rr)-1)/2,-1,1)))
    info.append((k+1,ib,ang))
angs=np.array([a for _,_,a in info])
top=np.argsort(-angs)[:6]
print('maiores passos no segmento 0:')
for t in sorted(top): print(f'  quadro #{info[t][0]} (id {info[t][1]}): {info[t][2]:.2f} gr   vizinhos: {angs[max(0,t-2):t+3].round(2)}')
# angulo acumulado da mesa em torno da normal da base
def az(T):
    # direcao do eixo optico da camera projetada no plano da base
    z=T[:3,:3]@np.array([0,0,-1.0]); return np.degrees(np.arctan2(z@v,z@u))
azs=np.unwrap(np.radians([az(T) for _,T in rows])); azs=np.degrees(azs)
fig,ax=plt.subplots(1,2,figsize=(15,4.8),dpi=100)
ax[0].plot(angs,lw=1); ax[0].set_title('Segmento 0: giro entre quadros consecutivos'); ax[0].set_xlabel('quadro'); ax[0].set_ylabel('graus')
ax[1].plot(azs,lw=1.2); ax[1].set_title('Segmento 0: azimute acumulado da camera em torno da peca'); ax[1].set_xlabel('quadro'); ax[1].set_ylabel('graus')
fig.tight_layout(); fig.savefig('/home/claude/difusor_passos_seg0.png')
np.save(S+'/seg0_angs.npy',angs)
