import sys,json,numpy as np,open3d as o3d,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S=sys.argv[1]
r=json.load(open('C:/Users/leandro.soliddesigne/Music/IA/34.nuvem_pontos/moviai_scan/saida_atual/raw_preview_relatorio.json'))
n=np.array(r['base']['n']); d0=r['base']['d']; u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
cen=np.array([166.45,-28.30]); R=22.91
fig,ax=plt.subplots(2,3,figsize=(20,13),dpi=100)
for row,name in enumerate(('fusao_original','fusao_corrigida')):
    W=np.asarray(o3d.io.read_point_cloud(f'{S}/{name}.ply').points); q=np.c_[W@u,W@v,W@n+d0]
    rr=np.linalg.norm(q[:,:2]-cen,axis=1)
    for k,(h0,h1) in enumerate([(50,58),(62,70)]):
        a=ax[row,k]; m=(q[:,2]>h0)&(q[:,2]<h1)&(rr>R-3); s=q[m]
        a.scatter(s[:,0],s[:,1],s=0.15,c='#333333',linewidths=0,rasterized=True)
        a.set_aspect('equal'); a.set_xlim(115,195); a.set_ylim(-58,22)
        a.set_title(f'{"ORIGINAL" if row==0 else "CORRIGIDA"}: fatia {h0} a {h1} mm',fontsize=12)
    a=ax[row,2]; t=q[(rr<R+0.5)&(q[:,2]>79.5)]
    sc=a.scatter(t[:,0],t[:,1],s=0.15,c=t[:,2],cmap='turbo',vmin=79.5,vmax=82.5,linewidths=0,rasterized=True)
    a.set_aspect('equal'); a.set_title(f'{"ORIGINAL" if row==0 else "CORRIGIDA"}: topo (pontos acima de 79,5 mm)',fontsize=12)
    # angulos dos rasgos
    th=np.degrees(np.arctan2(t[:,1]-cen[1],t[:,0]-cen[0]))%360; tr=np.linalg.norm(t[:,:2]-cen,axis=1)
    H,E=np.histogram(th[(tr>8)&(tr<15)],bins=360,range=(0,360))
    from scipy.signal import find_peaks
    Hs=np.convolve(np.r_[H[-3:],H,H[:3]],np.ones(7)/7,'valid')
    pk,_=find_peaks(-Hs,distance=20,prominence=Hs.mean()*0.3)
    print(name,'rasgos detectados:',len(pk),'angulos',E[pk].astype(int))
fig.suptitle('Fusao em resolucao completa a partir dos quadros brutos',fontsize=14)
fig.tight_layout(); fig.savefig(S+'/difusor_antes_depois.png')
