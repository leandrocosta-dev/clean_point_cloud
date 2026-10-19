import sys,sqlite3,json,numpy as np,shutil
S,P=sys.argv[1],sys.argv[2]
ds=P+'/data/20261015_204805/cache/frames.dataset'
tmp=S+'/frames_corr.sqlite'; shutil.copy(ds,tmp)
with open(tmp,'r+b') as h: h.write(b'SQLite format 3\x00')
r=json.load(open('/home/claude/moviai_scan/saida_real/3cd01b6f-raw_preview_relatorio.json'))
n=np.array(r['base']['n']); d0=r['base']['d']; u=np.cross(n,[1,0,0]); u/=np.linalg.norm(u); v=np.cross(n,u)
Bm=np.c_[u,v,n].T; cen=np.array([166.45,-28.30]); p0=Bm.T@(np.array([cen[0],cen[1],0.0])-np.array([0,0,d0]))
def Rw(deg):
    a=np.radians(deg); k=n; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Rm=np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K; M=np.eye(4); M[:3,:3]=Rm; M[:3,3]=p0-Rm@p0; return M
fixes=np.load(S+'/fixes.npy')
c=sqlite3.connect(tmp); rows=c.execute('select id,bin from FrameInfo order by id').fetchall()
nmod=0
for k,(i,b) in enumerate(rows):
    if fixes[k]==0: continue
    T=np.frombuffer(b[16:144],'<f8').reshape(4,4); T2=Rw(fixes[k])@T
    nb=b[:16]+T2.astype('<f8').tobytes()+b[144:]; assert len(nb)==len(b)
    c.execute('update FrameInfo set bin=? where id=?',(nb,i)); nmod+=1
c.commit(); c.execute('vacuum') if False else None; c.close()
with open(tmp,'r+b') as h: h.write(b'RPAlgo_file_sys\x00')
shutil.copy(tmp,ds)
# verifica: rele e compara
chk=S+'/chk.sqlite'; shutil.copy(ds,chk)
with open(chk,'r+b') as h: h.write(b'SQLite format 3\x00')
c=sqlite3.connect(chk); rows2=c.execute('select id,bin from FrameInfo order by id').fetchall()
err=max(np.abs(np.frombuffer(b2[16:144],'<f8').reshape(4,4)-Rw(fixes[k])@np.frombuffer(b[16:144],'<f8').reshape(4,4)).max() for k,((i,b),(i2,b2)) in enumerate(zip(rows,rows2)))
print('quadros com pose corrigida:',nmod,'| erro de gravacao:',err,'| cabecalho:',open(ds,'rb').read(16))
