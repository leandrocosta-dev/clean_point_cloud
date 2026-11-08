"""Avalia corte de apoio por scan e recuperacao por outras vistas, sem editar frames.

    python pipeline/avaliar_corte_apoio.py ORIGINAL LIMPO --poses alinhamento.json --saida NOVA

Plano estimado no preview ORIGINAL (a limpeza pode ter apagado a mesa).
Cobertura medida contra amostras dos outros scans apos o mesmo corte.
Nao classifica automaticamente os pontos retirados como mesa ou peca.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','REVO_THREADS'):
    os.environ[_k]='12'
import numpy as np
import open3d as o3d
import psutil
from scipy.spatial import cKDTree

from fusao_superficies import salvar_json
from limpar_por_placa import _area_plano, amostra_mundo
sys.path.insert(0,str(Path(__file__).resolve().parent/'revo_frames'))
from revo_projeto import RevoScan,ler_ply,plano_ransac


def plano_de_apoio(original,peca):
    """Maior area dentre quatro planos; rejeita corte atravessando a selecao."""
    original=np.asarray(original,float)
    resto=original.copy();candidatos=[]
    for i in range(4):
        if len(resto)<100:break
        n,d,_=plano_ransac(resto,thr=.4,it=1500,seed=i)
        n=n/np.linalg.norm(n)
        if np.median(peca@n+d)<0:n,d=-n,-d
        res=original@n+d
        inliers=np.abs(res)<.4
        mad=float(1.4826*np.median(np.abs(res[inliers]-np.median(res[inliers]))))
        candidatos.append(dict(n=n.tolist(),d=float(d),area_mm2=_area_plano(original,n,d,thr=.4),
            inliers=int(inliers.sum()),dispersao_mad_mm=mad,
            selecao_abaixo_menos_05_pct=float(100*np.mean(peca@n+d<-.5))))
        resto=resto[np.abs(resto@n+d)>=.4]
    if not candidatos:raise ValueError('nenhum plano de apoio encontrado')
    best=max(candidatos,key=lambda c:c['area_mm2'])
    if best['selecao_abaixo_menos_05_pct']>5 or best['dispersao_mad_mm']>.25:
        raise ValueError('plano incerto ou atravessando a selecao; nao autorizar corte')
    return best,candidatos


def cobertura_removidos(p,mascara,outros,tolerancias):
    rem=p[mascara]
    if not len(rem):return dict(removidos=0,cobertura={})
    if not len(outros):return dict(removidos=len(rem),cobertura={str(t):0 for t in tolerancias})
    dist=cKDTree(outros).query(rem,workers=12)[0]
    return dict(removidos=len(rem),distancia_outros_mediana_mm=float(np.median(dist)),
        cobertura={str(t):float(100*np.mean(dist<=t)) for t in tolerancias},
        sem_apoio_na_maior_tolerancia=int((dist>max(tolerancias)).sum()))


def executar(original,limpo,poses,saida,alturas=(.5,1.),tolerancias=(.3,.5)):
    if any(not np.isfinite(x) or x<=0 for x in (*alturas,*tolerancias)):
        raise ValueError('alturas e tolerancias devem ser positivas')
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    original,limpo,saida=map(Path,(original,limpo,saida))
    saida.mkdir(parents=True,exist_ok=False)
    doc=json.loads(Path(poses).read_text())
    transforms=doc.get('poses_usadas',doc)
    mapa=json.loads((limpo/'identificadores.json').read_text())
    scans=[g for g in transforms if g in mapa and (original/'data'/g/'raw_preview.ply').exists()]
    if len(scans)<2:raise ValueError('necessarios dois scans com poses e IDs correspondentes')
    pontos={};mundos={};heights={};rel={}
    for g in scans:
        with tempfile.TemporaryDirectory() as tmp:
            scan=RevoScan(str(limpo/'data'/mapa[g]),tmp)
            try:
                calib={}
                for m in sorted(set(scan.modo[scan.tem_prof].tolist())):
                    k,e=scan.calibrar(m);calib[str(m)]=dict(K=k,erro_preview_mm=e)
                p=amostra_mundo(scan,4,4).astype(float)
            finally:scan.c.close()
        # Centroides reduzem ruido e redundancia; nao sao amostra uniforme de area.
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).voxel_down_sample(.15)
        p=np.asarray(cloud.points)
        _,raw=ler_ply(original/'data'/g/'raw_preview.ply')
        plano,candidatos=plano_de_apoio(raw[:,:3],p)
        h=p@np.array(plano['n'])+plano['d']
        T=np.asarray(transforms[g],float)
        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):
            raise ValueError('pose invalida')
        pontos[g]=p;heights[g]=h;mundos[g]=p@T[:3,:3].T+T[:3,3]
        rel[g]=dict(plano=plano,candidatos=candidatos,calibracao=calib,centroides=len(p),cortes={})
        np.savez(saida/f'{g}.npz',pontos_scan=p,pontos_comuns=mundos[g],altura_mm=h)
        print(g,'plano',plano,'centroides',len(p),flush=True)
    for altura in alturas:
        for g in scans:
            outros=np.vstack([mundos[j][heights[j]>altura] for j in scans if j!=g])
            rem=heights[g]<=altura
            m=cobertura_removidos(mundos[g],rem,outros,tolerancias)
            m.update(mantidos=int((~rem).sum()),remocao_pct=float(100*np.mean(rem)))
            # Faixa adicional para comparar 1 mm contra 0,5 mm na mesma referencia.
            if altura>min(alturas):
                extra=(heights[g]>min(alturas)) & (heights[g]<=altura)
                m['faixa_adicional']=cobertura_removidos(mundos[g],extra,outros,tolerancias)
            rel[g]['cortes'][str(altura)]=m
    resultado=dict(scans=rel,alturas_mm=list(alturas),tolerancias_mm=list(tolerancias),
        origem=str(original.resolve()),base_limpa=str(limpo.resolve()),
        aviso='Medida amostrada de proximidade entre scans, nao prova de pertencimento a peca. '
              'Regioes removidas incluem mesa residual e superficie real. Falta de cobertura impede presumir recuperacao. '
              'Planos requerem conferencia geometrica antes de aplicar. Nenhum frame foi alterado.')
    salvar_json(saida/'avaliacao.json',resultado)
    return resultado


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('original');ap.add_argument('limpo')
    ap.add_argument('--poses',required=True);ap.add_argument('--saida',required=True)
    ap.add_argument('--alturas',nargs='+',type=float,default=[.5,1.])
    ap.add_argument('--tolerancias',nargs='+',type=float,default=[.3,.5])
    a=ap.parse_args()
    print(json.dumps(executar(a.original,a.limpo,a.poses,a.saida,a.alturas,a.tolerancias),indent=2))


if __name__=='__main__':main()
