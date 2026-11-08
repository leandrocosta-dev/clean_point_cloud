"""Fusao orientada CPU/CUDA, com camadas e pontos inconclusivos separados.

    python pipeline/fusao_superficies.py SCAN --saida NOVA --motor cuda

As poses e profundidades originais nao mudam. Normais/visadas orientadas
separam faces opostas antes da media. Resultados exigem inspecao antes do merge.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recursos import orcamento_threads  # noqa: E402
orcamento_threads(1)        # antes do numpy: define OMP/OPENBLAS/MKL/REVO_THREADS pelo modo de recursos
import numpy as np
import open3d as o3d
import psutil
from scipy.spatial import cKDTree

from comparar_fusao_cuda import sha256
sys.path.insert(0, str(Path(__file__).resolve().parent/'revo_frames'))
from revo_projeto import RevoScan


PREPARO_VERSAO = 1   # mudar quando preparar_observacoes mudar o conteudo: invalida o cache_observacoes


def salvar_json(path, value):
    path=Path(path)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp,path)


def salvar_nuvem(path, p, n=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(p, float)))
    if n is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(n, float))
    # Open3D nao grava nuvem vazia. Nao criar um PLY falso.
    if len(p) and not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f'falha ao gravar {path}')


def preparar_observacoes(pasta, saida, calibracao_fixa=None):
    """NPY em disco: xyz, visada, indice de quadro e pixel, sem perder origem."""
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(str(pasta), tmp)
        try:
            if scan.inconsistentes:
                raise ValueError('quadros inconsistentes: revisar antes de fundir')
            calibracao = {}
            for m in sorted(set(scan.modo[scan.tem_prof].tolist())):
                if calibracao_fixa and str(m) in calibracao_fixa:
                    k=tuple(calibracao_fixa[str(m)]['K'])
                    erro=calibracao_fixa[str(m)]['erro_preview_mm']
                    scan.K[m]=k
                else:
                    k, erro = scan.calibrar(m)
                calibracao[str(m)] = dict(K=k, erro_preview_mm=erro)
            total = sum(len(scan.profundidade(i)[3]) for k, i in enumerate(scan.ids) if scan.tem_prof[k])
            if not total:
                raise ValueError('scan sem observacoes')
            dtype = np.dtype([('p', '<f4', 3), ('v', '<f4', 3), ('quadro', '<i4'), ('pixel', '<i4')])
            obs = np.lib.format.open_memmap(saida/'observacoes.npy', mode='w+', dtype=dtype, shape=(total,))
            offset = 0
            frames = []
            for k, i in enumerate(scan.ids):
                if not scan.tem_prof[k]:
                    continue
                p, pix = scan.pontos(k)
                w, h, _, _, _ = scan.profundidade(i)
                v = scan.T[k][:3, 3]-p
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                if not np.isfinite(p).all() or not np.isfinite(v).all():
                    raise ValueError('observacoes nao finitas')
                end = offset+len(p)
                obs['p'][offset:end] = p
                obs['v'][offset:end] = v
                obs['quadro'][offset:end] = k
                obs['pixel'][offset:end] = pix
                frames.append(dict(indice=k, id=int(i), inicio=offset, fim=end, largura=w, altura=h))
                offset = end
            obs.flush()
            del obs
            meta = dict(scan=str(Path(pasta).resolve()), dataset_sha256=sha256(Path(pasta)/'cache/frames.dataset'),
                        preview_sha256=sha256(Path(pasta)/'raw_preview.ply'),
                        observacoes=total, quadros=frames, calibracao=calibracao,
                        referencial='mundo do scan; camera->mundo de FrameInfo, em mm')
            salvar_json(saida/'observacoes.json', meta)
            return meta
        finally:
            scan.c.close()


def reduzir_orientado(obs, frames, voxel, motor, bloco=4_000_000):
    """Cada voxel tem ate oito camadas de visada. Cada quadro vota uma vez/camada.

    Divisoes de lote ocorrem apenas ENTRE quadros: contagem de quadros e exata.
    Guarda uma observacao representativa por surfel; todas permanecem no NPY.
    """
    xp = np
    if motor == 'cuda':
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('CUDA solicitado mas indisponivel')
    p = obs['p']
    lo = p.min(axis=0).astype(float)-voxel
    dims = np.ceil((p.max(axis=0)-lo)/voxel).astype(np.int64)+2
    if int(dims[0])*int(dims[1])*int(dims[2])*8 >= np.iinfo(np.int64).max:
        raise ValueError('grade muito grande para chave int64')
    result = None
    grupos, lote, tamanho = [], [], 0
    for frame in frames:
        lote.append(frame)
        tamanho += frame['fim']-frame['inicio']
        if tamanho >= bloco:
            grupos.append(lote); lote=[]; tamanho=0
    if lote:
        grupos.append(lote)
    for batch in grupos:
        a, b = batch[0]['inicio'], batch[-1]['fim']
        x = xp.asarray(np.asarray(obs['p'][a:b]), dtype=xp.float64)
        v = xp.asarray(np.asarray(obs['v'][a:b]), dtype=xp.float64)
        k = xp.floor((x-xp.asarray(lo))/voxel).astype(xp.int64)
        lado = (v[:, 0]>=0).astype(xp.int64)+2*(v[:, 1]>=0)+4*(v[:, 2]>=0)
        key = ((k[:, 0]*int(dims[1])+k[:, 1])*int(dims[2])+k[:, 2])*8+lado
        u, primeiro, inv = xp.unique(key, return_index=True, return_inverse=True)
        count = xp.bincount(inv, minlength=len(u)).astype(xp.float64)
        data = xp.concatenate((x, v), axis=1)
        sums = xp.stack([xp.bincount(inv, weights=data[:, j], minlength=len(u)) for j in range(6)], axis=1)
        # unique de (surfel, quadro) para nao confundir pixels com visadas independentes.
        nf = len(frames)+1
        # indices de quadro podem incluir quadros sem profundidade: usar maximo real.
        nf = max(f['indice'] for f in frames)+1
        if int(dims[0])*int(dims[1])*int(dims[2])*8*nf >= np.iinfo(np.int64).max:
            raise ValueError('chave quadro/voxel excede int64')
        fk = xp.asarray(obs['quadro'][a:b], dtype=xp.int64)
        pares = xp.unique(key*nf+fk)//nf
        iu = xp.searchsorted(u, pares)
        votos = xp.bincount(iu, minlength=len(u)).astype(xp.float64)
        representante = primeiro.astype(xp.int64)+a
        if result is not None:
            keys = xp.concatenate((result[0], u))
            values = xp.concatenate((result[1], sums))
            counts = xp.concatenate((result[2], count))
            votes = xp.concatenate((result[3], votos))
            reps = xp.concatenate((result[4], representante))
            u, first, inv = xp.unique(keys, return_index=True, return_inverse=True)
            sums = xp.stack([xp.bincount(inv, weights=values[:, j], minlength=len(u)) for j in range(6)], axis=1)
            count = xp.bincount(inv, weights=counts, minlength=len(u))
            votos = xp.bincount(inv, weights=votes, minlength=len(u))
            representante = reps[first]
        result = u, sums, count, votos, representante
        print(f'camadas: {b:,}/{len(obs):,} observacoes, {len(u):,} surfels', flush=True)
    _, s, c, frames_count, rep = result
    def cpu(x):
        return xp.asnumpy(x) if motor == 'cuda' else np.asarray(x)
    pos = cpu(s[:, :3]/c[:, None]).astype(np.float32)
    vis = cpu(s[:, 3:])
    coerencia = np.linalg.norm(vis, axis=1)/cpu(c)
    vis /= np.maximum(np.linalg.norm(vis, axis=1, keepdims=True), 1e-12)
    return pos, vis.astype(np.float32), cpu(c).astype(np.uint32), cpu(frames_count).astype(np.uint32), cpu(rep), coerencia


def refinar_camadas(p, vis, motor='cpu', raio=.3, sigma=.08, max_desloc=.08, passadas=2):
    """Bilateral na normal; nao mistura normais opostas e preserva cantos ambiguos."""
    # Normais sobre centroides mais espacados: nos dados brutos o ruido pode
    # dominar os vizinhos imediatos e dar PCA isotropico em quase todo o scan.
    lado = np.argmax(np.abs(vis), axis=1)*2+(vis[np.arange(len(vis)),np.argmax(np.abs(vis),axis=1)]<0)
    keys = np.c_[np.floor(p/max(.5,6*sigma)).astype(np.int64), lado]
    _, inv, cnt = np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    centro = np.stack([np.bincount(inv,weights=p[:,j])/cnt for j in range(3)],axis=1)
    vc = np.stack([np.bincount(inv,weights=vis[:,j]) for j in range(3)],axis=1)
    vc /= np.maximum(np.linalg.norm(vc,axis=1,keepdims=True),1e-12)
    arv = cKDTree(centro)
    nc = np.empty_like(centro)
    ac = np.zeros(len(centro),bool)
    rn=max(1.2,4*raio)
    for a in range(0, len(centro), 30000):
        q = centro[a:a+30000]
        d, idx = arv.query(q, k=40, distance_upper_bound=rn, workers=12)
        valid = np.isfinite(d) & (idx < len(centro))
        idx = np.where(valid, idx, 0)
        # Camadas com visadas opostas nao participam do ajuste do plano.
        valid &= np.einsum('nkj,nj->nk', vc[idx], vc[a:a+len(q)]) > .15
        w = np.where(valid, np.exp(-np.minimum(d, rn)**2/(2*(rn/2)**2)), 0)
        sw = np.maximum(w.sum(axis=1), 1e-12)
        delta = centro[idx]-q[:, None, :]
        local = (delta*w[..., None]).sum(axis=1)/sw[:, None]
        y = delta-local[:, None, :]
        cov = np.einsum('nk,nki,nkj->nij', w, y, y)/sw[:, None, None]
        ev, en = np.linalg.eigh(cov)
        n = en[:, :, 0]
        n *= np.where(np.einsum('ij,ij->i', n, vc[a:a+len(q)])<0, -1, 1)[:, None]
        nc[a:a+len(q)] = n
        ac[a:a+len(q)] = (valid.sum(axis=1)<6) | (ev[:, 0]/np.maximum(ev[:, 1], 1e-12)>.35)
    # inv liga cada observacao ao seu centroide de mesma camada de visada.
    normals=nc[inv].astype(np.float32)
    ambiguo=ac[inv]
    xp = np
    if motor == 'cuda':
        import cupy as xp
    inicial = p.copy()
    out = p.copy()
    for _ in range(passadas):
        arv = cKDTree(out)
        destino = out.copy()
        for a in range(0, len(p), 30000):
            q = out[a:a+30000]
            d, idx = arv.query(q, k=64, distance_upper_bound=raio, workers=12)
            valid = np.isfinite(d) & (idx < len(p))
            idx = np.where(valid, idx, 0)
            delta = xp.asarray(out[idx]-q[:, None, :])
            n = xp.asarray(normals[a:a+len(q)])
            h = xp.einsum('nkj,nj->nk', delta, n)
            cos = xp.einsum('nkj,nj->nk', xp.asarray(normals[idx]), n)
            # Salto de altura e de normal define uma fronteira que nao e promediada.
            valid = xp.asarray(valid) & (cos>.9) & (xp.abs(h)<2*sigma)
            dd = xp.asarray(np.minimum(d, raio))
            tang = xp.maximum(dd*dd-h*h, 0)
            w = xp.where(valid, xp.exp(-tang/(2*(raio/2)**2)-h*h/(2*sigma*sigma)), 0)
            step = (w*h).sum(axis=1)/xp.maximum(w.sum(axis=1), 1e-12)
            step = xp.clip(step, -max_desloc/passadas, max_desloc/passadas)
            step = xp.asnumpy(step) if motor == 'cuda' else step
            step[ambiguo[a:a+len(q)]] = 0
            destino[a:a+len(q)] = q+step[:, None]*normals[a:a+len(q)]
        out = destino
    return out, normals, ambiguo, np.linalg.norm(out-inicial, axis=1)


def representantes(p, n, count, voxel):
    """Reamostra sem misturar normais opostas nem mover o representante escolhido."""
    axis=np.argmax(np.abs(n),axis=1)
    side=axis*2+(n[np.arange(len(n)),axis]<0)
    keys=np.c_[np.floor(p/voxel).astype(np.int64),side]
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    ordem=np.lexsort((-count.astype(np.int64),inv))
    _,first=np.unique(inv[ordem],return_index=True)
    return ordem[first]


def executar(scan, saida, motor='cuda', voxel=.08, min_quadros=3, calibracao_fixa=None, cache=None):
    """cache=None: prepara as observacoes em <saida>/origem (comportamento original).
    cache=True ou pasta: reutiliza cache_observacoes (chave = hashes do dataset/preview + versao + calibracao)
    e vincula <saida>/origem a ele (hardlink), entao quem le origem/ continua funcionando."""
    if voxel <= 0 or not np.isfinite(voxel) or min_quadros<1:
        raise ValueError('voxel positivo e min_quadros >=1')
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=False)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from recursos import aplicar_prioridade
        aplicar_prioridade()
    except Exception:
        pass
    t = time.perf_counter()
    info_cache = None
    if cache:
        from cache_observacoes import obter, vincular
        pasta, meta, reuso = obter(Path(scan), None if cache is True else cache, calibracao_fixa)
        vincular(pasta, saida/'origem')
        info_cache = dict(pasta=str(pasta), reutilizado=reuso, segundos=time.perf_counter()-t)
    else:
        meta = preparar_observacoes(Path(scan), saida/'origem', calibracao_fixa)
    obs = np.load(saida/'origem/observacoes.npy', mmap_mode='r')
    p, vis, count, nf, reps, coerencia = reduzir_orientado(obs, meta['quadros'], voxel, motor)
    p0 = p.copy()
    p, n, ambiguo, desl = refinar_camadas(p, vis, motor, raio=max(.2, 3*voxel), sigma=voxel)
    sel=representantes(p,n,count,voxel)
    p,p0,n,vis,count,nf,reps,coerencia,ambiguo,desl=[x[sel] for x in (p,p0,n,vis,count,nf,reps,coerencia,ambiguo,desl)]
    certo = (nf>=min_quadros) & ~ambiguo & (coerencia>.5)
    salvar_nuvem(saida/'nuvem.ply', p, n)
    salvar_nuvem(saida/'nuvem_confirmada.ply', p[certo], n[certo])
    salvar_nuvem(saida/'nuvem_incertos.ply', p[~certo], n[~certo])
    np.savez(saida/'superficies.npz', pontos=p, pontos_medidos=p0, normais=n, visadas=vis,
             observacoes=count, quadros=nf, representante=reps, confirmado=certo,
             deslocamento_mm=desl)
    rel = dict(versao_motor=1, codigo_sha256=sha256(Path(__file__)), motor=motor, voxel_mm=voxel, min_quadros=min_quadros,
               entrada=meta, pontos=len(p), confirmados=int(certo.sum()), incertos=int((~certo).sum()),
               deslocamento_p95_mm=float(np.percentile(desl, 95)),
               deslocamento_max_mm=float(desl.max()), segundos=time.perf_counter()-t,
               normal_voxel_mm=max(.5,6*voxel),normal_raio_mm=max(1.2,4*max(.2,3*voxel)),
               nuvem_sha256=sha256(saida/'nuvem.ply'), cache_observacoes=info_cache,
               estado='revisao_pendente',
               limites='Camadas por octante de visada: protecao parcial, nao garantia universal de separacao. '
                       'Quadros distintos nao equivalem a scans/poses independentes. Incertos preservados.')
    salvar_json(saida/'fusao.json', rel)
    print(json.dumps({k:v for k,v in rel.items() if k!='entrada'}, indent=2), flush=True)
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scan')
    ap.add_argument('--saida', required=True)
    ap.add_argument('--motor', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--voxel', type=float, default=.08)
    ap.add_argument('--min-quadros', type=int, default=3)
    ap.add_argument('--cache', nargs='?', const=True, default=None,
                    help='reutiliza observacoes em cache (pasta opcional; padrao pipeline/output/cache_obs)')
    a = ap.parse_args()
    executar(a.scan, a.saida, a.motor, a.voxel, a.min_quadros, cache=a.cache)


if __name__=='__main__':
    main()
