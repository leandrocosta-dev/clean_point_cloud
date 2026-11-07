"""Audita frames limpos pixel a pixel, IDs, poses e integridade SQLite.

    python pipeline/auditar_projeto.py ORIGINAL LIMPO --comparar LIMPEZA_ANTERIOR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'revo_frames'))
from revo_projeto import RevoScan


def mapa_ids(pasta):
    p = Path(pasta)/'identificadores.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def auditar(original, limpo, comparar=None):
    original, limpo = Path(original), Path(limpo)
    mapa = mapa_ids(limpo)
    anterior = mapa_ids(comparar) if comparar else {}
    projetos = list(limpo.glob('*.revo'))
    if len(projetos) != 1:
        raise ValueError('esperado um unico .revo na entrega')
    r = json.loads(projetos[0].read_text(encoding='utf-8'))
    ids = {n['guid'] for n in r['nodes']}
    assert len(ids) == len(r['nodes']), 'IDs duplicados'
    resultado = {}
    for pasta in sorted((original/'data').iterdir()):
        if not pasta.is_dir():
            continue
        ident = mapa.get(pasta.name, pasta.name)
        assert ident in ids, f'scan ausente: {pasta.name}'
        destino = limpo/'data'/ident
        if not (pasta/'raw_preview.ply').exists():
            iguais = True
            for arquivo in pasta.rglob('*'):
                if not arquivo.is_file() or arquivo.suffix == '.rvproj':
                    continue
                copia = destino/arquivo.relative_to(pasta)
                with arquivo.open('rb') as a, copia.open('rb') as b:
                    iguais &= hashlib.file_digest(a,'sha256').digest() == hashlib.file_digest(b,'sha256').digest()
            assert iguais, 'dados de scan sem quadros foram alterados'
            resultado[pasta.name] = dict(id_entrega=ident, sem_quadros=True, dados_copiados_identicos=True)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            bancos = []
            try:
                def abrir(caminho, nome):
                    work=Path(tmp)/nome;work.mkdir()
                    s=RevoScan(str(caminho),str(work));bancos.append(s);return s
                so=abrir(pasta,'original');sl=abrir(destino,'limpo')
                sb=abrir(Path(comparar)/'data'/anterior.get(pasta.name,pasta.name),'anterior') if comparar else None
                assert so.ids == sl.ids, 'IDs de quadros alterados'
                assert so.c.execute('select id,bin from FrameInfo order by id').fetchall() == sl.c.execute('select id,bin from FrameInfo order by id').fetchall(), 'FrameInfo ou poses alteradas'
                antes=depois=quadros=vazios=restaurados=extras=0
                for k,i in enumerate(so.ids):
                    if not so.tem_prof[k]:
                        continue
                    w,h,m,z,cauda=so.profundidade(i)
                    antes+=len(z);quadros+=1
                    novo=sl.profundidade(i)
                    ml=np.zeros_like(m)
                    if novo is None:
                        vazios+=1
                    else:
                        wl,hl,ml,zl,cl=novo
                        assert (w,h,cauda)==(wl,hl,cl), 'cabecalho de quadro alterado'
                        assert not (ml & ~m).any(), 'pixel novo'
                        assert np.array_equal(z[ml[m]],zl), 'profundidade alterada'
                        depois+=len(zl)
                    if sb:
                        antigo=sb.profundidade(i)
                        mb=antigo[2] if antigo else np.zeros_like(m)
                        restaurados+=int((ml & ~mb).sum())
                        extras+=int((mb & ~ml).sum())
                integ=sl.c.execute('pragma integrity_check').fetchone()[0]
                assert integ=='ok', integ
                resultado[pasta.name]=dict(id_entrega=ident, antes=antes, depois=depois,
                    quadros=quadros, vazios=vazios, poses_identicas=True, profundidades_preservadas=True,
                    integridade=integ, mantidos_so_novo=restaurados if sb else None,
                    removidos_so_novo=extras if sb else None)
            finally:
                for s in bancos:s.c.close()
    (limpo/'auditoria_integridade.json').write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    return resultado


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('original')
    ap.add_argument('limpo')
    ap.add_argument('--comparar', help='projeto de limpeza anterior, opcional')
    a=ap.parse_args()
    print(json.dumps(auditar(a.original,a.limpo,a.comparar),indent=2))


if __name__=='__main__':main()
