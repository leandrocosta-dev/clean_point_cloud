"""Geometria de visibilidade e integracao com o formato de quadros Revo."""
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pipeline/revo_frames'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pipeline'))
from precisao import reprojetar, decidir, ValidadorTemporal
from revo_projeto import RevoScan, limpar_scan, gravar_ply, SIG_REVO
from preparar_importacao import preparar


class VisibilidadeTest(unittest.TestCase):
    def comparar(self, profundidade, z=100):
        return reprojetar(np.array([[0., 0., -z]]), np.eye(4), (100,100,8,8), profundidade, 1)

    def test_superficie_confirmada(self):
        a,b,_=self.comparar(np.full((16,16),100.))
        self.assertTrue(a[0]); self.assertFalse(b[0])

    def test_espaco_livre(self):
        a,b,_=self.comparar(np.full((16,16),105.))
        self.assertFalse(a[0]); self.assertTrue(b[0])

    def test_oclusao_nao_rejeita(self):
        a,b,_=self.comparar(np.full((16,16),95.))
        self.assertFalse(a[0]); self.assertFalse(b[0])

    def test_falta_dados_nao_rejeita(self):
        a,b,_=self.comparar(np.full((16,16),np.nan))
        self.assertFalse(a[0]); self.assertFalse(b[0])

    def test_borda_com_lacuna_nao_rejeita(self):
        d=np.full((16,16),105.); d[7,7]=np.nan
        self.assertFalse(self.comparar(d)[1][0])

    def test_pose_e_convencao_camera(self):
        t=np.eye(4);t[:3,3]=[20,30,40]
        p=np.array([[20,30,-60.]])
        self.assertTrue(reprojetar(p,t,(100,100,8,8),np.full((16,16),100.),1)[0][0])

    def test_atras_camera_e_fora_imagem(self):
        a,b,_=reprojetar(np.array([[0,0,100.],[1000,0,-100.]]),np.eye(4),(100,100,8,8),np.full((16,16),105.),1)
        self.assertFalse(a.any());self.assertFalse(b.any())

    def test_apoio_prevalece_e_exige_dois_quadros(self):
        np.testing.assert_array_equal(decidir(np.ones(4,bool),np.array([0,0,1,0]),np.array([1,2,4,0])),[False,True,False,False])


def criar_scan(pasta, detalhe_persistente=False):
    (pasta/'cache').mkdir(parents=True)
    arquivo=pasta/'cache/frames.dataset'
    cn=sqlite3.connect(arquivo)
    cn.execute('create table FrameInfo (id integer primary key, bin blob)')
    cn.execute('create table FrameRGBD (id integer primary key, data blob)')
    for k in range(5):
        d=np.full((16,16),100.,np.float32)
        if k==2 or detalhe_persistente: d[8,8]=95
        blob=struct.pack('<III',16,16,256)+np.packbits(np.ones(256,bool),bitorder='little').tobytes()+b'\x00'*4+d.astype('<f4').tobytes()
        info=struct.pack('<IIIi',0,0,0,0)+np.eye(4,dtype='<f8').tobytes()
        cn.execute('insert into FrameInfo values (?,?)',(k+1,info))
        cn.execute('insert into FrameRGBD values (?,?)',(k+1,blob))
    cn.commit();cn.close()
    with arquivo.open('r+b') as f:f.write(SIG_REVO)
    header='ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    gravar_ply(pasta/'raw_preview.ply',header,np.array([[0,0,-100.]]))
    (pasta/'property.rvproj').write_text(json.dumps({'guid':'scan','merge_depends':[{'node':'scan','pose':[1,0]}]}))


class IntegracaoTest(unittest.TestCase):
    def test_detalhe_fino_confirmado_em_outros_quadros_preservado(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pasta=root/'scan';criar_scan(pasta,detalhe_persistente=True);(root/'work').mkdir()
            scan=RevoScan(str(pasta),str(root/'work'));scan.K={0:(100,100,8,8)}
            try:
                with patch('revo_projeto.mascara_manter',side_effect=lambda p,s:np.ones(len(p),bool)):
                    r=limpar_scan(scan,{},str(root/'limpo'),ilhas=False,precisao=True,
                                  pixel_adaptativo=True,temporal_passo=1,zerar_vazios=True)
                self.assertEqual(r['antes'],r['depois'])
                self.assertEqual(r['precisao']['com_apoio'],5)
            finally:scan.c.close()

    def test_pares_com_vies_alto_nao_autorizam_remocao(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pasta=root/'scan';criar_scan(pasta);(root/'work').mkdir()
            scan=RevoScan(str(pasta),str(root/'work'));scan.K={0:(100,100,8,8)}
            try:
                y,x=np.mgrid[:16,:16]
                for z in (97., 90.):
                    p=np.c_[((x.ravel()-8)*z/100),-((y.ravel()-8)*z/100),np.full(256,-z)]
                    v=ValidadorTemporal(scan,tol=1,passo=1)
                    rejeitar,_=v.validar(2,p,np.ones(256,bool))
                    self.assertFalse(rejeitar.any())
                    self.assertTrue(all(q['suspeita_desalinhamento'] for q in v.diagnosticos[0]['pares']))
            finally:scan.c.close()

    def test_limpa_apenas_pico_preserva_poses_e_profundidades(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pasta=root/'orig/data/scan';criar_scan(pasta)
            (root/'work').mkdir();(root/'work2').mkdir()
            scan=RevoScan(str(pasta),str(root/'work'));scan.K={0:(100,100,8,8)}
            try:
                with patch('revo_projeto.mascara_manter',side_effect=lambda p,s:np.ones(len(p),bool)):
                    r=limpar_scan(scan,{},str(root/'limpo/data/scan'),ilhas=False,precisao=True,
                                  pixel_adaptativo=True,temporal_passo=1,zerar_vazios=True)
                self.assertEqual(r['antes']-r['depois'],1)
                sl=RevoScan(str(root/'limpo/data/scan'),str(root/'work2'))
                try:
                    np.testing.assert_array_equal(scan.T,sl.T)
                    for i in scan.ids:
                        _,_,m,z,_=scan.profundidade(i);_,_,ml,zl,_=sl.profundidade(i)
                        self.assertFalse((ml & ~m).any())
                        np.testing.assert_array_equal(z[ml[m]],zl)
                    self.assertEqual(sl.c.execute('pragma integrity_check').fetchone()[0],'ok')
                finally:sl.c.close()
            finally:scan.c.close()

    def test_ids_referencias_e_nao_sobrescrita(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pasta=root/'orig/data/scan';criar_scan(pasta)
            (root/'orig/projeto.revo').write_text(json.dumps({'guid':'projeto','nodes':[{'guid':'scan','name':'scan','type':2}]}))
            antes=(pasta/'cache/frames.dataset').read_bytes()
            arq=preparar(root/'orig',root/'novo');r=json.loads(arq.read_text());ident=r['nodes'][0]['guid']
            self.assertNotEqual(ident,'scan')
            self.assertEqual(r['nodes'][0]['name'], 'scan_COPIA')
            p=root/'novo/data'/ident
            self.assertEqual((p/'cache/frames.dataset').read_bytes(),antes)
            self.assertEqual(json.loads((p/'property.rvproj').read_text())['merge_depends'][0]['node'],ident)
            with self.assertRaises(FileExistsError):preparar(root/'orig',root/'novo')


if __name__=='__main__':unittest.main()
