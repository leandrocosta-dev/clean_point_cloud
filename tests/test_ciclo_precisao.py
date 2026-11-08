"""Geometria, rastreabilidade, revisoes e bloqueio de merge do ciclo de precisao."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_precisao import criar_scan
from ciclo_precisao import (iniciar, carregar, corrigir, aprovar, mascara_regioes,
                            salvar_json, estado, validar_regioes, exportar_merge,
                            registrar_revo, comparar)
from fusao_superficies import reduzir_orientado, refinar_camadas, representantes, executar


def obs_sinteticas():
    dtype=np.dtype([('p','f4',3),('v','f4',3),('quadro','i4'),('pixel','i4')])
    obs=np.zeros(6,dtype=dtype)
    obs['p']=[[0,0,.01],[0,0,.01],[0,0,.06],[0,0,.01],[0,0,.06],[0,0,.06]]
    obs['v']=[[0,0,1],[0,0,1],[0,0,-1],[0,0,1],[0,0,-1],[0,0,-1]]
    obs['quadro']=[0,0,0,1,1,1]
    return obs,[dict(indice=0,inicio=0,fim=3),dict(indice=1,inicio=3,fim=6)]


class CamadasTest(unittest.TestCase):
    def test_faces_opostas_no_mesmo_voxel_nao_sao_promediadas(self):
        obs,frames=obs_sinteticas()
        p,v,c,nf,rep,_=reduzir_orientado(obs,frames,.1,'cpu',3)
        self.assertEqual(len(p),2)
        np.testing.assert_allclose(sorted(p[:,2]),[.01,.06],atol=1e-7)
        np.testing.assert_array_equal(nf,[2,2])
        np.testing.assert_array_equal(c,[3,3])
        self.assertTrue(np.all(rep<len(obs)))

    def test_cuda_preserva_camadas_e_conta_quadros(self):
        try:
            import cupy as cp
            if not cp.cuda.runtime.getDeviceCount():self.skipTest('CUDA ausente')
        except ImportError:self.skipTest('CuPy ausente')
        obs,frames=obs_sinteticas()
        a=reduzir_orientado(obs,frames,.1,'cpu',3)
        b=reduzir_orientado(obs,frames,.1,'cuda',100)
        for x,y in zip(a[:4],b[:4]):np.testing.assert_allclose(x,y,atol=1e-7)

    def test_reamostragem_preserva_face_oposta_e_coordenada(self):
        p=np.array([[0,0,.01],[0,0,.02],[0,0,.06]])
        n=np.array([[0,0,1],[0,0,1],[0,0,-1]])
        sel=representantes(p,n,np.array([1,5,3]),.1)
        self.assertEqual(set(sel),{1,2})

    def test_plano_sem_ruido_nao_e_deslocado(self):
        x,y=np.meshgrid(np.arange(-1,1,.1),np.arange(-1,1,.1))
        p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)].astype('f4')
        v=np.tile([0,0,1.],(len(p),1)).astype('f4')
        q,n,_,d=refinar_camadas(p,v,'cpu')
        np.testing.assert_allclose(q,p,atol=1e-7)
        self.assertLess(float(d.max()),1e-7)


class CicloTest(unittest.TestCase):
    def criar(self,root):
        src=root/'original';criar_scan(src/'data/scan')
        salvar_json(src/'projeto.revo',dict(guid='projeto',nodes=[dict(guid='scan',name='scan',type=2)]))
        m=iniciar(src,root/'ciclo')
        return src,m

    def test_snapshot_detecta_alteracao_e_nao_toca_original(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src,m=self.criar(root)
            self.assertTrue((src/'data/scan/cache/frames.dataset').exists())
            carregar(root/'ciclo')
            frozen=root/'ciclo/r000/projeto'/m['projeto']
            frozen.write_text('{}')
            with self.assertRaises(ValueError):carregar(root/'ciclo')

    def test_mascara_nao_muda_referencial_e_conflito_e_recusado(self):
        m=dict(id='m',scans=['s'])
        r=dict(scan='s',tipo='caixa',min=[-.1,-.1,-.1],max=[.1,.1,.1],classe='suporte',motivo='suporte identificado')
        validar_regioes(dict(manifesto='m',regioes=[r]),m)
        p=np.array([[0,0,0],[1,0,0]])
        np.testing.assert_array_equal(mascara_regioes(p,[r],'s'),[True,False])
        with self.assertRaises(ValueError):mascara_regioes(p,[r,dict(r,classe='proteger')],'s')
        with self.assertRaises(ValueError):validar_regioes(dict(manifesto='antigo',regioes=[r]),m)

    def test_correcao_auditada_cria_revisao_e_invalida_derivados(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src,m=self.criar(root)
            g=m['scans'][0];rev=root/'ciclo/r000'
            nativo=rev/'nativo_teste';nativo.mkdir()
            salvar_json(nativo/'fusao.json',dict(entrada=dict(calibracao={'0':dict(K=[100,100,8,8])})))
            st=estado(rev);st['nativo'][g]=nativo.name;salvar_json(rev/'estado.json',st)
            r=dict(scan=g,tipo='caixa',min=[-.1,-.1,-95.1],max=[.1,.1,-94.9],classe='residuo',motivo='pico sintetico confirmado')
            mask=root/'r.json';salvar_json(mask,dict(manifesto=m['id'],regioes=[r]))
            result=corrigir(root/'ciclo',mask)
            self.assertEqual(result[g]['removidos'],1)
            _,_,nova,novo=carregar(root/'ciclo')
            self.assertEqual(nova.name,'r001')
            self.assertNotEqual(novo['scans'][0],g)
            self.assertEqual(estado(nova)['estado'],'revisao_pendente')
            self.assertEqual(estado(nova)['revo'],{})
            self.assertTrue((src/'data/scan/cache/frames.dataset').exists())

    def test_aprovacao_sem_fusao_comparacao_e_inspecao_e_bloqueada(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,m=self.criar(root)
            p=root/'parecer.json';salvar_json(p,dict(manifesto=m['id'],scans={m['scans'][0]:dict(pendencias=[])}))
            with self.assertRaises(ValueError):aprovar(root/'ciclo',p)
            with self.assertRaises(ValueError):exportar_merge(root/'ciclo',root/'merge')

    def test_fusao_consulta_integral_e_visualizador(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);_,m=self.criar(root)
            g=m['scans'][0];rev=root/'ciclo/r000';out=rev/'nativo'
            executar(rev/'projeto/data'/g,out,'cpu',.2,min_quadros=2,
                     calibracao_fixa={'0':dict(K=[100,100,8,8],erro_preview_mm=0)})
            st=estado(rev);st['nativo'][g]=out.name;salvar_json(rev/'estado.json',st)
            from inspecao_fusao import consultar,gerar_html
            r=consultar(root/'ciclo',g,[-.1,-.1,-95.1,.1,.1,-94.9],root/'consulta')
            self.assertEqual(r['observacoes'],1)
            self.assertEqual(r['amostra'][0]['pixel'],136)
            self.assertEqual(r['amostra'][0]['quadro_id'],3)
            gerar_html(root/'ciclo',g,root/'vista.html')
            txt=(root/'vista.html').read_text(encoding='utf-8')
            self.assertIn('Plotly.react',txt)
            self.assertNotIn('__CONFIG__',txt)
            self.assertNotIn('<script src=',txt)
            # Referencia sintetica controlada para testar o contrato do ciclo,
            # sem fingir que um teste unitario executa o programa Revo Scan.
            import shutil
            ref=rev/'revo_fixture';(ref/g).mkdir(parents=True)
            shutil.copyfile(out/'nuvem.ply',ref/g/'Fused teste.ply')
            report=dict(revo=str((rev/'projeto'/m['projeto']).resolve()),parametros={},
                        scans=[dict(guid=g,fusao=dict(ponto_mm=.1,ms=1,otimizar_poses=0))])
            salvar_json(ref/'revo_pipeline.json',report)
            registrar_revo(root/'ciclo',ref/'revo_pipeline.json');comparar(root/'ciclo')
            consultar(root/'ciclo',g,[-.1,-.1,-95.1,.1,.1,-94.9],root/'consulta_final')
            parecer=root/'parecer.json'
            salvar_json(parecer,dict(manifesto=m['id'],scans={g:dict(pendencias=[],inspecao_3d_completa=True,
                suportes_excluidos=True,detalhes_preservados=True,incertos_revisados=True,
                evidencia='Fixture sintetica: plano, pico e todas as observacoes conferidos no teste.')}))
            aprovar(root/'ciclo',parecer)
            exportar_merge(root/'ciclo',root/'entrega')
            self.assertTrue((root/'entrega/LIBERACAO_MERGE.json').is_file())
            # Uma alteracao da referencia invalida a liberacao anterior.
            with (ref/g/'Fused teste.ply').open('ab') as f:f.write(b'\n')
            with self.assertRaises(ValueError):exportar_merge(root/'ciclo',root/'entrega2')


if __name__=='__main__':unittest.main()
