"""Cobertura de regioes cortadas: outra vista pode recuperar ou deixar lacuna."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline'))
from avaliar_corte_apoio import cobertura_removidos


class RecuperacaoTest(unittest.TestCase):
    def test_regiao_removida_so_e_recuperada_se_outro_scan_tem_dado_perto(self):
        p=np.array([[0.,0,0],[1,0,0],[2,0,0]])
        r=cobertura_removidos(p,np.array([True,True,False]),np.array([[0.,0,.05]]),[.1,.3])
        self.assertEqual(r['removidos'],2)
        self.assertEqual(r['cobertura']['0.1'],50)
        self.assertEqual(r['sem_apoio_na_maior_tolerancia'],1)

    def test_outra_vista_inteira_cortada_nao_confirma_recuperacao(self):
        r=cobertura_removidos(np.zeros((1,3)),np.array([True]),np.empty((0,3)),[.3])
        self.assertEqual(r['cobertura']['0.3'],0)


if __name__=='__main__':unittest.main()
