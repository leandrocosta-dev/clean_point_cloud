"""Operacoes na GPU dos scripts de formato antigo, Creality e utilitarios (geom_gpu.py e revo_antigo/) contra
referencias scipy/sklearn/Open3D em dados sinteticos."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

RAIZ = Path(__file__).resolve().parents[1]
import geom_gpu as gg
import gpu

from tests.gpu_utils import CUDA_DISPONIVEL


def _duas_placas(seed=0):
    rng = np.random.default_rng(seed)
    a = np.c_[rng.uniform(0, 30, (30000, 2)), 0.02 * rng.standard_normal(30000)]
    b = np.c_[rng.uniform(0, 10, (8000, 2)) + [40, 0], 0.02 * rng.standard_normal(8000)]
    r = rng.uniform(-10, 60, (200, 3))
    return np.vstack([a, b, r])


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class ComponentesTest(unittest.TestCase):
    def test_dbscan_igual_ao_sklearn(self):
        from sklearn.cluster import DBSCAN

        P = _duas_placas()
        ref = DBSCAN(eps=0.5, min_samples=10).fit(P).labels_
        rot = gg.dbscan(P, 0.5, 10)
        self.assertEqual(int((ref >= 0).sum()), int((rot >= 0).sum()))
        self.assertEqual(len(np.unique(rot[rot >= 0])), len(np.unique(ref[ref >= 0])))
        maior = ref == np.argmax(np.bincount(ref[ref >= 0]))
        self.assertTrue(np.array_equal(maior, gg.dbscan_maior(P, 0.5, 10)))
        self.assertTrue(np.array_equal(rot == 0, maior))

    def test_dbscan_sem_nucleos(self):
        P = np.random.default_rng(1).uniform(0, 100, (50, 3))
        self.assertFalse(gg.dbscan_maior(P, 0.1, 10).any())

    def test_componentes_triangulos_igual_ao_open3d(self):
        import open3d as o3d

        m = o3d.geometry.TriangleMesh.create_sphere(10, 40)
        m2 = o3d.geometry.TriangleMesh.create_sphere(3, 20)
        m2.translate([30, 0, 0])
        m3 = o3d.geometry.TriangleMesh.create_box(2, 2, 2)
        m3.translate([0, 40, 0])
        mm = m + m2 + m3
        F = np.asarray(mm.triangles)
        _, tam_ref, _ = mm.cluster_connected_triangles()
        rot, tam = gg.componentes_triangulos(F, len(mm.vertices))
        self.assertEqual(sorted(tam.tolist()), sorted(np.asarray(tam_ref).tolist()))
        self.assertEqual(tam[0], tam.max())
        self.assertEqual(int((rot == 0).sum()), int(tam[0]))

    def test_filtros_de_componentes_da_malha(self):
        import open3d as o3d

        mm = o3d.geometry.TriangleMesh.create_sphere(10, 40) + o3d.geometry.TriangleMesh.create_box(2, 2, 2)
        F = np.asarray(mm.triangles)
        ok = gg.remover_triangulos_fora_do_maior(F, len(mm.vertices), frac_min=0.5)
        self.assertEqual(int(ok.sum()), len(F) - 12)

    def test_rotulos_por_arestas(self):
        r = gpu.para_cpu(gg.rotulos_por_arestas(6, [0, 1, 3], [1, 2, 4]))
        self.assertEqual(r.tolist(), [0, 0, 0, 3, 3, 5])


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class NuvemTest(unittest.TestCase):
    def test_normais_hibrido_igual_ao_open3d(self):
        import open3d as o3d

        P = _duas_placas()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        N = gg.normais_hibrido(P, 1.0, 30)
        ok = np.isfinite(N[:, 0])
        cos = np.abs((np.asarray(pc.normals) * N).sum(1))[ok]
        self.assertGreater((cos > 0.99).mean(), 0.995)

    def test_normais_orientadas_e_nan_sem_vizinhos(self):
        P = np.vstack([_duas_placas()[:30000], [[900.0, 900.0, 900.0]]])
        N = gg.normais_hibrido(P, 1.0, 30, orientar_para=[0, 0, 100.0])
        self.assertTrue(np.isnan(N[-1]).all())
        self.assertGreater(np.nanmean(N[:30000, 2]), 0.99)

    def test_acumulador_voxel_igual_a_reducao_de_uma_vez(self):
        rng = np.random.default_rng(2)
        P = rng.uniform(-20, 20, (60000, 3))
        acc = gg.AcumuladorVoxel(0.8, pendente_max=5000)
        for i in range(0, len(P), 7000):
            acc.add(P[i : i + 7000], extra=P[i : i + 7000] * 2)
        C, cont, E = acc.resultado()
        C2, cont2 = gpu.reduzir_voxel(P, 0.8, origem=acc.origem)
        self.assertEqual(len(C), len(C2))
        np.testing.assert_allclose(C, C2, atol=1e-9)
        np.testing.assert_array_equal(cont, cont2)
        np.testing.assert_allclose(E, C * 2, atol=1e-9)

    def test_reduzir_voxel_normais_normaliza(self):
        P = _duas_placas()[:20000]
        N = np.tile([0.0, 0.0, 1.0], (len(P), 1))
        C, Nm = gg.reduzir_voxel_normais(P, N, 1.0)
        np.testing.assert_allclose(np.linalg.norm(Nm, axis=1), 1.0, atol=1e-9)
        self.assertLess(len(C), len(P))

    def test_mediana_por_grupo_igual_ao_numpy(self):
        rng = np.random.default_rng(3)
        k = rng.integers(0, 6, 501)
        v = rng.standard_normal((501, 3))
        u, med, cont = gg.mediana_por_grupo(k, v)
        for i, g in enumerate(u):
            np.testing.assert_allclose(med[i], np.median(v[k == g], 0), atol=1e-12)
            self.assertEqual(cont[i], (k == g).sum())

    def test_distancia_e_vizinho(self):
        rng = np.random.default_rng(4)
        A, Q = rng.uniform(0, 10, (2000, 3)), rng.uniform(0, 10, (500, 3))
        from scipy.spatial import cKDTree

        dr, jr = cKDTree(A).query(Q)
        np.testing.assert_allclose(gg.distancia_ate(A, Q), dr, atol=1e-4)
        d, j = gg.vizinho_ate(A, Q, 0.5)
        self.assertTrue(np.all((j >= 0) == (dr <= 0.5 + 1e-6)))
        self.assertTrue(np.isinf(gg.distancia_ate(np.zeros((0, 3)), Q)).all())

    def test_espalhamento_local_mede_a_rugosidade(self):
        rng = np.random.default_rng(5)
        P = np.c_[rng.uniform(0, 30, (60000, 2)), 0.03 * rng.standard_normal(60000)] + [100, 200, 300]
        W, C = gg.espalhamento_local(P, P[:200], 1.0)
        self.assertAlmostEqual(float(np.median(np.sqrt(W[:, 0]))), 0.03, delta=0.004)
        self.assertTrue((C > 15).all())

    def test_contagem_no_raio_igual_ao_remove_radius_outlier_do_open3d(self):
        import open3d as o3d

        P = np.random.default_rng(1).uniform(0, 10, (8000, 3))
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        _, ind = pc.remove_radius_outlier(8, 0.6)
        ref = np.zeros(len(P), bool)
        ref[ind] = True
        self.assertTrue(np.array_equal(ref, gpu.contar_no_raio(P, P, 0.6) > 8))

    def test_filtro_estatistico_igual_ao_open3d(self):
        import open3d as o3d

        P = np.random.default_rng(1).uniform(0, 10, (8000, 3))
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        _, ind = pc.remove_statistical_outlier(30, 3.0)
        ref = np.zeros(len(P), bool)
        ref[ind] = True
        self.assertEqual(int((ref != gpu.remover_soltos(P, k=29, desvios=3.0)).sum()), 0)

    def test_ransac_plano_open3d_acha_a_mesa(self):
        rng = np.random.default_rng(6)
        mesa = np.c_[rng.uniform(0, 100, (20000, 2)), 0.1 * rng.standard_normal(20000)]
        objeto = rng.uniform(0, 100, (3000, 3)) * [1, 1, 0.2] + [0, 0, 5]
        pl, inl = gg.ransac_plano_open3d(np.vstack([mesa, objeto]), limiar=0.5, iteracoes=500)
        self.assertAlmostEqual(abs(pl[2]), 1.0, places=2)
        self.assertGreater(len(inl) / 23000, 0.8)
        self.assertEqual(len(inl), len(np.unique(inl)))

    def test_icp_encadeado_recupera_a_transformacao(self):
        import open3d as o3d

        m = o3d.geometry.TriangleMesh.create_sphere(20, 60)
        V = np.asarray(m.vertices) * [1.0, 1.4, 0.7]
        A, NA = gg.amostrar_malha(V, np.asarray(m.triangles), 30000)
        ang = np.radians(4)
        T = np.eye(4)
        T[:3, :3] = [[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]
        T[:3, 3] = [0.6, -0.4, 0.3]
        B = gg.transformar(A, np.linalg.inv(T))
        Tr, info = gg.icp_encadeado(B, A, NA, np.eye(4), (3.0, 1.5, 0.6), iteracoes=60)
        np.testing.assert_allclose(Tr, T, atol=0.05)
        self.assertGreater(info["fitness"], 0.9)

    def test_amostrar_malha_fica_na_superficie_e_uniforme(self):
        import open3d as o3d

        m = o3d.geometry.TriangleMesh.create_sphere(10, 80)
        P, N = gg.amostrar_malha(np.asarray(m.vertices), np.asarray(m.triangles), 200000)
        r = np.linalg.norm(P, axis=1)
        self.assertTrue(r.min() > 9.9 and r.max() < 10.0001)
        self.assertLess(abs(P.mean(0)).max(), 0.05)  # uniforme: centroide no centro
        self.assertGreater((np.abs((P * N).sum(1)) / r).min(), 0.97)  # normal do triangulo ~ radial
        self.assertAlmostEqual(
            gg.area_malha(np.asarray(m.vertices), np.asarray(m.triangles)), m.get_surface_area(), delta=1e-6
        )

    def test_normais_de_vertice_e_triangulo_iguais_ao_open3d(self):
        import open3d as o3d

        m = o3d.geometry.TriangleMesh.create_sphere(10, 30)
        m.compute_vertex_normals()
        m.compute_triangle_normals()
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        cos = (gg.normais_vertices(V, F) * np.asarray(m.vertex_normals)).sum(1)
        self.assertGreater(float(cos.min()), 0.9999)
        cos = (gg.normais_triangulos(V, F) * np.asarray(m.triangle_normals)).sum(1)
        self.assertGreater(float(cos.min()), 0.9999)

    def test_campo_deriva_corrige_uma_deslocamento_suave(self):
        rng = np.random.default_rng(7)
        B = np.c_[rng.uniform(0, 120, (60000, 2)), np.zeros(60000)]
        nB = np.tile([0.0, 0.0, 1.0], (len(B), 1))
        V = B[::2].copy()
        V[:, 2] += 1.0  # A flutua 1 mm acima de B
        corr = gg.campo_deriva(V, B, nB, 20.0)(V)
        np.testing.assert_allclose(gpu.para_cpu(corr)[:, 2], -1.0, atol=0.05)
        np.testing.assert_allclose(gpu.para_cpu(corr)[:, :2], 0.0, atol=1e-6)

    def test_ler_nuvem_decima_e_estima_normais(self):
        import open3d as o3d

        P = _duas_placas()[:20000]
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "n.ply"
            o3d.io.write_point_cloud(str(arq), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)))
            C, N = gg.ler_nuvem(arq, voxel=0.5)
            self.assertEqual(C.shape, N.shape)
            self.assertLess(len(C), len(P))
            np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-6)


def _profundidade_sintetica():
    """Imagem 120x160: plano com buraco, bolha pequena isolada (separada por salto) e pixels voadores."""
    d = np.zeros((120, 160), np.uint16)
    d[20:100, 20:130] = 5000  # 500 mm (escala 0,1)
    d[50:55, 60:65] = 0  # buraco
    d[5:12, 140:150] = 4000  # bolha a 400 mm: salto de 100 mm
    d[110, 5] = 3000  # pixel voador
    return d


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class DphTest(unittest.TestCase):
    def test_mascara_peca_igual_a_versao_scipy(self):
        from pipeline.revo_antigo.limpar_dph import mascara_peca

        def referencia(d, salto=2.0, dilat=4, escala=0.1):
            m = d > 0
            z = d.astype(np.float32) * escala
            g = np.zeros_like(m)
            for dy, dx in ((0, 1), (1, 0)):
                a = np.abs(z - np.roll(z, (dy, dx), (0, 1))) > salto
                g |= a | np.roll(a, (-dy, -dx), (0, 1))
            lab, n = ndi.label(ndi.binary_dilation(m & ~g, iterations=dilat) & m)
            sz = ndi.sum(m, lab, range(1, n + 1))
            return lab == 1 + int(np.argmax(sz))

        d = _profundidade_sintetica()
        got = mascara_peca(d)
        self.assertEqual(got.dtype, bool)
        self.assertTrue(np.array_equal(got, referencia(d)))
        self.assertFalse(got[5:12, 140:150].any())  # bolha fora
        self.assertFalse(mascara_peca(np.zeros((10, 10), np.uint16)).any())

    def test_pontos_gpu_igual_a_pontos(self):
        from pipeline.revo_antigo.dph import pontos, pontos_gpu

        d = _profundidade_sintetica()
        K = (800.0, 80.0, 60.0, 160.0, 0.1)
        for passo in (1, 3):
            p, v, u = pontos(d, K, passo=passo)
            pg, vg, ug = pontos_gpu(d, K, passo=passo)
            np.testing.assert_allclose(gpu.para_cpu(pg), p, rtol=1e-5, atol=1e-3)
            np.testing.assert_array_equal(gpu.para_cpu(vg), v)
            np.testing.assert_array_equal(gpu.para_cpu(ug), u)

    def test_para_mundo(self):
        from pipeline.revo_antigo.dph import para_mundo

        p = np.random.default_rng(0).uniform(-5, 5, (100, 3))
        T = np.eye(4)
        T[:3, :3] = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        T[:3, 3] = [1, 2, 3]
        np.testing.assert_allclose(para_mundo(p, T, saida_gpu=False), p @ T[:3, :3].T + T[:3, 3], atol=1e-12)

    def test_pose_do_quadro_escolhe_a_que_concorda_com_o_fuse(self):
        """Quadro sintetico: a pose da .inf coloca os pontos sobre o fuse; a registrada fica 5 mm fora."""
        from pipeline.revo_antigo.dph import pose_do_quadro

        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            (pasta / "cache").mkdir()
            d = np.zeros((300, 400), np.uint16)
            d[40:260, 40:360] = 4000
            f = pasta / "cache" / "frame_000_0000.dph"
            d.tofile(f)
            Ti = np.eye(4)
            inf = bytes(16) + Ti[:3, :].astype("<f8").tobytes() + bytes(64)
            (pasta / "cache" / "frame_000_0000.inf").write_bytes(inf)
            K = (500.0, 200.0, 150.0, 400.0, 0.1)
            from pipeline.revo_antigo.dph import pontos

            p, _, _ = pontos(d, K, passo=4)
            fuse = p.astype(np.float64)  # o fuse e exatamente a nuvem pela pose da .inf
            Tr = np.eye(4)
            Tr[:3, 3] = [0, 0, 5.0]
            T, qual = pose_do_quadro(str(f), {0: Tr}, K, fuse)
            self.assertEqual(qual, "inf")
            np.testing.assert_allclose(T, Ti)
            T, qual = pose_do_quadro(str(f), {0: Ti.copy()}, K, fuse + [0, 0, 7.0])
            self.assertEqual(qual, "reg")


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class UtilitariosTest(unittest.TestCase):
    def _cena(self):
        rng = np.random.default_rng(8)
        mesa = np.c_[rng.uniform(0, 80, (40000, 2)), 0.05 * rng.standard_normal(40000)]
        u = rng.standard_normal((8000, 3))
        peca = 10 * u / np.linalg.norm(u, axis=1, keepdims=True) + [40, 40, 12]  # esfera uniforme sobre a mesa
        longe = rng.uniform(-30, 110, (60, 3)) + [0, 0, 25]
        return np.vstack([mesa, peca, longe]), 8000

    def test_part_indices_acha_a_peca(self):
        import build_revo_project as b

        P, npeca = self._cena()
        vert = np.c_[P, np.zeros_like(P)].astype(np.float32)
        idx, nn = b.part_indices(vert, 0.5, 2, 4.0)
        self.assertGreater(nn, 0)
        self.assertGreater((idx >= 40000).mean(), 0.99)
        self.assertGreater(len(idx), 0.95 * npeca)

    def test_clean_revo_acha_a_peca(self):
        import clean_revo
        import open3d as o3d

        P, npeca = self._cena()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        out = clean_revo.clean(pc, 0.5, 2, 4.0, 20, 2.0)
        Q = np.asarray(out.points)
        self.assertGreater(len(Q), 0.95 * npeca)
        self.assertGreater(Q[:, 2].min(), 1.0)  # mesa fora

    def test_clean_mesh_mantem_so_a_maior_componente(self):
        import build_revo_project as b
        import open3d as o3d

        m = o3d.geometry.TriangleMesh.create_sphere(10, 30) + o3d.geometry.TriangleMesh.create_sphere(2, 10).translate(
            [40, 0, 0]
        )
        V = np.asarray(m.vertices, np.float32)
        F = np.asarray(m.triangles, np.uint32)
        nv, nf = b.clean_mesh(np.c_[V, V], F, V.astype(np.float64), 1.0)
        self.assertEqual(len(nv), len(np.unique(nf)))
        self.assertLess(len(nf), len(F))
        self.assertLess(np.abs(nv[:, 0]).max(), 11)  # a esfera pequena (x ~ 40) saiu
        self.assertEqual(nf.dtype, np.uint32)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class UnirDoisTest(unittest.TestCase):
    def test_unir_dois_acha_a_transformacao_de_uma_peca_assimetrica(self):
        import open3d as o3d

        rng = np.random.default_rng(9)

        def caixa(x, y, z, o):
            m = o3d.geometry.TriangleMesh.create_box(x, y, z)
            m.translate(o)
            return m

        peca = caixa(60, 20, 10, [0, 0, 0]) + caixa(15, 12, 25, [5, 0, 0]) + caixa(10, 25, 6, [40, 0, 0])
        P, N = gg.amostrar_malha(np.asarray(peca.vertices), np.asarray(peca.triangles), 40000)
        P = P + 0.01 * rng.standard_normal(P.shape)
        ang = np.radians(35)
        T = np.eye(4)
        T[:3, :3] = [[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]]
        T[:3, 3] = [3.0, -2.0, 1.5]
        B = gg.transformar(P, np.linalg.inv(T))  # B = A visto de outra pose; T leva B -> A
        with tempfile.TemporaryDirectory() as tmp:
            pa, pb = Path(tmp) / "a.ply", Path(tmp) / "b.ply"
            for arq, X in ((pa, P), (pb, B)):
                pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
                pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
                o3d.io.write_point_cloud(str(arq), pc)
            saida = Path(tmp) / "T.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(RAIZ / "pipeline" / "revo_antigo" / "unir_dois.py"),
                    str(pa),
                    str(pb),
                    "--saida",
                    str(saida),
                    "--passo-ang",
                    "15",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            res = json.loads(saida.read_text())
            self.assertGreater(res[0]["fb"], 0.8)
            Tm = np.array(res[0]["T"])
            erro = np.linalg.norm((Tm @ np.linalg.inv(T) - np.eye(4))[:3, 3])
            self.assertLess(erro, 1.5)


if __name__ == "__main__":
    unittest.main()
