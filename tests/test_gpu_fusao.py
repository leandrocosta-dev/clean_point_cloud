"""Fusao e limpeza de nuvem na GPU (grupo 3): cada rotina contra uma referencia numpy/scipy escrita aqui (CPU so nos testes)."""

import tempfile
import unittest
from pathlib import Path

import gpu
import numpy as np
from scipy.spatial import cKDTree

from tests.gpu_utils import CUDA_DISPONIVEL


def superficie(n=20000, lado=6.0, ruido=0.01, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, lado, (n, 3))
    P[:, 2] = 0.4 * np.sin(P[:, 0]) * np.cos(P[:, 1]) + ruido * rng.standard_normal(n)
    return P


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class Pca3Test(unittest.TestCase):
    def test_menor_autopar_igual_eigh(self):
        import pca3_gpu

        rng = np.random.default_rng(1)
        A = rng.normal(size=(5000, 3, 3))
        A = A @ A.transpose(0, 2, 1)  # simetricas positivas
        D = np.diag([1.0, 0.8, 1e-6])  # "planos": menor autovalor muito menor
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        B = np.tile(R @ D @ R.T, (500, 1, 1))
        M = np.concatenate([A, B])
        w, v = np.linalg.eigh(M)
        lam, n = pca3_gpu.menor_autopar(M)
        lam, n = gpu.para_cpu(lam), gpu.para_cpu(n)
        np.testing.assert_allclose(lam, w[:, 0], atol=1e-9)
        self.assertLess(float((1 - np.abs((v[:, :, 0] * n).sum(1))).max()), 1e-9)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-12)

    def test_isotropica_nao_gera_nan(self):
        import pca3_gpu

        lam, n = pca3_gpu.menor_autopar(np.zeros((4, 3, 3)))
        self.assertTrue(np.isfinite(gpu.para_cpu(n)).all())


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class FundirNitidoGPUTest(unittest.TestCase):
    def test_grupos_identico_ao_np_unique(self):
        import fundir_nitido as fn

        X = np.random.default_rng(3).uniform(-5, 5, (50000, 3))
        inv, n = fn.grupos(X, 0.2)
        u, inv_ref, n_ref = np.unique(
            np.floor(X / 0.2).astype(np.int64), axis=0, return_inverse=True, return_counts=True
        )
        np.testing.assert_array_equal(gpu.para_cpu(inv), inv_ref.ravel())
        np.testing.assert_array_equal(gpu.para_cpu(n), n_ref)

    def test_centroides_com_pesos_e_minimo(self):
        import fundir_nitido as fn

        rng = np.random.default_rng(4)
        X = rng.uniform(-2, 2, (30000, 3))
        w = rng.uniform(0.1, 1, len(X))
        C, n, _ = fn.centroides(X, 0.25, w=w, min_obs=5)
        k = np.floor(X / 0.25).astype(np.int64)
        u, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        sw = np.bincount(inv, w)
        ref = np.stack([np.bincount(inv, X[:, i] * w) / sw for i in range(3)], 1)[cnt >= 5]
        np.testing.assert_allclose(gpu.para_cpu(C), ref, atol=1e-12)
        np.testing.assert_array_equal(gpu.para_cpu(n), cnt[cnt >= 5])

    def test_normais_e_espessura_contra_referencia(self):
        import fundir_nitido as fn

        P = superficie(15000, ruido=0.02)
        ref_dir = np.tile([0, 0, 1.0], (len(P), 1))
        N = gpu.para_cpu(fn.normais_locais(P, ref_dir, raio=0.5, sigma=0.25))
        arv = cKDTree(P)
        for i in range(0, 400, 20):
            e = P[arv.query_ball_point(P[i], 0.5)] - P[i]
            w = np.exp(-(e * e).sum(1) / (2 * 0.25**2))
            mu = (w[:, None] * e).sum(0) / w.sum()
            y = e - mu
            Mi = np.einsum("k,ki,kj->ij", w, y, y)
            nr = np.linalg.eigh(Mi)[1][:, 0]
            nr *= np.sign(nr @ ref_dir[0] + 1e-12)
            self.assertGreater(float(N[i] @ nr), 0.9999)
        e, nv = fn.espessura_local(P, 0.3)
        e, nv = gpu.para_cpu(e), gpu.para_cpu(nv)
        for i in range(0, 300, 30):
            q = P[arv.query_ball_point(P[i], 0.3)] - P[i]
            y = q - q.mean(0)
            ref = np.sqrt(max(np.linalg.eigvalsh(y.T @ y)[0], 0) / max(len(q), 1))
            self.assertAlmostEqual(float(e[i]), float(ref), delta=2e-5)
            self.assertEqual(int(nv[i]), len(q))

    def test_sem_soltos_e_mean_shift_igual_referencia(self):
        import fundir_nitido as fn
        import vizinhos_gpu as V

        F = superficie(60000, lado=8.0, ruido=0.01)
        F[:, 2] = 0.01 * np.random.default_rng(5).standard_normal(len(F))
        Q = F[:800] + [0, 0, 0.04]
        NQ = np.tile([0, 0, 1.0], (len(Q), 1))
        W = np.random.default_rng(1).uniform(0.2, 1, len(F))
        VF = np.tile([0, 0, 1.0], (len(F), 1))
        VF[::7] = [0, 0, -1.0]
        S, D = fn.mean_shift(
            gpu.para_gpu(Q), gpu.para_gpu(NQ), gpu.para_gpu(F), gpu.para_gpu(W), gpu.para_gpu(VF), 0.12, 0.08
        )
        R = V.mean_shift_cpu(F, Q, NQ, W, VF, 0.35, 0.12, 0.08)
        self.assertLess(float(np.abs(gpu.para_cpu(D) - R).max()), 2e-6)
        np.testing.assert_allclose(gpu.para_cpu(S), Q + R[:, None] * NQ, atol=2e-6)
        fn._GRADES.clear()
        P = np.vstack([F[:3000], [[50, 50, 50], [60, 60, 60]]])
        C, ref, soltos = fn.sem_soltos(gpu.para_gpu(P), gpu.para_gpu(P), 3, 0.3)
        n_ref = np.array(cKDTree(P).query_ball_point(P, 0.3, return_length=True))
        self.assertEqual(soltos, int((n_ref <= 3).sum()))

    def test_icp_multivista_recupera_pose(self):
        import fundir_nitido as fn

        rng = np.random.default_rng(6)
        xy = rng.uniform(0, 30, (150000, 2))
        z = 4 * np.sin(xy[:, 0] / 4) * np.cos(xy[:, 1] / 5) + 1.5 * np.sin(xy[:, 0] / 1.7 + xy[:, 1] / 2.3)
        S = np.c_[xy, z]
        A = S[(S[:, 0] < 20)]
        B = S[(S[:, 0] > 8)]
        a = np.radians(0.15)
        E = np.eye(4)
        E[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        E[:3, 3] = [0.12, -0.08, 0.05]
        # B medido = inv(E) aplicado a B; a pose que o leva de volta a A e E.
        Bm = (B - E[:3, 3]) @ E[:3, :3]  # inv(E) aplicado a B => medido
        T = fn.icp_multivista({"a": A, "b": Bm}, {"a": np.eye(4), "b": np.eye(4)})
        D = T["b"] @ np.linalg.inv(E)  # deve ser ~ identidade
        self.assertLess(float(np.linalg.norm(D[:3, 3])), 0.01)
        self.assertLess(float(np.abs(D[:3, :3] - np.eye(3)).max()), 2e-4)

    def test_pontos_gpu_igual_revoscan_e_cache(self):
        from cache_observacoes import calibracao_scan, pontos_gpu, pontos_scan
        from revo_projeto import RevoScan

        from tests.test_precisao import criar_scan

        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "scan"
            criar_scan(scan)
            K = (100.0, 100.0, 8.0, 8.0)
            work = Path(tmp) / "w"
            work.mkdir()
            sc = RevoScan(str(scan), str(work))
            Pg, pix, w, h = pontos_gpu(sc, 0, K)
            Pc, pixc = sc.pontos(0, K)
            np.testing.assert_allclose(gpu.para_cpu(Pg), Pc, atol=1e-9)
            np.testing.assert_array_equal(pix, pixc)
            chamadas = []

            def gerar():
                chamadas.append(1)
                return Pg, np.array([sc.T[0][:3, 3]], np.float32), np.array([len(Pg)])

            sc.c.close()
            r1 = pontos_scan(scan, K, gerar, Path(tmp) / "cache")
            r2 = pontos_scan(scan, K, gerar, Path(tmp) / "cache")
            self.assertEqual((r1[4], r2[4]), (False, True))
            self.assertEqual(len(chamadas), 1)
            np.testing.assert_array_equal(gpu.para_cpu(r1[0]), gpu.para_cpu(r2[0]))
            pontos_scan(scan, (101.0, 100.0, 8.0, 8.0), gerar, Path(tmp) / "cache")  # K novo: nao reaproveita
            self.assertEqual(len(chamadas), 2)
            n = []

            def calc():
                n.append(1)
                return (100.0, 100.0, 8.0, 8.0), 0.1

            c1 = calibracao_scan(scan, calc, 0, Path(tmp) / "cache")
            c2 = calibracao_scan(scan, calc, 0, Path(tmp) / "cache")
            self.assertEqual((c1[2], c2[2], len(n)), (False, True, 1))
            self.assertEqual(c1[0], c2[0])


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class FusaoSuperficiesGPUTest(unittest.TestCase):
    def _obs(self, n=60000, seed=7):
        rng = np.random.default_rng(seed)
        dtype = np.dtype([("p", "<f4", 3), ("v", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")])
        obs = np.zeros(n, dtype)
        p = rng.uniform(-3, 3, (n, 3))
        p[:, 2] = 0.05 * (rng.integers(0, 2, n)) + 0.003 * rng.standard_normal(n)
        obs["p"] = p
        v = np.tile([0, 0, 1.0], (n, 1))
        v[obs["p"][:, 2] > 0.025] = [0, 0, -1.0]
        obs["v"] = v
        obs["quadro"] = np.sort(rng.integers(0, 40, n))
        frames, ini = [], 0
        for q in np.unique(obs["quadro"]):
            c = int((obs["quadro"] == q).sum())
            frames.append(dict(indice=int(q), inicio=ini, fim=ini + c))
            ini += c
        return obs, frames

    def test_reduzir_orientado_contra_referencia_e_independente_do_bloco(self):
        from fusao_superficies import reduzir_orientado

        obs, frames = self._obs()
        a = reduzir_orientado(obs, frames, 0.1, bloco=10**9)
        b = reduzir_orientado(obs, frames, 0.1, bloco=5000)
        for x, y in zip(a[:5], b[:5]):
            np.testing.assert_allclose(x, y, atol=2e-6)
        # referencia numpy: camadas por octante de visada
        p, v = obs["p"].astype(np.float64), obs["v"].astype(np.float64)
        lo = p.min(0) - 0.1
        k = np.floor((p - lo) / 0.1).astype(np.int64)
        lado = (v[:, 0] >= 0).astype(np.int64) + 2 * (v[:, 1] >= 0) + 4 * (v[:, 2] >= 0)
        u, primeiro, inv, cnt = np.unique(
            np.c_[k, lado], axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        inv = inv.ravel()
        pos = np.stack([np.bincount(inv, p[:, i]) / cnt for i in range(3)], 1)
        votos = np.array([len(set(obs["quadro"][inv == g])) for g in range(len(u))])
        np.testing.assert_array_equal(a[2], cnt)
        np.testing.assert_allclose(a[0], pos, atol=1e-5)
        np.testing.assert_array_equal(a[3], votos)
        np.testing.assert_array_equal(a[4], primeiro)

    def test_representantes_igual_versao_numpy(self):
        from fusao_superficies import representantes

        rng = np.random.default_rng(8)
        p = rng.uniform(0, 3, (20000, 3))
        n = rng.normal(size=(20000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        cont = rng.integers(1, 6, 20000)
        axis = np.argmax(np.abs(n), axis=1)
        side = axis * 2 + (n[np.arange(len(n)), axis] < 0)
        _, inv = np.unique(np.c_[np.floor(p / 0.08).astype(np.int64), side], axis=0, return_inverse=True)
        inv = inv.ravel()
        ordem = np.lexsort((-cont.astype(np.int64), inv))
        _, first = np.unique(inv[ordem], return_index=True)
        np.testing.assert_array_equal(representantes(p, n, cont, 0.08), ordem[first])

    def test_refinar_camadas_contra_referencia_cpu(self):
        from fusao_superficies import refinar_camadas

        p = superficie(12000, lado=4.0, ruido=0.02).astype(np.float32)
        vis = np.tile([0, 0, 1.0], (len(p), 1)).astype(np.float32)
        q, n, amb, desl = refinar_camadas(p, vis, raio=0.24, sigma=0.08)
        r = _refinar_camadas_ref(p, vis, raio=0.24, sigma=0.08)
        self.assertLess(float(np.abs(q - r[0]).max()), 5e-4)
        self.assertGreater(float(np.mean(np.abs((n * r[1]).sum(1)) > 0.999)), 0.995)
        self.assertGreater(float(np.mean(amb == r[2])), 0.995)
        self.assertGreater(float(np.median(desl)), 0)  # mexeu (ruido de 20 um) ...
        self.assertLess(float(desl.max()), 0.08 + 1e-6)  # ... mas dentro de max_desloc


def _refinar_camadas_ref(p, vis, raio=0.3, sigma=0.08, max_desloc=0.08, passadas=2):
    """Copia do caminho antigo (cKDTree + numpy), usada como referencia."""
    lado = np.argmax(np.abs(vis), axis=1) * 2 + (vis[np.arange(len(vis)), np.argmax(np.abs(vis), axis=1)] < 0)
    keys = np.c_[np.floor(p / max(0.5, 6 * sigma)).astype(np.int64), lado]
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    centro = np.stack([np.bincount(inv, weights=p[:, j]) / cnt for j in range(3)], axis=1)
    vc = np.stack([np.bincount(inv, weights=vis[:, j]) for j in range(3)], axis=1)
    vc /= np.maximum(np.linalg.norm(vc, axis=1, keepdims=True), 1e-12)
    arv = cKDTree(centro)
    nc = np.empty_like(centro)
    ac = np.zeros(len(centro), bool)
    rn = max(1.2, 4 * raio)
    d, idx = arv.query(centro, k=40, distance_upper_bound=rn)
    valid = np.isfinite(d) & (idx < len(centro))
    idx = np.where(valid, idx, 0)
    valid &= np.einsum("nkj,nj->nk", vc[idx], vc) > 0.15
    w = np.where(valid, np.exp(-(np.minimum(d, rn) ** 2) / (2 * (rn / 2) ** 2)), 0)
    sw = np.maximum(w.sum(axis=1), 1e-12)
    delta = centro[idx] - centro[:, None, :]
    local = (delta * w[..., None]).sum(axis=1) / sw[:, None]
    y = delta - local[:, None, :]
    cov = np.einsum("nk,nki,nkj->nij", w, y, y) / sw[:, None, None]
    ev, en = np.linalg.eigh(cov)
    nn = en[:, :, 0]
    nn *= np.where(np.einsum("ij,ij->i", nn, vc) < 0, -1, 1)[:, None]
    nc[:] = nn
    ac[:] = (valid.sum(axis=1) < 6) | (ev[:, 0] / np.maximum(ev[:, 1], 1e-12) > 0.35)
    normals = nc[inv].astype(np.float32)
    ambiguo = ac[inv]
    out = p.astype(np.float64).copy()
    for _ in range(passadas):
        arv = cKDTree(out)
        destino = out.copy()
        d, idx = arv.query(out, k=64, distance_upper_bound=raio)
        valid = np.isfinite(d) & (idx < len(p))
        idx = np.where(valid, idx, 0)
        delta = out[idx] - out[:, None, :]
        h = np.einsum("nkj,nj->nk", delta, normals)
        cos = np.einsum("nkj,nj->nk", normals[idx], normals)
        valid = valid & (cos > 0.9) & (np.abs(h) < 2 * sigma)
        dd = np.minimum(d, raio)
        tang = np.maximum(dd * dd - h * h, 0)
        w = np.where(valid, np.exp(-tang / (2 * (raio / 2) ** 2) - h * h / (2 * sigma * sigma)), 0)
        step = np.clip(
            (w * h).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-12), -max_desloc / passadas, max_desloc / passadas
        )
        step[ambiguo] = 0
        destino[:] = out + step[:, None] * normals
        out = destino
    return out, normals, ambiguo


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class NuvemGPUTest(unittest.TestCase):
    def test_refinar_nuvem_reagrupar_e_passada(self):
        import refinar_nuvem as rn

        rng = np.random.default_rng(9)
        P = rng.uniform(0, 5, (30000, 3)).astype(np.float32)
        P[:, 2] = (0.3 * np.sin(P[:, 0]) + 0.01 * rng.standard_normal(len(P))).astype(np.float32)
        c = rng.integers(1, 9, len(P)).astype(np.uint16)
        V = np.tile([0, 0, 1.0], (len(P), 1)).astype(np.float32)
        Q, W, Vv = (gpu.para_cpu(x) for x in rn.reagrupar(P, c, V, 0.5))
        mn = P.min(0).astype(np.float64) - 1
        k = np.floor((P - mn) / 0.5).astype(np.int64)
        u, inv = np.unique(np.c_[k], axis=0, return_inverse=True)
        inv = inv.ravel()
        w = c.astype(np.float64)
        sw = np.bincount(inv, weights=w)
        ref = np.stack([np.bincount(inv, weights=w * P[:, a]) for a in range(3)], 1) / sw[:, None]
        np.testing.assert_allclose(Q, ref, atol=1e-5)
        np.testing.assert_allclose(W, sw, rtol=1e-6)
        # passada contra cKDTree
        P1, V1 = gpu.para_gpu(Q), gpu.para_gpu(Vv)
        saida, normal, ok, desl = (gpu.para_cpu(x) for x in rn.suavizar_passada(P1, V1, 1.8, 28, 0.5, 6))
        arv = cKDTree(Q.astype(np.float64))
        d, idx = arv.query(Q.astype(np.float64)[:200], k=28, distance_upper_bound=1.8)
        for i in range(0, min(200, len(Q)), 25):
            val = np.isfinite(d[i])
            Y = Q[idx[i][val]].astype(np.float64)
            wv = np.exp(-(d[i][val] ** 2) / (2 * 0.9**2))
            cen = (wv[:, None] * Y).sum(0) / wv.sum()
            Dm = Y - cen
            cov = np.einsum("k,ki,kj->ij", wv, Dm, Dm) / wv.sum()
            nr = np.linalg.eigh(cov)[1][:, 0]
            if nr @ Vv[i] < 0:
                nr = -nr
            dist = np.clip((Q[i] - cen) @ nr, -0.5, 0.5)
            np.testing.assert_allclose(saida[i], Q[i] - dist * nr, atol=2e-4)
            self.assertEqual(bool(ok[i]), bool(val.sum() >= 6))

    def test_suavizar_nuvem_reduz_ruido_e_iguala_referencia(self):
        import suavizar_nuvem as sn

        P = superficie(20000, lado=5.0, ruido=0.03)
        N = np.tile([0, 0, 1.0], (len(P), 1))
        Pn = sn.suavizar(P, N, raio=0.8, passadas=1, cos_min=0.8, k=32)
        d, i = cKDTree(P).query(P, k=32, distance_upper_bound=0.8)
        ok = np.isfinite(d)
        i2 = np.where(ok, i, 0)
        w = np.exp(-(d**2) / (2 * 0.4**2)) * ok
        h = np.einsum("nkc,nc->nk", P[i2] - P[:, None, :], N)
        dh = (w * h).sum(1) / np.maximum(w.sum(1), 1e-9)
        dh[ok.sum(1) < 5] = 0
        np.testing.assert_allclose(Pn, P + N * dh[:, None], atol=2e-4)
        ref_z = 0.4 * np.sin(P[:, 0]) * np.cos(P[:, 1])
        self.assertLess(float(np.std(Pn[:, 2] - ref_z)), float(np.std(P[:, 2] - ref_z)))

    def test_limpar_artefatos_remove_voadores_e_pequenos_e_poupa_a_borda(self):
        import limpar_artefatos as la

        rng = np.random.default_rng(10)
        x, y = np.meshgrid(np.arange(0, 12, 0.07), np.arange(0, 12, 0.07))
        sup = np.c_[x.ravel(), y.ravel(), 0.003 * rng.standard_normal(x.size)]
        voando = np.array([[3, 3, 0.4], [6, 6, 0.3], [8, 2, 0.5]])
        ilha = rng.normal(size=(100, 3)) * 0.02 + [40, 40, 40]
        X = np.vstack([sup, voando, ilha])
        manter, mot = la.limpar(X, raio=0.2, viz_nucleo=15, dist_nucleo=0.15, comp_min=500, fora_plano=0.03)
        self.assertFalse(manter[len(sup) : len(sup) + 3].any())  # voando
        self.assertFalse(manter[len(sup) + 3 :].any())  # ilha pequena desconectada
        self.assertGreater(float(manter[: len(sup)].mean()), 0.99)  # superficie (inclusive bordas) preservada
        # referencia scipy do mesmo criterio de contagem
        n_ref = np.array([len(v) for v in cKDTree(X).query_ball_point(X, 0.2)]) - 1
        self.assertEqual(mot["nucleo"], int((n_ref >= 15).sum()))
        rot, sz = la.componentes_por_raio(np.vstack([sup[:500], ilha]), 0.2)
        self.assertGreaterEqual(len(sz), 2)

    def test_limpar_fragmentos_componentes_e_consulta(self):
        import limpar_fragmentos as lf

        rng = np.random.default_rng(11)
        grande = rng.uniform(0, 30, (20000, 3))
        pequeno = rng.uniform(0, 3, (200, 3)) + [200, 200, 200]
        P = np.vstack([grande, pequeno])
        mn, lab, tam, rot, forma = lf.componentes_grade(P, 3.0, 9.0)
        tam = gpu.para_cpu(tam)
        self.assertEqual(int((tam > 0).sum()), 2)
        self.assertEqual(sorted(tam[tam > 0].tolist()), [200, 20000])
        r = gpu.para_cpu(lf.ids_de(P[[0, -1]], mn, lab, 3.0, forma))
        self.assertNotEqual(r[0], r[1])
        self.assertEqual(int(gpu.para_cpu(lf.ids_de(np.array([[1e4, 0, 0]]), mn, lab, 3.0, forma))[0]), 0)

    def test_orientar_normais_vira_para_fora_so_o_visivel(self):
        import orientar_normais as on

        from tests.test_fusao_regressoes import orientar_cpu_original

        rng = np.random.default_rng(12)
        v = rng.normal(size=(40000, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        P = 10 * v
        N = v.copy()
        topo = P[:, 2] > 8
        N[topo] *= -1  # normais do topo apontando para dentro
        Nn, visto, flip = on.orientar(P, N, celula=1.0, tol=0.8, cone=65.0)
        Nc, visto_cpu, flip_cpu = orientar_cpu_original(P, N, celula=1.0, tol=0.8, cone=65.0)
        # O cone e definido pelo eixo menor PCA, nao por +z. Na esfera, esse eixo e arbitrario:
        # parte do topo pode ficar oculta. O contrato preserva esses pontos, em vez de vira-los.
        np.testing.assert_array_equal(visto, visto_cpu)
        np.testing.assert_array_equal(flip, flip_cpu)
        np.testing.assert_allclose(Nn, Nc, atol=1e-6)
        topo_confiavel = topo & visto_cpu
        self.assertGreater(int(topo_confiavel.sum()), 0)
        self.assertGreater(float(((Nn * v).sum(1)[topo_confiavel] > 0).mean()), 0.99)
        self.assertGreater(float(((Nn * v).sum(1)[~topo] > 0).mean()), 0.99)
        self.assertTrue((topo & ~visto_cpu).any())
        self.assertFalse(flip[~visto_cpu].any())
        np.testing.assert_allclose(Nn[~visto_cpu], N[~visto_cpu], atol=1e-6)

    def test_uniao_icp_leve_e_fundir_proprio_agrega(self):
        import fundir_proprio as fp
        import uniao_revo_propria as ur

        rng = np.random.default_rng(13)
        xy = rng.uniform(0, 40, (200000, 2))
        z = 4 * np.sin(xy[:, 0] / 5) * np.cos(xy[:, 1] / 6) + 1.2 * np.sin(xy[:, 0] / 2 + xy[:, 1] / 3)
        R = np.c_[xy, z]
        a = np.radians(0.2)
        E = np.eye(4)
        E[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        E[:3, 3] = [0.3, -0.2, 0.1]
        S = (R[::2] - E[:3, 3]) @ E[:3, :3]  # propria = Revo deslocado por inv(E)
        T = ur.icp_leve(S, R)
        D = T @ np.linalg.inv(E)
        self.assertLess(float(np.linalg.norm(D[:3, 3])), 0.05)
        self.assertLess(float(np.abs(D[:3, :3] - np.eye(3)).max()), 1e-3)
        chaves = np.array([5, 3, 5, 9, 3, 3])
        soma = np.arange(12.0).reshape(6, 2)
        u, s, c = (gpu.para_cpu(x) for x in fp._agrega(chaves, soma, np.ones(6)))
        np.testing.assert_array_equal(u, [3, 5, 9])
        np.testing.assert_allclose(s, [[2 + 8 + 10, 3 + 9 + 11], [0 + 4, 1 + 5], [6, 7]])
        np.testing.assert_array_equal(c, [3, 2, 1])

    def test_comparar_fusao_avaliar_contra_scipy(self):
        import open3d as o3d
        from comparar_fusao_cuda import avaliar, fundir

        rng = np.random.default_rng(14)
        R = superficie(8000, lado=4.0, ruido=0.002)
        p = R[rng.permutation(len(R))[:6000]] + 0.01
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "ref.ply"
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(R))
            pc.normals = o3d.utility.Vector3dVector(np.tile([0, 0, 1.0], (len(R), 1)))
            o3d.io.write_point_cloud(str(arq), pc, write_ascii=False)
            r = avaliar(p, arq)
        d = cKDTree(R).query(p)[0]
        self.assertAlmostEqual(r["nativo_para_revo"]["mediana_mm"], float(np.median(d)), delta=1e-5)
        self.assertAlmostEqual(r["nativo_para_revo"]["p95_mm"], float(np.percentile(d, 95)), delta=1e-5)
        dr = cKDTree(p).query(R)[0]
        self.assertAlmostEqual(r["revo_para_nativo"]["p90_mm"], float(np.percentile(dr, 90)), delta=1e-5)
        j = cKDTree(R).query(p)[1]
        res = ((p - R[j]) * [0, 0, 1.0]).sum(1)
        m = d < 0.3
        cel = np.floor(p[m] / 30).astype(np.int64)
        u = np.unique(cel, axis=0)
        self.assertEqual(len(r["vies_celulas_30mm"]), len(u[[int((cel == x).all(1).sum()) >= 30 for x in u]]))
        self.assertAlmostEqual(r["vies_celulas_30mm"][0]["mediana_assinada_mm"], float(np.median(res[m])), delta=1e-9)
        pts = np.float32(rng.uniform(-1, 1, (3000, 3)))
        out, _ = fundir(pts, 0.5, 2)
        k = np.floor((pts - (pts.min(0).astype(np.float64) - 0.5)) / 0.5).astype(np.int64)
        u, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
        self.assertEqual(len(out), int((cnt >= 2).sum()))


if __name__ == "__main__":
    unittest.main()
