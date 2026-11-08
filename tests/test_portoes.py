"""Portoes da Fase 1: recursos, consenso, alinhar_direto, alinhar_varios, exclusoes, limpar_raw e rodar_tudo."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from tests.gpu_utils import CUDA_DISPONIVEL

RAIZ = Path(__file__).resolve().parents[1]
PIPE = RAIZ / "pipeline"
PY = sys.executable

import alinhar_direto as AD
import alinhar_varios as AV
import limpar_por_consenso as LC
import revo_projeto as RP

from tests.test_precisao import criar_scan


def rodar(codigo, **env):
    e = dict(os.environ)
    for k in ("REVO_RECURSOS", "REVO_TETO_THREADS", "REVO_THREADS"):
        e.pop(k, None)
    e.update(env)
    r = subprocess.run([PY, "-c", codigo], capture_output=True, text=True, env=e, cwd=str(PIPE))
    if r.returncode:
        raise AssertionError(r.stderr)
    return r.stdout.strip()


class RecursosTest(unittest.TestCase):
    COD = (
        "import os,recursos as r;j,t=r.orcamento_threads(4);j2,t2=r.orcamento_threads(4,teto=4);"
        "j3,t3=r.orcamento_threads(99);print(r.MODO,r.TETO_THREADS,j,t,t2,j3,t3,os.environ['OMP_NUM_THREADS'],r.threads_atuais())"
    )

    def test_modo_conservador_teto_12(self):
        modo, teto, j, t, t2, j3, t3, omp, atual = rodar(self.COD, REVO_RECURSOS="conservador").split()
        cpus = os.cpu_count()
        self.assertEqual(modo, "conservador")
        self.assertEqual(int(teto), 12)
        self.assertEqual((int(j), int(t)), (4, min(12, cpus) // 4))
        self.assertEqual(int(t2), 1)  # teto menor do chamador vale no conservador
        self.assertLessEqual(int(j3) * int(t3), 12)
        self.assertLessEqual(int(atual), 12)

    def test_modo_maximo_usa_todos(self):
        modo, teto, j, t, t2, *_ = rodar(self.COD).split()
        cpus = os.cpu_count()
        self.assertEqual(modo, "maximo")
        self.assertEqual(int(teto), cpus)
        self.assertEqual(int(t), cpus // 4)
        self.assertEqual(int(t2), cpus // 4)  # teto menor do chamador ignorado

    def test_teto_fixo_por_variavel(self):
        _, teto, j, t, *_ = rodar(self.COD, REVO_TETO_THREADS="5").split()
        self.assertEqual((int(teto), int(j), int(t)), (5, 4, 1))

    def test_threads_revo_projeto_por_modo(self):
        cod = "import revo_projeto as p;print(p.THREADS)"
        self.assertEqual(int(rodar(cod, REVO_RECURSOS="conservador")), min(12, os.cpu_count()))
        self.assertEqual(int(rodar(cod, REVO_RECURSOS="conservador", REVO_THREADS="-1")), 12)
        self.assertEqual(int(rodar(cod, REVO_THREADS="3")), 3)
        self.assertEqual(int(rodar(cod)), os.cpu_count())

    def test_baixa_prioridade_reexportada(self):
        import recursos

        self.assertIs(AV.baixa_prioridade, recursos.baixa_prioridade)
        self.assertIs(LC.baixa_prioridade, recursos.baixa_prioridade)


def plano(n=40, z=0.0, passo=1.0, x0=0.0):
    g = np.arange(n) * passo
    x, y = np.meshgrid(g + x0, g)
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class ConsensoTest(unittest.TestCase):
    def test_validar(self):
        for args in [(1, 1, 1, "euclidiano"), (5, 6, 2, "euclidiano"), (3, 2, 3, "euclidiano"), (64, 2, 2, "voxel")]:
            with self.assertRaises(LC.ConsensoInviavel):
                LC.validar(*args)
        LC.validar(6, 6, 2, "euclidiano")
        LC.validar(64, 6, 2, "euclidiano")

    def test_popcount_64_bits(self):
        m = np.array([1, 1 << 48, 1 << 62, (1 << 63) - 1, 0], dtype=np.uint64)
        np.testing.assert_array_equal(LC.popcount(m), [1, 1, 1, 63, 0])

    def test_euclidiano_distancia_real_e_nucleo(self):
        A = plano()
        B = plano(z=0.2)
        C = np.vstack([plano(z=-0.2), [[20, 20, 1.5]], [[20, 20, 30.0]]])  # 1,5 mm e 30 mm fora
        with tempfile.TemporaryDirectory() as t:
            npz = os.path.join(t, "c.npz")
            info = LC.construir_euclidiano([A, B, C], nucleo=3, tol=1.0, saida_npz=npz)
            self.assertEqual(info["nucleo"], 3 * len(A))
            dec = LC.Decisor(npz, bit=2, apoio_outros=2, tol=1.0, raio=15.0)
            ok = dec(np.array([[10, 10, -0.2], [20, 20, 1.5], [20, 20, 30.0], [10, 10, 1.1]]))
            # 1,5 mm e 1,1 mm do apoio nao passam a tol 1 mm (o metodo voxel aceitava ate 3,46 mm)
            np.testing.assert_array_equal(ok, [True, False, False, False])
            dec_raio = LC.Decisor(npz, bit=2, apoio_outros=0, tol=1.0, raio=15.0)
            np.testing.assert_array_equal(dec_raio(np.array([[20, 20, 14.0], [20, 20, 16.0]])), [True, False])

    def test_nucleo_vazio_detectado(self):
        with tempfile.TemporaryDirectory() as t:
            info = LC.construir_euclidiano(
                [plano(x0=0), plano(x0=500), plano(x0=1000)], 2, 1.0, os.path.join(t, "c.npz")
            )
            self.assertEqual(info["nucleo"], 0)

    def test_voxel_raio_zero_nao_dilata(self):
        with tempfile.TemporaryDirectory() as t:
            A = plano(n=10, passo=3.0)
            r0 = LC.construir_voxel([A, A.copy()], 2, 0.0, os.path.join(t, "a.npz"))
            r9 = LC.construir_voxel([A, A.copy()], 2, 9.0, os.path.join(t, "b.npz"))
            self.assertLessEqual(r0["voxels_nucleo_dilatado"], r0["nucleo"])
            self.assertGreater(r9["voxels_nucleo_dilatado"], r0["voxels_nucleo_dilatado"])

    def _segmentos(self, root, nuvens):
        pasta = root / "seg"
        (pasta / "c").mkdir(parents=True)
        ind, segs = {}, {}
        for i, X in enumerate(nuvens):
            g = f"scan{i}"
            arq = pasta / "c" / f"{g}.npy"
            np.save(arq, X.astype(np.float32))
            ind[g] = [dict(scan=g, seg=0, k0=0, k1=4, arquivo=str(arq))]
            segs[g] = [dict(seg=0, k0=0, k1=4, T=np.eye(4).tolist())]
        (pasta / "segmentos_indice.json").write_text(json.dumps(ind))
        (pasta / "direto.json").write_text(json.dumps(dict(referencia="scan0", segmentos=segs)))
        return pasta

    def test_cli_aborta_sem_gravar(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "proj" / "data").mkdir(parents=True)
            pasta = self._segmentos(root, [plano(x0=0), plano(x0=500), plano(x0=1000)])
            for extra, msg in ((["--nucleo", "99"], "CONSENSO INVIAVEL"), (["--nucleo", "2"], "nucleo vazio")):
                saida = root / ("saida" + extra[-1])
                r = subprocess.run(
                    [
                        PY,
                        str(PIPE / "limpar_por_consenso.py"),
                        str(root / "proj"),
                        str(pasta),
                        str(pasta / "direto.json"),
                        "--saida",
                        str(saida),
                        *extra,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(msg, r.stdout + r.stderr)
                self.assertFalse(saida.exists())

    def test_limpar_scan_remove_so_o_sem_apoio(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            criar_scan(root / "proj/data/scanA")  # quadro 2 tem um pixel a 95 mm (5 mm fora do plano)
            sc = RP.RevoScan(str(root / "proj/data/scanA"), t)
            try:
                plano_a = sc.pontos(0, K=(100, 100, 8, 8))[0]
            finally:
                sc.c.close()
            npz = os.path.join(t, "c.npz")
            info = LC.construir_euclidiano([plano_a, plano_a + [0, 0, 0.1]], nucleo=2, tol=1.0, saida_npz=npz)
            self.assertGreater(info["nucleo"], 0)
            segs = [dict(seg=0, k0=0, k1=4, T=np.eye(4).tolist())]
            with patch.object(RP.RevoScan, "calibrar", return_value=((100, 100, 8, 8), 0.0)):
                st = LC._limpar_scan((str(root / "proj"), str(root / "out"), "scanA", 0, segs, npz, 1, 1, 1.0, 15.0))
            self.assertEqual(st["antes"] - st["depois"], 1)
            self.assertEqual(st["integridade"], "ok")
            self.assertEqual(st["apagados"], 0)


def superficie(n=60, passo=1.5):
    g = np.arange(n) * passo
    x, y = np.meshgrid(g, g)
    z = 3 * np.sin(x / 9.0) * np.cos(y / 11.0) + 0.02 * x
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def rot(eixo, graus):
    eixo = np.asarray(eixo, float) / np.linalg.norm(eixo)
    T = AD.exp_se3(np.r_[eixo * np.radians(graus), 0, 0, 0])
    return T


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class AlinharDiretoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        X = superficie()
        cls.nos = []
        for i in range(2):
            arq = os.path.join(cls.tmp.name, f"n{i}.npy")
            np.save(arq, X)
            cls.nos.append(dict(scan=f"s{i}", seg=0, k0=0, k1=0, arquivo=arq))
        erro = rot([1, 2, 0.5], 0.4)
        erro[:3, 3] = [0.5, -0.4, 0.6]
        cls.T_ini = [np.eye(4), erro]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tukey_e_conectividade(self):
        self.assertEqual(AD.tukey_c(6.0), 1.5)
        self.assertEqual(AD.tukey_c(1.2), 0.5)
        self.assertEqual(AD.tukey_c(6.0, "0.5"), 0.5)
        self.assertEqual(AD.conectividade(4, [(0, 1)], [(2, 3)], 0), [2, 3])
        self.assertEqual(AD.conectividade(3, [(0, 1)], [(1, 2)], 0), [])

    def erro(self, T):
        D = np.linalg.inv(T[0]) @ T[1]
        return np.linalg.norm(D[:3, 3]) + 100 * np.linalg.norm(D[:3, :3] - np.eye(3))

    def test_custo_nunca_sobe_e_melhora(self):
        T, hist = AD.otimizar_direto(
            self.nos,
            [x.copy() for x in self.T_ini],
            [(0, 1)],
            [],
            0,
            iteracoes=6,
            amostra=3000,
            tol_ini=3.0,
            tol_fim=1.0,
            log=lambda *a, **k: None,
        )
        self.assertEqual(len(hist), 6)
        for h in hist:
            self.assertLessEqual(h["custo_depois"], h["custo_antes"] + 1e-9)
            self.assertTrue(np.isfinite(h["rms_ponderado_um"]))
        self.assertTrue(any(h["aceito"] for h in hist))
        self.assertLess(self.erro(T), 0.3 * self.erro(self.T_ini))
        np.testing.assert_array_equal(T[0], np.eye(4))  # referencia fixa
        (antes, depois), n = AD.vies_reservado(self.nos, [self.T_ini, T], [(0, 1)])
        self.assertEqual(n, 1)
        self.assertLess(depois, antes)

    def test_passo_ruim_rejeitado(self):
        real = np.linalg.solve
        with patch.object(AD.np.linalg, "solve", side_effect=lambda A, b: real(A, b) * 0 + 5.0):
            T, hist = AD.otimizar_direto(
                self.nos,
                [x.copy() for x in self.T_ini],
                [(0, 1)],
                [],
                0,
                iteracoes=2,
                amostra=2000,
                tol_ini=2.0,
                tol_fim=1.0,
                log=lambda *a, **k: None,
            )
        self.assertFalse(any(h["aceito"] for h in hist))
        for a, b in zip(T, self.T_ini):
            np.testing.assert_array_equal(a, b)


class AlinharVariosTest(unittest.TestCase):
    def test_escala_variavel(self):
        for v, esperado in (("0.2", 0.2), ("abc", 1.0), ("5", 1.0), ("0.01", 0.1)):
            with patch.dict(os.environ, {"ALINHAR_ESCALA": v}):
                self.assertAlmostEqual(AV.escala(), esperado)

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_escala_auto(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:

            def arq(lado):
                p = os.path.join(t, f"{lado}.npy")
                np.save(p, rng.uniform(0, lado, (20000, 3)).astype(np.float32))
                return p

            e, d = AV.escala_auto([arq(50)])
            self.assertAlmostEqual(d, 50 * np.sqrt(3) * 0.98, delta=3)
            self.assertAlmostEqual(e, d / 350, places=6)
            self.assertEqual(AV.escala_auto([arq(2000)])[0], 1.0)
            self.assertEqual(AV.escala_auto([arq(10)])[0], 0.15)


def seg_minimo(**extra):
    seg = dict(
        n=np.array([0, 0, 1.0]),
        d=0.0,
        lo=np.zeros(3),
        voxel=1.0,
        manter=np.zeros((1, 1, 1), bool),
        u=np.array([1.0, 0, 0]),
        v=np.array([0, 1.0, 0]),
        uv_lo=np.zeros(2),
        pegada=np.zeros((1, 1), bool),
        hcol=np.zeros((1, 1)),
        faixa_base=1.0,
        h_max=100.0,
    )
    seg.update(extra)
    return seg


class ExclusoesTest(unittest.TestCase):
    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_carregar_e_aplicar(self):
        with tempfile.TemporaryDirectory() as t:
            np.save(os.path.join(t, "sup.npy"), np.array([[50.0, 50, 50]]))
            arq = os.path.join(t, "ex.json")
            Path(arq).write_text(
                json.dumps(
                    {
                        "*": [{"tipo": "caixa", "min": [0, 0, 0], "max": [1, 1, 1]}],
                        "A": [
                            {"tipo": "esfera", "centro": [10, 0, 0], "raio": 2},
                            {"tipo": "semiespaco", "normal": [0, 0, 1], "d": -20},
                            {"tipo": "pontos", "arquivo": "sup.npy", "raio": 1},
                        ],
                    }
                )
            )
            self.assertEqual(len(RP.carregar_exclusoes(arq, "B")), 1)
            reg = RP.carregar_exclusoes(arq, "A")
            P = np.array([[0.5, 0.5, 0.5], [11, 0, 0], [5, 5, 25], [50, 50, 50.5], [5, 5, 5]])
            np.testing.assert_array_equal(RP.dentro_exclusoes(P, reg), [True, True, True, True, False])

    def test_formatos_invalidos(self):
        ruins = [
            {"tipo": "cubo"},
            {"tipo": "caixa", "min": [0, 0], "max": [1, 1, 1]},
            {"tipo": "caixa", "min": [2, 2, 2], "max": [1, 1, 1]},
            {"tipo": "esfera", "centro": [0, 0, 0], "raio": -1},
            {"tipo": "semiespaco", "normal": [0, 0, 0], "d": 1},
            {"tipo": "esfera", "centro": [0, 0, 0]},
        ]
        with tempfile.TemporaryDirectory() as t:
            for r in ruins:
                arq = os.path.join(t, "ex.json")
                Path(arq).write_text(json.dumps({"*": [r]}))
                with self.assertRaises((ValueError, KeyError), msg=str(r)):
                    RP.carregar_exclusoes(arq, "A")

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_exclusao_prevalece_sobre_fuse(self):
        P = np.array([[0, 0, 5.0], [10, 0, 5.0]])
        seg = seg_minimo(ref_arvore=cKDTree(P), ref_margem=2.5)
        np.testing.assert_array_equal(RP.mascara_manter(P, seg), [True, True])  # recuperados pelo fuse
        seg["exclusoes"] = [dict(tipo="esfera", centro=[10, 0, 5], raio=1.0)]
        np.testing.assert_array_equal(RP.mascara_manter(P, seg), [True, False])

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_preview_modo_precisao_nao_tira_ilhas_candidatas(self):
        for precisao, esperado in ((True, 1), (False, 0)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                criar_scan(root / "scan")
                (root / "work").mkdir()
                sc = RP.RevoScan(str(root / "scan"), str(root / "work"))
                sc.K = {0: (100, 100, 8, 8)}
                try:
                    with (
                        patch("revo_projeto.mascara_manter", side_effect=lambda p, s: np.ones(len(p), bool)),
                        patch("revo_projeto.ilhas_3d", return_value=dict(n_comp=2, n_removidas=1)),
                        patch("revo_projeto.mascara_ilhas", side_effect=lambda p, i: np.zeros(len(p), bool)),
                    ):
                        RP.limpar_scan(
                            sc,
                            {},
                            str(root / "limpo"),
                            ilhas=True,
                            pixels=False,
                            precisao=precisao,
                            temporal_passo=1,
                            zerar_vazios=True,
                        )
                    _, prev = RP.ler_ply(str(root / "limpo/raw_preview.ply"))
                    self.assertEqual(len(prev), esperado, f"precisao={precisao}")
                finally:
                    sc.c.close()


class _Exec:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, proj, scans, *resto):
        return [(s, dict(status="revisar" if s == "b" else "limpo"), []) for s in scans]


class LimparRawTest(unittest.TestCase):
    def projeto(self, root):
        p = root / "proj"
        (p / "data").mkdir(parents=True)
        (p / "p.revo").write_text(json.dumps({"nodes": [{"guid": "a"}, {"guid": "b"}]}))
        return p

    def rodar_main(self, argv):
        import limpar_raw

        with (
            patch.object(sys, "argv", ["limpar_raw.py"] + argv),
            patch.object(limpar_raw, "ProcessPoolExecutor", _Exec),
            patch.object(limpar_raw, "gravar_revo"),
            patch("relatorio_qualidade.gravar_relatorio"),
        ):
            limpar_raw.main()

    def test_parcial_sai_com_2(self):
        with tempfile.TemporaryDirectory() as t:
            p = self.projeto(Path(t))
            with self.assertRaises(SystemExit) as cm:
                self.rodar_main([str(p), "--saida", str(Path(t) / "s1"), "--manter-ids"])
            self.assertEqual(cm.exception.code, 2)
            self.assertEqual(json.loads((Path(t) / "s1/relatorio_limpeza.json").read_text())["b"]["status"], "revisar")
            self.rodar_main([str(p), "--saida", str(Path(t) / "s2"), "--manter-ids", "--permitir-parcial"])

    def test_exclusoes_invalidas_antes_de_gravar(self):
        with tempfile.TemporaryDirectory() as t:
            p = self.projeto(Path(t))
            ex = Path(t) / "ex.json"
            ex.write_text(json.dumps({"*": [{"tipo": "cubo"}]}))
            for arq in (str(ex), str(Path(t) / "nao_existe.json")):
                with self.assertRaises(SystemExit) as cm:
                    self.rodar_main([str(p), "--saida", str(Path(t) / "s"), "--exclusoes", arq])
                self.assertEqual(cm.exception.code, 2)
                self.assertFalse((Path(t) / "s").exists())


class RodarTudoTest(unittest.TestCase):
    def rodar(self, root, extra, rc=0, recomendacao=None):
        import rodar_tudo

        p = root / "proj"
        p.mkdir(exist_ok=True)
        (p / "p.revo").write_text('{"nodes": []}')
        chamadas = []

        def falso(cmd, *a, **k):
            chamadas.append(cmd)
            saida = Path(cmd[cmd.index("--saida") + 1])
            saida.mkdir(parents=True, exist_ok=True)
            (saida / "p_limpo.revo").write_text("{}")
            return subprocess.CompletedProcess(cmd, rc)

        perfil = dict(recomendacao=dict(caso="teste", limpar_raw=recomendacao or {}))
        with (
            patch.object(
                sys, "argv", ["rodar_tudo.py", str(p), "--so-limpar", "--saida-limpo", str(root / "limpo")] + extra
            ),
            patch.object(rodar_tudo.subprocess, "run", side_effect=falso),
            patch("preparar_importacao.reidentificar", return_value={}),
            patch("triagem.triar", return_value=perfil),
        ):
            try:
                rodar_tudo.main()
                codigo = 0
            except SystemExit as e:
                codigo = e.code
        return chamadas, codigo

    def test_repassa_opcoes(self):
        with tempfile.TemporaryDirectory() as t:
            ex = Path(t) / "ex.json"
            ex.write_text("{}")
            ch, cod = self.rodar(
                Path(t), ["--ext-max-mm", "100", "--so-peca", "--min-mantido", "0.02", "--exclusoes", str(ex)]
            )
            c1 = ch[0]
            self.assertEqual(c1[c1.index("--ext-max-mm") + 1], "100.0")
            self.assertIn("--so-peca", c1)
            self.assertEqual(c1[c1.index("--min-mantido") + 1], "0.02")
            self.assertEqual(c1[c1.index("--exclusoes") + 1], str(ex.resolve()))
            self.assertEqual(cod, 0)

    def test_parcial_nao_aciona_revo(self):
        with tempfile.TemporaryDirectory() as t:
            ch, cod = self.rodar(Path(t), [], rc=2)
            self.assertEqual(len(ch), 1)
            self.assertIn("PARCIAL", str(cod))

    def test_auto_usa_recomendacao_sem_sobrescrever_usuario(self):
        rec = dict(ext_max_mm=124, so_peca=True, min_mantido=0.02)
        with tempfile.TemporaryDirectory() as t:
            c1 = self.rodar(Path(t), ["--auto"], recomendacao=rec)[0][0]
            self.assertEqual(c1[c1.index("--ext-max-mm") + 1], "124")
            self.assertIn("--so-peca", c1)
            self.assertEqual(c1[c1.index("--min-mantido") + 1], "0.02")
        with tempfile.TemporaryDirectory() as t:
            c1 = self.rodar(Path(t), ["--auto", "--ext-max-mm", "60"], recomendacao=rec)[0][0]
            self.assertEqual(c1[c1.index("--ext-max-mm") + 1], "60.0")


class CicloTest(unittest.TestCase):
    def test_ciclo_confere_pendencias_por_conta(self):
        import ciclo_precisao as ciclo

        with tempfile.TemporaryDirectory() as t:
            raiz = Path(t)
            origem = raiz / "origem"
            for scan in ("a", "b"):
                criar_scan(origem / "data" / scan)
            (origem / "projeto.revo").write_text(
                json.dumps({"nodes": [{"guid": scan, "name": scan} for scan in ("a", "b")]}), encoding="utf-8"
            )
            (origem / "identificadores.json").write_text(json.dumps({"orig_a": "a", "orig_b": "b"}), encoding="utf-8")
            experimento = raiz / "ciclo"
            ciclo.iniciar(origem, experimento)
            _, _, ativa, manifesto = ciclo.carregar(experimento)
            self.assertEqual(set(manifesto["identificadores_origem"]), {"orig_a", "orig_b", "a", "b"})
            poses_originais = raiz / "poses_originais.json"
            ciclo.salvar_json(poses_originais, {"orig_a": np.eye(4).tolist(), "orig_b": np.eye(4).tolist()})
            poses_ativas = ciclo._poses_ativas(poses_originais, ativa, manifesto)
            self.assertEqual(set(poses_ativas), set(manifesto["scans"]))
            hash_origem = ciclo.inventario(origem)
            hash_ativa = ciclo.inventario(ativa / "projeto")
            checkpoint = (experimento / "ciclo.json").read_bytes()

            def processo_parcial(cmd, **kwargs):
                self.assertFalse(kwargs["check"])
                self.assertNotIn("REVO_PERMITIR_CPU", kwargs["env"])
                entrada = Path(cmd[2])
                saida = Path(cmd[cmd.index("--saida") + 1])
                import shutil

                shutil.copytree(entrada, saida)
                ciclo.salvar_json(
                    saida / "relatorio_limpeza.json",
                    {
                        scan: {"status": "revisar" if scan == manifesto["scans"][1] else "limpo"}
                        for scan in manifesto["scans"]
                    },
                )
                return subprocess.CompletedProcess(cmd, 0)

            with (
                patch("triagem.triar", return_value={"recomendacao": {"limpar_raw": {}}}),
                patch.object(ciclo.subprocess, "run", side_effect=processo_parcial),
                self.assertRaises(ciclo.ResultadoParcial),
            ):
                ciclo.limpar(experimento)

            self.assertEqual((experimento / "ciclo.json").read_bytes(), checkpoint)
            self.assertEqual(ciclo.inventario(origem), hash_origem)
            self.assertEqual(ciclo.inventario(ativa / "projeto"), hash_ativa)
            self.assertEqual(ciclo.ler(experimento / "ciclo.json")["ativa"], ativa.name)
            preparados = list(experimento.glob("r001_preparando_*"))
            self.assertEqual(len(preparados), 1)
            self.assertEqual(ciclo.ler(preparados[0] / "falha.json")[manifesto["scans"][1]]["status"], "revisar")
            with self.assertRaisesRegex(ValueError, "merge bloqueado"):
                ciclo.exportar_merge(experimento, raiz / "entrega")

            def processo_consenso(cmd, **kwargs):
                entrada = Path(cmd[2])
                saida = Path(cmd[cmd.index("--saida") + 1])
                import shutil

                shutil.copytree(entrada, saida)
                _, ids = ciclo.scans(saida)
                ciclo.salvar_json(
                    saida / "consenso_visibilidade.json",
                    {
                        "estado": "revisao_pendente",
                        "scans": {
                            scan: {
                                "integridade": "ok",
                                "pixels_antes": 1000,
                                "pixels_depois": 950,
                                "calibracao": {"0": {"K": [100, 100, 8, 8], "erro_preview_mm": 0.0}},
                            }
                            for scan in ids
                        },
                    },
                )
                return subprocess.CompletedProcess(cmd, 0)

            with (
                patch.object(ciclo.subprocess, "run", side_effect=processo_consenso),
                patch("auditar_projeto.auditar", return_value={"ok": True}),
            ):
                ciclo.consenso(experimento, poses_originais)
            _, _, revisao_nova, manifesto_novo = ciclo.carregar(experimento)
            self.assertTrue({"orig_a", "orig_b", "a", "b"}.issubset(manifesto_novo["identificadores_origem"]))
            self.assertEqual(
                set(manifesto_novo["identificadores_origem"][x] for x in ("orig_a", "orig_b")),
                set(manifesto_novo["scans"]),
            )
            poses_remapeadas = ciclo._poses_ativas(poses_originais, revisao_nova, manifesto_novo)
            self.assertEqual(set(poses_remapeadas), set(manifesto_novo["scans"]))


if __name__ == "__main__":
    unittest.main()
