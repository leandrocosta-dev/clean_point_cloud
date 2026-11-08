"""Triagem, revisoes de consenso e rollback do ciclo sem executar benchmarks CUDA."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ciclo_precisao as ciclo
import numpy as np
import rodar_ciclo_precisao as wrapper
from consenso_visibilidade import carregar_scan, validar_calibracao_fixa

from tests.test_precisao import criar_scan

CALIBRACAO = {"0": {"K": [100, 100, 8, 8], "erro_preview_mm": 0.0}}


class CicloOrquestradoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origem = self.root / "origem"
        for ident in ("scan_a", "scan_b"):
            criar_scan(self.origem / "data" / ident)
        ciclo.salvar_json(
            self.origem / "projeto.revo",
            dict(guid="projeto", nodes=[dict(guid=g, name=g, type=2) for g in ("scan_a", "scan_b")]),
        )
        self.hashes_originais = ciclo.inventario(self.origem)
        self.caminho = self.root / "ciclo"
        self.m = ciclo.iniciar(self.origem, self.caminho)
        self.rev = self.caminho / "r000"
        self.poses = self.root / "poses.json"
        ciclo.salvar_json(self.poses, {g: np.eye(4).tolist() for g in ("scan_a", "scan_b")})
        self.m["calibracao_fixa"] = {g: CALIBRACAO for g in self.m["scans"]}
        ciclo.salvar_json(self.rev / "manifesto.json", self.m)

    def perfil(self, projeto, scans):
        self.assertEqual(Path(projeto), self.rev / "projeto")
        self.assertEqual(scans, self.m["scans"])
        return dict(recomendacao=dict(limpar_raw=dict(ext_max_mm=100, so_peca=True, min_mantido=0.02)))

    def processo_identico(self, cmd, **kwargs):
        self.assertFalse(kwargs["check"])
        self.assertNotIn("REVO_PERMITIR_CPU", kwargs["env"])
        entrada = Path(cmd[2])
        saida = Path(cmd[cmd.index("--saida") + 1])
        self.assertFalse(saida.exists())
        shutil.copytree(entrada, saida)
        _, ids = ciclo.scans(saida)
        if Path(cmd[1]).name == "limpar_raw.py":
            self.assertNotIn("--permitir-parcial", cmd)
            ciclo.salvar_json(saida / "relatorio_limpeza.json", {g: dict(status="limpo") for g in ids})
        else:
            poses = ciclo.ler(cmd[cmd.index("--poses") + 1])
            fixas = ciclo.ler(cmd[cmd.index("--calibracoes") + 1]) if "--calibracoes" in cmd else {}
            self.assertEqual(set(poses), set(ids))
            ciclo.salvar_json(
                saida / "consenso_visibilidade.json",
                dict(
                    estado="revisao_pendente",
                    scans={
                        g: dict(
                            integridade="ok", pixels_antes=1000, pixels_depois=1000, calibracao=fixas.get(g, CALIBRACAO)
                        )
                        for g in poses
                    },
                ),
            )
        return subprocess.CompletedProcess(cmd, 0)

    def auditoria_identica(self, entrada, saida):
        # A fixture do subprocesso copia frames: confirme as identidades antes do remapeamento.
        _, ids = ciclo.scans(saida)
        for g in ids:
            self.assertEqual(
                ciclo.sha256(Path(entrada) / "data" / g / "cache/frames.dataset"),
                ciclo.sha256(Path(saida) / "data" / g / "cache/frames.dataset"),
            )
        return {g: dict(integridade="ok", poses_identicas=True, profundidades_preservadas=True) for g in ids}

    def test_triagem_e_somente_leitura_e_destino_protegido(self):
        congelado = ciclo.inventario(self.rev / "projeto")
        ativa = (self.caminho / "ciclo.json").read_bytes()
        with patch("triagem.triar", side_effect=self.perfil):
            r = ciclo.triagem(self.caminho)
        self.assertEqual(ciclo.inventario(self.rev / "projeto"), congelado)
        self.assertEqual((self.caminho / "ciclo.json").read_bytes(), ativa)
        self.assertEqual(ciclo.ler(r["arquivo"])["ciclo"]["manifesto"], self.m["id"])
        for pasta in (self.origem, self.rev / "projeto"):
            with self.subTest(pasta=pasta), self.assertRaises(ValueError):
                ciclo.triagem(self.caminho, pasta / "perfil.json")

    def test_limpeza_triagem_merge_configuracao_e_aliases_ate_consenso(self):
        config = self.root / "limpar.json"
        ciclo.salvar_json(config, dict(so_peca=False, pixel_tol=0.2))
        with (
            patch("triagem.triar", side_effect=self.perfil),
            patch.object(ciclo.subprocess, "run", side_effect=self.processo_identico) as run,
            patch("auditar_projeto.auditar", side_effect=self.auditoria_identica),
        ):
            ciclo.limpar(self.caminho, config)
            cmd = run.call_args.args[0]
            self.assertNotIn("--so-peca", cmd)
            self.assertEqual(cmd[cmd.index("--ext-max-mm") + 1], "100")
            self.assertEqual(cmd[cmd.index("--pixel-tol") + 1], "0.2")
            ciclo.consenso(self.caminho, self.poses)
        _, _, rev, m = ciclo.carregar(self.caminho)
        self.assertEqual(rev.name, "r002")
        self.assertEqual(set(m["poses_alinhamento"]), set(m["scans"]))
        self.assertEqual({m["identificadores_origem"][g] for g in ("scan_a", "scan_b")}, set(m["scans"]))
        self.assertEqual(set(m["calibracao_fixa"]), set(m["scans"]))
        self.assertEqual(ciclo.inventario(self.origem), self.hashes_originais)

    def test_consenso_remapeia_poses_calibracao_e_invalida_derivados(self):
        st = ciclo.estado(self.rev)
        st.update(
            estado="liberado_para_merge",
            nativo={"antigo": "nativo"},
            revo={"velho": {}},
            comparacao={"velha": {}},
            parecer={},
        )
        ciclo.salvar_json(self.rev / "estado.json", st)
        with (
            patch.dict(os.environ, {"REVO_PERMITIR_CPU": "1"}),
            patch.object(ciclo.subprocess, "run", side_effect=self.processo_identico),
            patch("auditar_projeto.auditar", side_effect=self.auditoria_identica) as audit,
        ):
            r = ciclo.consenso(self.caminho, self.poses)
        audit.assert_called_once()
        _, _, rev, m = ciclo.carregar(self.caminho)
        self.assertEqual(r["revisao"], "r001")
        self.assertNotEqual(m["id"], self.m["id"])
        novo = ciclo.estado(rev)
        for chave in ("nativo", "revo", "comparacao", "revisoes_3d"):
            self.assertEqual(novo[chave], {})
        self.assertNotIn("parecer", novo)
        self.assertEqual(novo["invalidado"], ["fusao", "malha", "comparacao", "merge"])
        self.assertEqual(novo["estado"], "revisao_pendente")
        self.assertEqual({m["identificadores_origem"][g] for g in ("scan_a", "scan_b")}, set(m["scans"]))
        self.assertEqual(ciclo.inventario(self.origem), self.hashes_originais)

    def test_poses_invalidas_ambiguas_ou_repetidas_falham_antes_processar(self):
        matriz = np.eye(4)
        invalidas = [np.zeros((3, 3)).tolist(), (matriz * np.nan).tolist(), np.zeros((4, 4)).tolist()]
        homogenea_ruim = matriz.copy()
        homogenea_ruim[3, 0] = 0.1
        invalidas.append(homogenea_ruim.tolist())
        g = self.m["scans"][0]
        docs = [{"scan_a": T, "scan_b": matriz.tolist()} for T in invalidas]
        docs += [
            {"scan_a": matriz.tolist()},
            {"desconhecido": matriz.tolist(), "scan_b": matriz.tolist()},
            {"scan_a": matriz.tolist(), g: matriz.tolist(), "scan_b": matriz.tolist()},
        ]
        with patch.object(ciclo.subprocess, "run") as run:
            for doc in docs:
                ciclo.salvar_json(self.poses, doc)
                with self.subTest(doc=doc), self.assertRaises(ValueError):
                    ciclo.consenso(self.caminho, self.poses)
            self.poses.write_text('{"scan_a": [], "scan_a": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repetida"):
                ciclo.consenso(self.caminho, self.poses)
        run.assert_not_called()
        self.assertEqual(ciclo.ler(self.caminho / "ciclo.json")["ativa"], "r000")

    def test_falha_parcial_e_falha_auditoria_preservam_ativa(self):
        for codigo in (1, 2):
            with self.subTest(codigo=codigo):
                antes = (self.caminho / "ciclo.json").read_bytes(), (self.rev / "estado.json").read_bytes()
                with patch.object(ciclo.subprocess, "run", return_value=subprocess.CompletedProcess([], codigo)):
                    with self.assertRaises(ciclo.ResultadoParcial if codigo == 2 else subprocess.CalledProcessError):
                        ciclo.consenso(self.caminho, self.poses)
                self.assertEqual(
                    antes, ((self.caminho / "ciclo.json").read_bytes(), (self.rev / "estado.json").read_bytes())
                )
        with (
            patch.object(ciclo.subprocess, "run", side_effect=self.processo_identico),
            patch("auditar_projeto.auditar", side_effect=AssertionError("pose alterada")),
            self.assertRaisesRegex(AssertionError, "pose alterada"),
        ):
            ciclo.consenso(self.caminho, self.poses)
        self.assertEqual(ciclo.ler(self.caminho / "ciclo.json")["ativa"], "r000")
        self.assertFalse((self.caminho / "r001").exists())

    def test_relatorio_parcial_ou_calibracao_alterada_nao_publicam(self):
        for parcial in (True, False):

            def processo(cmd, **kw):
                r = self.processo_identico(cmd, **kw)
                path = Path(cmd[cmd.index("--saida") + 1]) / "consenso_visibilidade.json"
                rel = ciclo.ler(path)
                if parcial:
                    rel["estado"] = "parcial"
                else:
                    row = next(iter(rel["scans"].values()))
                    row["calibracao"]["0"]["K"][0] = 123
                ciclo.salvar_json(path, rel)
                return r

            with (
                self.subTest(parcial=parcial),
                patch.object(ciclo.subprocess, "run", side_effect=processo),
                self.assertRaises(ciclo.ResultadoParcial if parcial else ValueError),
            ):
                ciclo.consenso(self.caminho, self.poses)
        self.assertEqual(ciclo.ler(self.caminho / "ciclo.json")["ativa"], "r000")

    def test_wrapper_encaminha_poses_e_saida_da_triagem(self):
        for acao, extras in (
            ("consenso", ["--poses", str(self.poses)]),
            ("triagem", ["--saida", str(self.root / "perfil.json")]),
        ):
            with (
                patch.object(sys, "argv", ["wrapper", "--acao", acao, "--ciclo", str(self.caminho), *extras]),
                patch.object(wrapper.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)) as run,
                self.assertRaises(SystemExit) as encerrado,
            ):
                wrapper.main()
            self.assertEqual(encerrado.exception.code, 2)
            self.assertEqual(run.call_args.args[0][-2:], extras)

    def test_cli_parcial_tem_codigo_2(self):
        with (
            patch.object(sys, "argv", ["ciclo", "consenso", str(self.caminho), "--poses", str(self.poses)]),
            patch.object(ciclo, "consenso", side_effect=ciclo.ResultadoParcial("parcial")),
            contextlib.redirect_stdout(io.StringIO()) as saida,
            self.assertRaises(SystemExit) as exc,
        ):
            ciclo.main()
        self.assertEqual(exc.exception.code, 2)
        self.assertTrue(json.loads(saida.getvalue())["ativa_preservada"])


class CalibracaoConsensoTest(unittest.TestCase):
    def test_validacao_cobre_modos_e_rejeita_foco_invalido(self):
        self.assertEqual(validar_calibracao_fixa(CALIBRACAO, [0])[0]["K"], [100, 100, 8, 8])
        for calib, modos in (
            (CALIBRACAO, [0, 1]),
            ({"0": {"K": [0, 100, 8, 8]}}, [0]),
            ({"0": {"K": [1, np.inf, 8, 8]}}, [0]),
        ):
            with self.subTest(calib=calib), self.assertRaises(ValueError):
                validar_calibracao_fixa(calib, modos)

    def test_calibracao_fixa_nao_recalibra_e_respeita_modo_do_quadro(self):
        # Scan sem pixels: exercita os modos e o contrato de K sem calcular na GPU.
        class Scan:
            modo = np.array([0, 1])
            tem_prof = np.array([True, True])
            ids = [1, 2]
            T = np.array([np.eye(4), np.eye(4)])

            def __init__(self, *args):
                self.K = {}
                self.c = unittest.mock.Mock()
                self.calibrar = unittest.mock.Mock(side_effect=AssertionError("recalibracao indevida"))

            def pontos(self, k, **kwargs):
                self.projetados.append((k, self.K[int(self.modo[k])]))
                return None, None

            projetados = []

        # O retorno sem observacoes evita kernels; sc.K ainda precisa conter ambos os modos.
        scan = Scan()
        fixas = {"0": CALIBRACAO["0"], "1": dict(K=[200, 201, 9, 10], erro_preview_mm=0)}
        with (
            patch("revo_projeto.RevoScan", return_value=scan),
            patch("consenso_visibilidade.gpu.xp", return_value=np),
        ):
            dados = carregar_scan("nao_usado", "s", np.eye(4), 0.5, 1, "tmp", fixas)
        scan.calibrar.assert_not_called()
        self.assertEqual(dados["pixels"], 0)
        self.assertEqual(scan.K[0], (100, 100, 8, 8))
        self.assertEqual(scan.K[1], (200, 201, 9, 10))
        self.assertEqual(scan.projetados, [(0, scan.K[0]), (1, scan.K[1])])
        scan.c.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
