"""Testes de proveniencia e portoes da limpeza de qualidade 3D."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ciclo_precisao as ciclo
import limpar_qualidade_3d as qualidade
import numpy as np
from comparar_fusao_cuda import validar_ply
from comum import gravar_ply
from revo_projeto import RevoScan

from tests.test_precisao import criar_scan


class LimparQualidade3DTest(unittest.TestCase):
    def _projeto(self, root):
        projeto = root / "projeto"
        for guid in ("scan_a", "scan_b"):
            criar_scan(projeto / "data" / guid)
        (projeto / "projeto.revo").write_text(
            json.dumps({"guid": "projeto", "nodes": [{"guid": g, "name": g} for g in ("scan_a", "scan_b")]}),
            encoding="utf-8",
        )
        return projeto

    def test_mascara_remove_99_e_preserva_pixel_exclusivo_frameinfo_e_origem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "scan"
            dst = root / "copia"
            criar_scan(src)
            shutil.copytree(src, dst)
            with tempfile.TemporaryDirectory() as work:
                scan = RevoScan(str(src), work)
                frame_ids = list(scan.ids)
                before_info = scan.c.execute("select id, bin from FrameInfo order by id").fetchall()
                scan.c.close()
            obs = np.zeros(100, dtype=[("p", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")])
            obs["p"] = [0, 0, -100]
            obs["quadro"] = 0
            obs["pixel"] = np.arange(100, dtype=np.int32)
            before_hash = ciclo.sha256(src / "cache" / "frames.dataset")
            stats, mascara = qualidade._escrever_frames(src, dst, np.arange(99), obs)
            with tempfile.TemporaryDirectory() as work:
                after = RevoScan(str(dst), work)
                try:
                    self.assertEqual(
                        after.c.execute("select id, bin from FrameInfo order by id").fetchall(), before_info
                    )
                    kept = after.profundidade(frame_ids[0])
                    self.assertEqual(int(kept[2].sum()), 157)
                    self.assertTrue(kept[2][99])
                    self.assertEqual(after.c.execute("pragma integrity_check").fetchone()[0], "ok")
                finally:
                    after.c.close()
            self.assertEqual(ciclo.sha256(src / "cache" / "frames.dataset"), before_hash)
            self.assertEqual(stats["pixels_antes"] - stats["pixels_depois"], 99)
            self.assertEqual(mascara["quadros"].tolist(), [0])
            pixels_marcados = np.unpackbits(mascara["bits"], bitorder="little")[: 16 * 16]
            self.assertEqual(int(pixels_marcados.sum()), 99)
            self.assertFalse(pixels_marcados[99])

    def test_frame_totalmente_ruim_apaga_so_rgbd_e_preserva_frameinfo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / "scan", root / "copia"
            criar_scan(src)
            shutil.copytree(src, dst)
            with tempfile.TemporaryDirectory() as work:
                scan = RevoScan(str(src), work)
                before_info = scan.c.execute("select id, bin from FrameInfo order by id").fetchall()
                scan.c.close()
            obs = np.zeros(256, dtype=[("p", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")])
            obs["p"] = [0, 0, -100]
            obs["pixel"] = np.arange(256, dtype=np.int32)
            stats, mascara = qualidade._escrever_frames(src, dst, np.arange(256), obs)
            with tempfile.TemporaryDirectory() as work:
                after = RevoScan(str(dst), work)
                try:
                    self.assertEqual(stats["quadros_zerados"], 1)
                    self.assertFalse(after.tem_prof[0])
                    self.assertTrue(after.tem_prof[1:].all())
                    self.assertEqual(
                        after.c.execute("select id, bin from FrameInfo order by id").fetchall(), before_info
                    )
                    self.assertEqual(after.c.execute("pragma integrity_check").fetchone()[0], "ok")
                finally:
                    after.c.close()
            self.assertEqual(mascara["observacoes"].tolist(), list(range(256)))

    def test_scan_sem_pixels_restantes_recusa_copia(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / "scan", root / "copia"
            criar_scan(src)
            shutil.copytree(src, dst)
            dtype = [("p", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")]
            obs = np.zeros(5 * 256, dtype=dtype)
            obs["p"] = [0, 0, -100]
            obs["quadro"] = np.repeat(np.arange(5, dtype=np.int32), 256)
            obs["pixel"] = np.tile(np.arange(256, dtype=np.int32), 5)
            obs["p"][2 * 256 + 136, 2] = -95
            before = ciclo.sha256(dst / "cache" / "frames.dataset")
            with self.assertRaisesRegex(ValueError, "sem observacoes"):
                qualidade._escrever_frames(src, dst, np.arange(len(obs)), obs)
            self.assertEqual(ciclo.sha256(dst / "cache" / "frames.dataset"), before)

    def test_hash_dataset_obsoleto_recusa_antes_de_criar_destino(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projeto = self._projeto(root)
            poses_path = root / "poses.json"
            ciclo.salvar_json(poses_path, {g: np.eye(4).tolist() for g in ("scan_a", "scan_b")})
            saida = root / "limpo"
            preparado = {
                "hashes": {g: {"dataset_sha256": "proposta-antiga"} for g in ("scan_a", "scan_b")},
            }
            with (
                patch("medicoes_qualidade_3d.preparar_medicoes", return_value=preparado),
                self.assertRaisesRegex(ValueError, "mudou depois da medicao"),
            ):
                qualidade.aplicar(projeto, poses_path, saida)
            self.assertFalse(saida.exists())
            self.assertEqual(ciclo.scans(projeto)[1], ["scan_a", "scan_b"])

    def test_colisao_artefatos_preserva_dados_preexistentes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projeto = self._projeto(root)
            poses_path = root / "poses.json"
            ciclo.salvar_json(poses_path, {g: np.eye(4).tolist() for g in ("scan_a", "scan_b")})
            saida = root / "limpo"
            artefatos = root / "qualidade_3d"
            artefatos.mkdir()
            marcador = artefatos / "dado-existente.txt"
            marcador.write_text("nao apagar", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                qualidade.aplicar(projeto, poses_path, saida)
            self.assertFalse(saida.exists())
            self.assertEqual(marcador.read_text(encoding="utf-8"), "nao apagar")

    def test_html_e_navegador_webgl_com_corte_e_camadas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "navegacao_amostrada.html"
            arrays = {
                "preferidas_mm": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
                "redundantes_candidatas_mm": np.array([[0, 1, 0]], dtype=np.float32),
                "sem_referencia_mm": np.array([[0, 0, 1]], dtype=np.float32),
                "pixels_removidos_mm": np.array([[1, 1, 0]], dtype=np.float32),
            }
            qualidade._escrever_html_navegacao(path, arrays, {"pendencias": 1}, max_pontos=2)
            html = path.read_text(encoding="utf-8")
            self.assertIn("type:'scatter3d'", html)
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("scrollZoom:true", html)
            self.assertIn("aplicarCorte", html)
            self.assertIn("Plotly.restyle", html)
            self.assertIn("Suspeitas sem refer\\u00eancia segura", html)
            self.assertIn("até 2 pontos por camada", html)
            self.assertNotIn("<script src=", html.lower())

    def test_escritor_usa_diretorio_de_artefatos_reservado(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp) / "qualidade_3d"
            pasta.mkdir()
            pontos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            obs = np.zeros(1, dtype=[("p", "<f4", 3)])
            obs["p"] = [[0.0, 0.0, 1.0]]
            preparado = {
                "medicoes": {
                    "scan_a": {
                        "pontos": pontos,
                        "rastreio_ruim": np.array([False, True]),
                        "ruido_mm": np.zeros(2),
                    }
                },
                "observacoes": {"scan_a": obs},
            }
            selecao = {
                "remover": {"scan_a": np.array([False, True])},
                "referencia_scan": {"scan_a": np.array(["scan_b", None], dtype=object)},
            }
            rel = {
                "estado": "revisao_pendente",
                "pendencias": ["sem referencia"],
                "scans": {"scan_a": {"pixels_removidos": 1, "quadros_zerados": 0}},
            }
            qualidade._artefatos(
                pasta,
                preparado,
                selecao,
                {"scan_a": np.array([0])},
                {"scan_a": np.eye(4)},
                rel,
            )
            self.assertTrue((pasta / "navegacao_amostrada.html").is_file())
            self.assertTrue((pasta / "observacoes_3d_integral.npz").is_file())

    def test_revalida_desvio_normal_preserva_pixels_bons_e_detalhe_sem_fonte(self):
        class Indice:
            def __init__(self, points, dist_max):
                self.points = np.asarray(points)
                self.dist_max = dist_max

            def mais_proximo(self, query):
                delta = np.asarray(query)[:, None, :] - self.points[None, :, :]
                dist = np.linalg.norm(delta, axis=2)
                idx = np.argmin(dist, axis=1)
                d = dist[np.arange(len(idx)), idx]
                return d, np.where(d <= self.dist_max, idx, -1)

        dtype = [("p", "<f4", 3), ("v", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")]
        cand = np.zeros(3, dtype=dtype)
        cand["p"] = [[0, 0, 0.1], [0, 0, 0.01], [10, 0, 0]]
        cand["v"] = [[0, 0, 1]] * 3
        ref = np.zeros(1, dtype=dtype)
        ref["p"] = [[0, 0, 0]]
        ref["v"] = [[0, 0, 1]]
        poses = {"bad": np.eye(4), "good": np.eye(4)}
        prepared = {
            "poses": poses,
            "observacoes": {"bad": cand, "good": ref},
            "origem": {
                "bad": {"observacao_global": np.array([0, 2])},
                "good": {"observacao_global": np.array([0])},
            },
            "medicoes": {
                "bad": {
                    "pontos": np.array([[0, 0, 0], [10, 0, 0]], dtype=float),
                    "normais": np.array([[0, 0, 1], [0, 0, 1]], dtype=float),
                    "ruido_mm": np.array([0.2, 0.0]),
                    "detalhe_protegido": np.array([False, True]),
                },
                "good": {
                    "pontos": np.array([[0, 0, 0]], dtype=float),
                    "normais": np.array([[0, 0, 1]], dtype=float),
                    "ruido_mm": np.array([0.01]),
                },
            },
        }
        selection = {
            "remover": {"bad": np.array([True, False]), "good": np.array([False])},
            "referencia_scan": {"bad": ["good", None], "good": [None]},
            "referencia_indice": {"bad": np.array([0, -1]), "good": np.array([-1])},
        }
        with (
            patch("limpar_qualidade_3d.gpu.xp", return_value=np),
            patch("limpar_qualidade_3d.gpu.para_gpu", side_effect=lambda x, dtype=None: np.asarray(x, dtype=dtype)),
            patch("limpar_qualidade_3d.gpu.para_cpu", side_effect=np.asarray),
            patch("limpar_qualidade_3d.gpu.Indice", Indice),
        ):
            remover, _ = qualidade._propor_observacoes(prepared, selection, 0.3, 0.95, 0.2)
        np.testing.assert_array_equal(remover["bad"], [0])
        self.assertEqual(len(remover["good"]), 0)

    def test_ciclo_publica_pendente_e_parecer_nao_pode_apagar_alarme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projeto = self._projeto(root)
            exp = root / "ciclo"
            m = ciclo.iniciar(projeto, exp)
            _, _, rev, m = ciclo.carregar(exp)
            poses = root / "poses.json"
            ciclo.salvar_json(poses, {g: np.eye(4).tolist() for g in m["scans"]})
            hashes = {g: ciclo.sha256(rev / "projeto/data" / g / "cache/frames.dataset") for g in m["scans"]}
            aligned = ciclo._poses_ativas(poses, rev, m)
            pose_hash = hashlib.sha256(
                json.dumps(aligned, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=float).encode()
            ).hexdigest()

            def processo(cmd, **kwargs):
                saida = Path(cmd[cmd.index("--saida") + 1])
                shutil.copytree(rev / "projeto", saida)
                ciclo.salvar_json(
                    saida.parent / "qualidade_3d.json",
                    {
                        "estado": "revisao_pendente",
                        "poses_sha256": pose_hash,
                        "hashes": {g: {"dataset_sha256": hashes[g]} for g in m["scans"]},
                        "scans": {g: {"pixels_removidos": 0} for g in m["scans"]},
                        "pendencias": [{"guid": m["scans"][0], "motivo": "sem referencia segura"}],
                    },
                )
                return subprocess.CompletedProcess(cmd, 0)

            with (
                patch.object(ciclo.subprocess, "run", side_effect=processo),
                patch("auditar_projeto.auditar", return_value={"ok": True}),
            ):
                resultado = ciclo.qualidade(exp, poses)
            _, _, nova, m_nova = ciclo.carregar(exp)
            self.assertEqual(resultado["revisao"], nova.name)
            self.assertTrue(m_nova["qualidade_3d"]["pendencias"])
            parecer = root / "parecer.json"
            ciclo.salvar_json(
                parecer,
                {"manifesto": m_nova["id"], "scans": {g: {"pendencias": []} for g in m_nova["scans"]}},
            )
            with self.assertRaisesRegex(ValueError, "pendencias de qualidade 3D"):
                ciclo.aprovar(exp, parecer)

    def test_quality_sem_pixels_em_todo_scan_retorna_parcial_e_mantem_ativa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projeto = self._projeto(root)
            exp = root / "ciclo"
            ciclo.iniciar(projeto, exp)
            _, _, rev, m = ciclo.carregar(exp)
            before_active = (exp / "ciclo.json").read_bytes()
            before_origin = ciclo.inventario(projeto)
            poses = root / "poses.json"
            ciclo.salvar_json(poses, {g: np.eye(4).tolist() for g in m["scans"]})
            with (
                patch.object(ciclo.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)),
                self.assertRaises(ciclo.ResultadoParcial),
            ):
                ciclo.qualidade(exp, poses)
            self.assertEqual((exp / "ciclo.json").read_bytes(), before_active)
            self.assertEqual(ciclo.inventario(projeto), before_origin)
            self.assertEqual(ciclo.ler(exp / "ciclo.json")["ativa"], rev.name)

    def test_importacao_revo_recebe_seed_nativo_validado_sem_iniciar_revo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projeto = self._projeto(root)
            exp = root / "ciclo"
            m = ciclo.iniciar(projeto, exp)
            _, _, rev, m = ciclo.carregar(exp)
            st = ciclo.estado(rev)
            st["nativo"] = {}
            header = (
                "ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\n"
                "property float y\nproperty float z\nend_header\n"
            )
            for guid in m["scans"]:
                pasta = rev / f"nativo_{guid}"
                pasta.mkdir()
                nuvem = pasta / "nuvem.ply"
                gravar_ply(nuvem, header, np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32))
                ciclo.salvar_json(
                    pasta / "fusao.json",
                    {
                        "entrada": {
                            "dataset_sha256": ciclo.sha256(rev / "projeto/data" / guid / "cache/frames.dataset"),
                            "calibracao": {},
                        },
                        "nuvem_sha256": ciclo.sha256(nuvem),
                        "pontos": 2,
                    },
                )
                st["nativo"][guid] = pasta.name
            ciclo.salvar_json(rev / "estado.json", st)
            original_hashes = {g: ciclo.sha256(rev / "projeto/data" / g / "cache/frames.dataset") for g in m["scans"]}

            def validar_seed(cmd, **kwargs):
                projeto_importado = Path(cmd[2]).parent
                mapa = ciclo.ler(projeto_importado / "identificadores.json")
                saida = Path(cmd[cmd.index("--saida") + 1])
                link = ciclo.ler(saida / "vinculo.json")
                self.assertEqual(link["frames_dataset_sha256"], original_hashes)
                for guid in m["scans"]:
                    clone = projeto_importado / "data" / mapa[guid]
                    self.assertEqual(ciclo.sha256(clone / "cache/frames.dataset"), original_hashes[guid])
                    self.assertEqual(validar_ply(clone / "fuse.ply"), 2)
                    prop = ciclo.ler(clone / "property.rvproj")
                    self.assertEqual(prop["fuse_file"], "fuse.ply")
                    self.assertEqual(prop["point_count"], 2)
                    self.assertFalse(prop["can_fuse_optimize"])
                    self.assertEqual(prop["scan_param"]["scan_mode"], 1)
                return subprocess.CompletedProcess(cmd, 0)

            with (
                patch.object(ciclo.subprocess, "run", side_effect=validar_seed) as run,
                patch.object(ciclo, "registrar_revo", return_value={"estado": "revisao_pendente"}) as registrar,
            ):
                result = ciclo.executar_revo(exp)
            run.assert_called_once()
            registrar.assert_called_once()
            self.assertEqual(result["estado"], "revisao_pendente")


if __name__ == "__main__":
    unittest.main()
