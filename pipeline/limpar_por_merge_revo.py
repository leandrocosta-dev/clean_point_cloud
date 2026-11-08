"""Limpa os quadros de um projeto usando um Merge Alignment do Revo como referencia (fiapos, migalhas, apoio).

    python pipeline/avaliar_merge_revo.py <pasta_do_Merge> --saida AVAL
    python pipeline/limpar_por_merge_revo.py <projeto_limpo> AVAL --saida <projeto_novo> [--tol-lixo 0.6 --tol-outros 0.5]

Regiao de lixo de cada scan = pontos da fusao dele sem confirmacao de nenhum outro scan (nap == 0, ver
avaliar_merge_revo) e a > 0,3 mm de superficie confirmada por >= 2 scans. Em cada quadro saem os pixels a
<= --tol-lixo mm dessa regiao que ficam a > --tol-outros mm das fusoes dos outros scans. Pose do quadro ->
merge = pose do scan no merge (poses_final.json). O original nao e tocado; o destino nao pode existir.
Projeto 9: 33b78bab sem confirmacao 8,1% -> 0,4% depois de uma passada.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from recursos import baixa_prioridade

AQUI = os.path.dirname(os.path.abspath(__file__))
import gpu
import revo_projeto as RP
from revo_projeto import ArvoreGPU, RevoScan, gravar_revo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("avaliacao")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--tol-lixo", type=float, default=0.6)
    ap.add_argument("--tol-outros", type=float, default=0.5)
    ap.add_argument("--dist-boa", type=float, default=0.3)
    a = ap.parse_args()
    baixa_prioridade()
    proj, V, out = os.path.abspath(a.projeto), os.path.abspath(a.avaliacao), os.path.abspath(a.saida)
    if os.path.exists(out):
        sys.exit(f"destino ja existe: {out}")
    revo = next(f for f in os.listdir(proj) if f.endswith(".revo"))
    nos = json.load(open(os.path.join(proj, revo), encoding="utf-8"))["nodes"]
    poses = {k: np.array(v) for k, v in json.load(open(os.path.join(V, "poses_final.json"))).items()}
    ids = [n["guid"] for n in nos if n["guid"] in poses]
    F = {g: np.load(os.path.join(V, f"fused_{g[:8]}.npy")) for g in ids}
    NAP = {g: np.load(os.path.join(V, f"nap_{g[:8]}.npy")) for g in ids}
    os.makedirs(os.path.join(out, "data"))
    for n in nos:
        g = n["guid"]
        src = os.path.join(proj, "data", g)
        if g not in poses:
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(out, "data", g))
            print(g[:8], "fora do merge: copiado")
            continue
        lixo, boa = F[g][NAP[g] == 0], F[g][NAP[g] >= 2]
        if len(lixo) and len(boa):
            # fica o lixo a mais de --dist-boa de qualquer ponto confirmado por >= 2 scans (indice na GPU)
            perto_boa = np.isfinite(ArvoreGPU(boa).query(lixo, distance_upper_bound=a.dist_boa)[0])
            lixo = lixo[~perto_boa]
        print(g[:8], f"lixo na fusao: {len(lixo)} ({len(lixo) / len(F[g]) * 100:.2f}%)", flush=True)
        if len(lixo) == 0:
            shutil.copytree(src, os.path.join(out, "data", g))
            continue
        tl, to, T = ArvoreGPU(lixo), ArvoreGPU(np.vstack([F[j] for j in ids if j != g])), poses[g]
        with tempfile.TemporaryDirectory() as tmp:
            scan = RevoScan(src, tmp)
            for m in set(scan.modo.tolist()):
                scan.calibrar(m)

            def mascara(P, seg):
                # GPU: P numpy ou cupy (limpar_scan entrega cupy); sai bool do mesmo tipo
                m = gpu.xp()
                na_gpu = RP._eh_cupy(P)
                A = gpu.para_gpu(P, m.float64) @ gpu.para_gpu(T[:3, :3].T) + gpu.para_gpu(T[:3, 3])
                perto = m.isfinite(tl.query(A, distance_upper_bound=a.tol_lixo)[0])
                # sai o pixel perto do lixo que fica a mais de --tol-outros das fusoes dos outros scans
                ok = ~(perto & ~m.isfinite(to.query(A, distance_upper_bound=a.tol_outros)[0]))
                return ok if na_gpu else gpu.para_cpu(ok)

            RP.mascara_manter = mascara
            log = []
            RP.limpar_scan(
                scan,
                {},
                os.path.join(out, "data", g),
                log=lambda *x: log.append(" ".join(map(str, x))),
                pixels=False,
                ilhas=False,
                min_mantido=0.0,
                sem_ply_antigos=True,
                zerar_vazios=True,
                min_pix_quadro=200,
            )
            scan.c.close()
        print("\n".join(log), flush=True)
    nome = os.path.splitext(revo)[0]
    gravar_revo(os.path.join(proj, revo), out, nome, [n["guid"] for n in nos])
    print("projeto .revo:", os.path.join(out, nome + ".revo"))


if __name__ == "__main__":
    main()
