"""Leitor do formato antigo do Revo Scan (cache/frame_NNN_NNNN.dph + .inf, sem frames.dataset).

.dph = profundidade uint16 800x600 (x depth_scale 0,1 = mm); .inf = 16 bytes de cabecalho +
doubles: pose 4x4 (3 primeiras linhas, camera->mundo, com eixos da camera (x,-y,-z) vistos da imagem) + restante.
Intrinsecos em param/Q.bin (float32 4x4: -cx, -cy, f).
"""
import glob
import os
import numpy as np

H, W = 600, 800


def intrinsecos(pasta):
    """(f, cx, cy, largura_de_referencia, escala_mm). Q.bin vale na largura do IR (ir_width: 800 no scanner A23C..., 640 no A22C...);
    a profundidade pode vir em metade dessa largura (400x300). escala = depth_scale do property (0,1 ou 0,005 mm por unidade)."""
    import json
    sp = json.load(open(os.path.join(pasta, "property.rvproj")))["scan_param"]
    qf = os.path.join(pasta, "param", "Q.bin")
    if not os.path.exists(qf):  # scan sem param (copia incompleta): usa o de um irmao do mesmo scanner (Q identico nos que tem)
        for irm in sorted(glob.glob(os.path.join(os.path.dirname(pasta), "*", "param", "Q.bin"))):
            sj = json.load(open(os.path.join(os.path.dirname(os.path.dirname(irm)), "property.rvproj")))["scan_param"]
            if sj.get("scanner_sn") == sp.get("scanner_sn"):
                qf = irm
                break
    Q = np.fromfile(qf, np.float32).reshape(4, 4)
    return float(Q[2, 3]), float(-Q[0, 3]), float(-Q[1, 3]), float(sp.get("ir_width", 800)), float(sp.get("depth_scale", 0.1))


def listar(pasta):
    return sorted(glob.glob(os.path.join(pasta, "cache", "frame_*.dph")))


def pose(dph):
    b = open(dph[:-4] + ".inf", "rb").read()
    T = np.eye(4)
    T[:3, :] = np.frombuffer(b[16:16 + 96], np.float64).reshape(3, 4)
    return T


def profundidade(dph):
    """Resolucao inferida do tamanho: 800x600 (960000 B) ou 400x300 (240000 B, scans 'accuracy_type 0')."""
    n = os.path.getsize(dph) // 2
    h, w = {H * W: (H, W), (H // 2) * (W // 2): (H // 2, W // 2), 400 * 640: (400, 640)}[n]
    return np.fromfile(dph, np.uint16).reshape(h, w)


def pontos(d, K, escala=None, passo=1):
    f, cx, cy, W0, esc = K  # intrinsecos valem na largura W0; se a profundidade tem metade da largura, tudo cai pela metade
    escala = esc if escala is None else escala
    k = d.shape[1] / W0
    f, cx, cy = f * k, cx * k, cy * k
    v, u = np.nonzero(d[::passo, ::passo])
    v *= passo
    u *= passo
    z = d[v, u].astype(np.float32) * escala
    # convencao verificada contra fuse.ply do Revo (mediana 0,2 mm) e por coerencia entre quadros no A22C...: camera = (x, -y, -z)
    return np.stack([(u - cx) * z / f, -(v - cy) * z / f, -z], 1), v, u


def poses_registradas(pasta):
    """cache/global_register_pose.pose: uint32 n, depois n x (4 bytes + int32 numero do quadro, 4x4 float64)."""
    b = open(os.path.join(pasta, "cache", "global_register_pose.pose"), "rb").read()
    n = int(np.frombuffer(b[:4], np.uint32)[0])
    out = {}
    for i in range(n):
        o = 4 + i * 136
        k = int(np.frombuffer(b[o + 4:o + 8], np.int32)[0])
        out[k] = np.frombuffer(b[o + 8:o + 136], np.float64).reshape(4, 4).copy()
    return out


def pose_do_quadro(f, reg, K, arvore_fuse=None, passo=4):
    """Pose camera->mundo de um quadro: a registrada (global_register_pose) ou a da .inf.
    Se ha fuse do Revo (arvore KD), escolhe a que mais concorda com ele. Achado em 183125: nos 15 primeiros quadros a
    registrada fica ~5 mm fora do fuse e a .inf bate 100%; nos demais a registrada bate."""
    Ti = pose(f)
    Tr = reg.get(int(f[-8:-4])) if reg else None
    if Tr is None:
        return Ti, "inf"
    if arvore_fuse is None:
        return Tr, "reg"
    p, _, _ = pontos(profundidade(f), K, passo=passo)
    if not len(p):
        return Tr, "reg"
    mi = np.median(arvore_fuse.query(p @ Ti[:3, :3].T + Ti[:3, 3])[0])
    mr = np.median(arvore_fuse.query(p @ Tr[:3, :3].T + Tr[:3, 3])[0])
    return (Tr, "reg") if mr <= mi else (Ti, "inf")
