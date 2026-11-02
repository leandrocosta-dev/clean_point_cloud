"""Cria um projeto novo na BIBLIOTECA do Revo Scan 6 a partir de um projeto de merge existente,
trocando a nuvem e a malha por arquivos .ply proprios (so cria pasta nova; nao altera o modelo base).

    python pipeline/criar_projeto_biblioteca.py <pasta_base_da_biblioteca> --nuvem n.ply --malha m.ply --nome Merge_suave
"""
import argparse, json, os, shutil, uuid
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("--nuvem", required=True); ap.add_argument("--malha", required=True)
ap.add_argument("--nome", required=True)
a = ap.parse_args()
raiz = os.path.dirname(os.path.abspath(a.base))
novo = os.path.join(raiz, str(uuid.uuid4()))
shutil.copytree(a.base, novo, ignore=shutil.ignore_patterns("*.revox", "pc_merge.ply", "meshs", "frames.dataset-*"))
os.makedirs(os.path.join(novo, "meshs"))
base_revox = [f for f in os.listdir(a.base) if f.endswith(".revox")][0]
d = json.load(open(os.path.join(a.base, base_revox), encoding="utf-8"))
pc = o3d.io.read_point_cloud(a.nuvem)
o3d.io.write_point_cloud(os.path.join(novo, "pointclouds", "pc_merge.ply"), pc)       # ply binario
m = o3d.io.read_triangle_mesh(a.malha); m.compute_vertex_normals()
nome_malha = a.nome + " (malha full)"
o3d.io.write_triangle_mesh(os.path.join(novo, "meshs", nome_malha + ".ply"), m, write_vertex_normals=True)
for me in d["meshes"]:
    me.update(file="meshs/" + nome_malha + ".ply", name=nome_malha, guid=str(uuid.uuid4()),
              face_count=len(m.triangles), vertex_count=len(m.vertices))
d["point_clouds"][0]["guid"] = d["meshes"][0]["sourceId"] = str(uuid.uuid4())
d["thumbnail"] = os.path.join(novo, "preview.png")
d["create_time"] = d["edit_time"] = str(int(__import__("time").time()))
tam = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(novo) for f in fs)
d["size"] = str(tam)
json.dump(d, open(os.path.join(novo, a.nome + ".revox"), "w", encoding="utf-8"), indent=4)
print(novo)
