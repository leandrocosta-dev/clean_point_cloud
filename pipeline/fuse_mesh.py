"""
Fusao de malha (Poisson Surface Reconstruction) a partir de uma nuvem de
pontos ja limpa (saida de Clean-PointCloud.ps1 / Fix-RotationalDrift.ps1).

Fluxo: carrega PLY -> estima normais (se a nuvem nao tiver, ou para
reorienta-las de forma consistente) -> Poisson -> corta a malha pelas
densidades baixas (remove "balao" fantasma do Poisson fora da nuvem real) ->
fechamento de furos pequenos -> salva malha final.
"""

import argparse
import sys

import numpy as np
import open3d as o3d


def load_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Nuvem vazia ou formato nao suportado: {path}")
    return pcd


def ensure_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int) -> None:
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
    pcd.orient_normals_consistent_tangent_plane(k=max_nn)


def poisson_reconstruct(pcd: o3d.geometry.PointCloud, depth: int, density_quantile: float):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)

    if density_quantile > 0:
        threshold = np.quantile(densities, density_quantile)
        low_density_vertices = densities < threshold
        mesh.remove_vertices_by_mask(low_density_vertices)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def fill_holes(mesh: o3d.geometry.TriangleMesh, max_hole_size: float) -> o3d.geometry.TriangleMesh:
    # fill_holes so existe na API tensor (o3d.t.geometry), nao na legacy
    # (usada pela reconstrucao Poisson) -- precisa converter ida e volta.
    if max_hole_size <= 0:
        return mesh
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    t_mesh = t_mesh.fill_holes(hole_size=max_hole_size)
    return t_mesh.to_legacy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fusao de malha (Poisson) a partir de nuvem limpa")
    parser.add_argument("--input", required=True, help="PLY de entrada (nuvem de pontos limpa)")
    parser.add_argument("--output", required=True, help="Caminho da malha de saida (.ply ou .obj ou .stl)")
    parser.add_argument("--depth", type=int, default=9, help="Profundidade do octree do Poisson (8-10 tipico)")
    parser.add_argument("--normal-radius", type=float, default=1.0, help="Raio de busca p/ estimar normais")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Max vizinhos p/ estimar/orientar normais")
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.02,
        help="Quantil de densidade abaixo do qual vertices sao removidos (0 desativa)",
    )
    parser.add_argument(
        "--max-hole-size",
        type=float,
        default=0.0,
        help="Tamanho maximo de furo a fechar (0 desativa; em unidades do modelo)",
    )
    args = parser.parse_args()

    print(f"Carregando {args.input}")
    pcd = load_cloud(args.input)
    print(f"  {len(pcd.points)} pontos")
    had_colors = pcd.has_colors()

    print("Estimando/orientando normais")
    ensure_normals(pcd, args.normal_radius, args.normal_max_nn)

    print(f"Poisson reconstruction (depth={args.depth})")
    mesh = poisson_reconstruct(pcd, args.depth, args.density_quantile)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos (pos-filtro de densidade)")

    if args.max_hole_size > 0:
        print(f"Fechando furos ate {args.max_hole_size}")
        mesh = fill_holes(mesh, args.max_hole_size)

    mesh.compute_vertex_normals()

    # A reconstrucao Poisson sempre cria um buffer de cor por vertice (mesh
    # has_vertex_colors() == True), mesmo quando a nuvem de entrada nao tinha
    # cor -- nesse caso ele vem zerado (preto). So escreve cor se a nuvem
    # ORIGINAL de fato tinha cor; senao deixa o visualizador sombrear pelas
    # normais.
    ok = o3d.io.write_triangle_mesh(args.output, mesh, write_vertex_colors=had_colors)
    if not ok:
        print(f"Falha ao salvar {args.output}", file=sys.stderr)
        return 1

    print(f"OK: malha salva em {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
