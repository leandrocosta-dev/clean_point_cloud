"""Malha (Poisson Surface Reconstruction) a partir de uma nuvem de pontos ja limpa.

Fluxo: carrega PLY -> estima normais (se a nuvem nao tiver) e reorienta de forma
consistente -> Poisson -> corta a malha pelas densidades baixas (remove o "balao"
fantasma do Poisson fora da nuvem real) -> fecha furos pequenos -> salva.
"""
import numpy as np
import open3d as o3d


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
        mesh.remove_vertices_by_mask(densities < threshold)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def fill_holes(mesh: o3d.geometry.TriangleMesh, max_hole_size: float) -> o3d.geometry.TriangleMesh:
    # fill_holes so existe na API tensor (o3d.t.geometry), nao na legacy
    # (usada pela reconstrucao Poisson): precisa converter ida e volta.
    if max_hole_size <= 0:
        return mesh
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    return t_mesh.fill_holes(hole_size=max_hole_size).to_legacy()


def gerar_malha(entrada, saida, depth=9, normal_radius=1.0, normal_max_nn=30,
                density_quantile=0.02, max_hole_size=0.0, log=print):
    log(f"Carregando {entrada}")
    pcd = o3d.io.read_point_cloud(entrada)
    if len(pcd.points) == 0:
        raise ValueError(f"Nuvem vazia ou formato nao suportado: {entrada}")
    log(f"  {len(pcd.points)} pontos")
    had_colors = pcd.has_colors()

    log("Estimando/orientando normais")
    ensure_normals(pcd, normal_radius, normal_max_nn)

    log(f"Poisson reconstruction (depth={depth})")
    mesh = poisson_reconstruct(pcd, depth, density_quantile)
    log(f"  {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangulos (pos-filtro de densidade)")

    if max_hole_size > 0:
        log(f"Fechando furos ate {max_hole_size}")
        mesh = fill_holes(mesh, max_hole_size)

    mesh.compute_vertex_normals()

    # O Poisson sempre cria cor por vertice, zerada (preto) quando a nuvem nao
    # tinha cor. So grava cor se a nuvem ORIGINAL tinha; senao o visualizador
    # sombreia pelas normais.
    if not o3d.io.write_triangle_mesh(saida, mesh, write_vertex_colors=had_colors):
        raise IOError(f"Falha ao salvar {saida}")
    log(f"OK: malha salva em {saida}")
    return mesh
