"""Funcoes geometricas basicas."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, ConvexHull


def to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.ascontiguousarray(pts, dtype=np.float64)))


def mean_spacing(pts: np.ndarray, sample: int = 50_000, seed: int = 42) -> float:
    """Mediana da distancia ao vizinho mais proximo (robusta a outliers)."""
    rng = np.random.default_rng(seed)
    tree = cKDTree(pts)
    idx = rng.choice(len(pts), size=min(sample, len(pts)), replace=False)
    d, _ = tree.query(pts[idx], k=2, workers=-1)
    return float(np.median(d[:, 1]))


def make_working_cloud(pts: np.ndarray, max_points: int, seed: int = 42):
    """Reduz por voxel ate ficar abaixo de max_points. Retorna (pts_trabalho, reduziu?)."""
    if len(pts) <= max_points:
        return pts, False
    s = mean_spacing(pts, seed=seed)
    # Superficie: numero de pontos escala com 1/voxel^2
    voxel = s * np.sqrt(len(pts) / max_points) * 1.1
    for _ in range(6):
        down = np.asarray(to_pcd(pts).voxel_down_sample(voxel).points)
        if len(down) <= max_points:
            return down, True
        voxel *= 1.2
    return down, True


def plane_basis(n: np.ndarray):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def hull_area_2d(p2: np.ndarray) -> float:
    if len(p2) < 3:
        return 0.0
    try:
        return float(ConvexHull(p2).volume)  # em 2D, volume = area
    except Exception:
        return 0.0


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    return pts @ T[:3, :3].T + T[:3, 3]
