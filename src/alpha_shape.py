import numpy as np
from scipy.spatial import Delaunay

def alpha_shape(x, alpha_std = 0.0):
    triangles = set() # set of sets of 3D points (tuples)
    DT = Delaunay(x) # 3D Delaunay triangulation

    idx_pointer, indexes = DT.vertex_neighbor_vertices
    min_dists = []

    for i in range(DT.points.shape[0]):
        for j in range(idx_pointer[i], idx_pointer[i + 1]):
            min_dists.append(np.linalg.norm(DT.points[i] - DT.points[indexes[j]]))

    min_dists = np.array(min_dists)
    avg = np.mean(min_dists)
    std = np.std(min_dists)
    alpha = avg + alpha_std * std

    for simplex_idx in DT.simplices:
        r = circumscribed_radius(DT.points[simplex_idx])

        if r < alpha:
            tri_a = frozenset({simplex_idx[0], simplex_idx[1], simplex_idx[2]})
            tri_b = frozenset({simplex_idx[0], simplex_idx[1], simplex_idx[3]})
            tri_c = frozenset({simplex_idx[0], simplex_idx[2], simplex_idx[3]})
            tri_d = frozenset({simplex_idx[1], simplex_idx[2], simplex_idx[3]})

            for tri in (tri_a, tri_b, tri_c, tri_d):
                if tri in triangles:
                    triangles.remove(tri)
                else:
                    triangles.add(tri)

    res = []

    for tri in triangles:
        res.append(list(tri))

    return DT.points, np.array(res)

def circumscribed_radius(simplex):
    a, b, c, d = simplex
    V = np.linalg.norm(np.dot(b - a, np.cross(c - a, d - a))) / 6.0 # volume
    dist_a = np.linalg.norm(a - b) * np.linalg.norm(c - d)
    dist_b = np.linalg.norm(a - c) * np.linalg.norm(b - d)
    dist_c = np.linalg.norm(a - d) * np.linalg.norm(b - c)
    return np.sqrt((dist_a + dist_b + dist_c) * (-dist_a + dist_b + dist_c) * (dist_a - dist_b + dist_c) * (dist_a + dist_b - dist_c)) / (V * 24.0)
