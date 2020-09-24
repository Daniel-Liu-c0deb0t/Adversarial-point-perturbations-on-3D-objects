import numpy as np
from scipy.spatial import Delaunay
from numba import jit
from projection import cross

def alpha_shape_border(x, alpha_std = 0.0, epsilon = 0.003):
    if epsilon is not None:
        # jiggle the points a little, so less holes form, as the 3D Delaunay
        # triangulation seeks to create tetrahedrons, while we want surface triangles
        x_perturbed = x + epsilon * np.random.random(x.shape) * np.random.choice(np.array([-1, 1]), size = x.shape)
        x_perturbed2 = x + epsilon * np.random.random(x.shape) * np.random.choice(np.array([-1, 1]), size = x.shape)
        x = np.concatenate((x, x_perturbed, x_perturbed2))

    triangles = set() # set of sets of 3D points (tuples)
    DT = Delaunay(x) # 3D Delaunay triangulation

    idx_pointer, indexes = DT.vertex_neighbor_vertices
    min_dists = []

    # compute the alpha value by averaging the distance to nearby points
    # based on the Delaunay triangulation, for simplicity and speed
    # each point should only be connected to a few other points in the triangulation
    for i in range(DT.points.shape[0]):
        for j in range(idx_pointer[i], idx_pointer[i + 1]):
            min_dists.append(np.linalg.norm(DT.points[i] - DT.points[indexes[j]]))

    min_dists = np.array(min_dists)
    avg = np.mean(min_dists)
    std = np.std(min_dists)
    alpha = avg + alpha_std * std

    for simplex_idx in DT.simplices:
        r = circumscribed_radius(DT.points[simplex_idx])

        if r < alpha and r > 0.0:
            # add faces of the tetrahedron to the boundary set, and remove inner faces that are repeated
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

# helper to compute the circumscribed circle's radius of a tetrahedron
@jit(nopython = True)
def circumscribed_radius(simplex):
    epsilon = 1e-8
    a = simplex[0]
    b = simplex[1]
    c = simplex[2]
    d = simplex[3]
    V = np.abs(np.dot(b - a, cross(c - a, d - a))) / 6.0 # volume

    if np.abs(V) < epsilon:
        return 0.0

    dist_a = np.linalg.norm(a - b) * np.linalg.norm(c - d)
    dist_b = np.linalg.norm(a - c) * np.linalg.norm(b - d)
    dist_c = np.linalg.norm(a - d) * np.linalg.norm(b - c)
    return np.sqrt((dist_a + dist_b + dist_c) * (-dist_a + dist_b + dist_c) * (dist_a - dist_b + dist_c) * (dist_a + dist_b - dist_c)) / (V * 24.0)
