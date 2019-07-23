import numpy as np
from numba import jit
from projection import project_point_to_triangle

def project_points_to_triangles(x, t):
    triangles = set()

    for tri in t:
        triangles.add(tuple(sorted([tuple(a) for a in tri])))

    triangles = np.array(list(triangles))

    return _project_points_to_triangles(x, triangles)

@jit(nopython = True)
def _project_points_to_triangles(x, t):
    x_proj = np.empty((len(x), 3))

    for i in range(len(x)):
        min_dist = np.inf

        for j in range(len(t)):
            p = project_point_to_triangle(x[i], t[j])
            dist = np.linalg.norm(x[i] - p)

            if dist < min_dist:
                x_proj[i] = p
                min_dist = dist

    return x_proj
