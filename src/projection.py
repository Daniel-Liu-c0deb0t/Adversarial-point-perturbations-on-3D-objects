import numpy as np
from numba import jit

@jit(nopython = True, cache = True)
def cross(a, b):
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))

@jit(nopython = True, cache = True)
def norm(a):
    return np.sqrt(np.sum(a ** 2, axis = 1))

@jit(nopython = True, cache = True)
def project_point_to_triangle(p_perturb, tri, thickness = 0.0):
    epsilon = 1e-8

    p = np.sum(tri, axis = 0) / 3.0 # get centroid

    if np.all(np.abs(p - p_perturb) < epsilon): # no projection if perturbation is close to centroid
        return p_perturb

    A = tri[0]
    B = tri[1]
    C = tri[2]

    n = cross(B - A, C - A) # find normal vector
    n = n / np.linalg.norm(n) # normalize

    # vector perpendicular to the triangle's plane, from plane to p_perturb
    proj_perpendicular = n * np.dot(p_perturb - A, n)
    proj_perpendicular_norm = np.linalg.norm(proj_perpendicular)
    # vector that describes the thickness of each triangle
    if np.abs(proj_perpendicular_norm) < epsilon:
        tri_width = np.zeros(3)
    else:
        tri_width = thickness * proj_perpendicular / proj_perpendicular_norm

    if proj_perpendicular_norm > np.linalg.norm(tri_width):
        p_proj = p_perturb - proj_perpendicular + tri_width # project and offset due to the thickness
        proj_offset = tri_width
    else:
        p_proj = p_perturb # keep perturbation since it is in the thick triangle
        proj_offset = proj_perpendicular

    # next, the projection is clipped to be within the triangle

    p = p + proj_offset # ensure that the centroid and projected perturbation are on the same plane

    if np.all(np.abs(p - p_proj) < epsilon): # no border intersection if projection is on the centroid
        return p_proj

    # create the planes that represent the borders of the triangle, which are perpendicular to the triangle
    # border planes are used to bypass floating point calculation issues
    border_planes = ((A, B, A + n), (A, C, A + n), (B, C, B + n))
    border_planes_n = np.vstack((cross(n, A + n - B), cross(n, A + n - C), cross(n, B + n - C)))
    border_planes_n = border_planes_n / norm(border_planes_n).reshape((3, 1))

    at_least_one = False
    intersection_points = np.full((3, 3), np.inf)
    p_to_p_proj = p_proj - p

    for i in range(3):
        plane_n = border_planes_n[i]
        plane_p = border_planes[i][0]
        distance = np.dot(p_to_p_proj, plane_n)

        if np.abs(distance) >= epsilon:
            # otherwise, the plane and perturbation are parallel, so no intersection
            d = -np.dot(p - plane_p, plane_n) / distance

            if 0.0 < d <= 1.0:
                intersection_points[i] = p + d * p_to_p_proj
                at_least_one = True

    if not at_least_one:
        return p_proj

    # get closest intersection point
    intersection_dists = norm(intersection_points - p)
    p_clip = intersection_points[np.argmin(intersection_dists)]

    return p_clip

@jit(nopython = True, cache = True)
def bounding_sphere(tri):
    # minimum bounding sphere of 3D triangle
    A = tri[0]
    B = tri[1]
    C = tri[2]
    A_to_B = B - A
    A_to_C = C - A
    B_to_C = C - B

    if np.dot(A_to_B, A_to_C) <= 0.0 and np.dot(A_to_B, B_to_C) <= 0.0 and np.dot(A_to_C, B_to_C) <= 0.0:
        # right or obtuse triangle
        edges = np.array((np.linalg.norm(A_to_B), np.linalg.norm(A_to_C), np.linalg.norm(B_to_C)))
        idx = np.argmax(edges)
        radius = edges[idx] / 2.0
        a = np.vstack((A, B, A, C, B, C))
        b = a[idx * 2:idx * 2 + 1]
        center = np.sum(b, axis = 0) / 2.0
    else:
        # acute triangle
        normal = cross(A_to_B, A_to_C)
        # get the center of the bounding sphere
        center = A + (np.sum(A_to_B ** 2) * cross(A_to_C, normal) + np.sum(A_to_C ** 2) * cross(normal, A_to_B)) / (np.sum(normal ** 2) * 2.0)
        # get the radius of the bounding sphere
        radius = np.max(norm(tri - center))

    return center, radius

@jit(nopython = True, cache = True)
def corner_point(points):
    res = np.full(3, -np.inf)

    for i in range(len(points)):
        diff_idx = np.nonzero(points[i] != res)[0][0]

        if points[i][diff_idx] > res[diff_idx]:
            res = points[i]

    return res
