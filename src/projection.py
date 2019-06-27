import numpy as np
from numba import jit

@jit(nopython = True)
def cross(a, b):
    return np.array((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))

@jit(nopython = True)
def norm(a):
    return np.sqrt(np.sum(a ** 2, axis = 1))

@jit(nopython = True)
def project_point_to_triangle(p_perturb, tri, thickness = 0.0):
    epsilon = 1e-8
    A = tri[0]
    B = tri[1]
    C = tri[2]

    n = cross(B - A, C - A) # normal of triangle
    n = n / np.linalg.norm(n)

    proj_perpendicular = n * np.dot(p_perturb - A, n) # vector from triangle to p_perturb
    proj_perpendicular_norm = np.linalg.norm(proj_perpendicular)

    if np.abs(proj_perpendicular_norm) < epsilon:
        tri_width = np.zeros(3) # perturbation is on the triangle
    else:
        tri_width = thickness * proj_perpendicular / proj_perpendicular_norm # vector describing triangle thickness

    if proj_perpendicular_norm > np.linalg.norm(tri_width):
        p_proj = p_perturb - proj_perpendicular + tri_width # project and offset due to the thickness
        proj_offset = tri_width
    else:
        p_proj = p_perturb # keep perturbation since it is in the thick triangle
        proj_offset = proj_perpendicular

    p_proj_tri = p_perturb - proj_perpendicular # projection onto triangle, ignoring thickness
    A_n = cross(B - A, p_proj_tri - A)
    B_n = cross(C - B, p_proj_tri - B)
    C_n = cross(A - C, p_proj_tri - C)

    if np.dot(n, A_n) < 0.0 or np.dot(n, B_n) < 0.0 or np.dot(n, C_n) < 0.0: # projection not in triangle
        border_planes = ((A, B, A + n), (A, C, A + n), (B, C, B + n))
        border_planes_n = np.vstack((cross(n, A + n - B), cross(n, A + n - C), cross(n, B + n - C)))
        border_planes_n = border_planes_n / norm(border_planes_n).reshape((3, 1))

        border_points = np.empty((3, 3))

        for i in range(3):
            plane = border_planes[i]
            normal = border_planes_n[i]
            center = (plane[0] + plane[1]) / 2.0 # center and radius (half of the length) of an edge
            radius = np.linalg.norm(center - plane[0])
            center = center + proj_offset
            p_plane = p_proj - normal * np.dot(p_proj - plane[0], normal) # project p_proj onto plane

            if np.linalg.norm(p_plane - center) > radius:
                points = np.vstack((plane[0] + proj_offset, plane[1] + proj_offset))
                idx = np.argmin(norm(points - p_plane)) # get closest vertex of triangle
                border_points[i] = points[idx]
            else:
                border_points[i] = p_plane

        # get closest intersection point
        border_dists = norm(p_proj - border_points)
        p_proj = border_points[np.argmin(border_dists)]

    return p_proj

@jit(nopython = True)
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

@jit(nopython = True)
def corner_point(points):
    res = np.full(3, -np.inf)

    for i in range(len(points)):
        diff_idx = np.nonzero(points[i] != res)[0][0]

        if points[i][diff_idx] > res[diff_idx]:
            res = points[i]

    return res
