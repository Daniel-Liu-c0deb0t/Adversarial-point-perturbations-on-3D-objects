import numpy as np

def project_point_to_triangle(p_perturb, tri):
    p = np.average(tri, axis = 0) # get centroid

    if np.all(np.isclose(p, p_perturb)): # no projection if perturbation is close to centroid
        return p_perturb

    A, B, C = tri

    n = np.cross(B - A, C - A) # find normal vector
    n = n / np.linalg.norm(n) # normalize

    p_proj = p_perturb - n * np.dot(p_perturb - A, n) # project point onto the triangle's plane

    # next, the projection is clipped to be within the triangle

    # create the planes that represent the borders of the triangle, which are perpendicular to the triangle
    # border planes are used to bypass floating point calculation issues
    border_planes = [(A, B, A + n), (A, C, A + n), (B, C, B + n)]
    border_planes_n = np.vstack((np.cross(n, A + n - B), np.cross(n, A + n - C), np.cross(n, B + n - C)))
    border_planes_n = border_planes_n / np.linalg.norm(border_planes_n, axis = 1, keepdims = True)

    intersection_points = []
    p_to_p_proj = p_proj - p

    for plane, plane_n in zip(border_planes, border_planes_n):
        plane_p = plane[0]
        direction = np.dot(p_to_p_proj, plane_n)

        if not np.isclose(direction, 0.0):
            # otherwise, the plane and perturbation are parallel, so no intersection
            d = -np.dot(p - plane_p, plane_n) / direction

            if 0.0 < d <= 1.0:
                intersection_points.append(p + d * p_to_p_proj)

    if not intersection_points:
        return p_proj

    # get closest intersection point
    intersection_points = np.vstack(intersection_points)
    intersection_dists = np.linalg.norm(intersection_points - p[np.newaxis, :], axis = 1)
    p_clip = intersection_points[np.argmin(intersection_dists)]

    return p_clip
