import numpy as np

def project_point_to_triangle(p_perturb, tri, thickness = 0.0):
    p = np.mean(tri, axis = 0) # get centroid

    if np.all(np.isclose(p, p_perturb)): # no projection if perturbation is close to centroid
        return p_perturb

    A, B, C = tri

    n = np.cross(B - A, C - A) # find normal vector
    n = n / np.linalg.norm(n) # normalize

    # vector perpendicular to the triangle's plane, from plane to p_perturb
    proj_perpendicular = n * np.dot(p_perturb - A, n)
    proj_perpendicular_norm = np.linalg.norm(proj_perpendicular)
    # vector that describes the thickness of each triangle
    if np.isclose(proj_perpendicular_norm, 0.0):
        tri_width = 0.0
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

    if np.all(np.isclose(p, p_proj)): # no border intersection if projection is on the centroid
        return p_proj

    # create the planes that represent the borders of the triangle, which are perpendicular to the triangle
    # border planes are used to bypass floating point calculation issues
    border_planes = [(A, B, A + n), (A, C, A + n), (B, C, B + n)]
    border_planes_n = np.vstack((np.cross(n, A + n - B), np.cross(n, A + n - C), np.cross(n, B + n - C)))
    border_planes_n = border_planes_n / np.linalg.norm(border_planes_n, axis = 1, keepdims = True)

    intersection_points = []
    p_to_p_proj = p_proj - p

    for plane, plane_n in zip(border_planes, border_planes_n):
        plane_p = plane[0]
        distance = np.dot(p_to_p_proj, plane_n)

        if not np.isclose(distance, 0.0):
            # otherwise, the plane and perturbation are parallel, so no intersection
            d = -np.dot(p - plane_p, plane_n) / distance

            if 0.0 < d <= 1.0:
                intersection_points.append(p + d * p_to_p_proj)

    if not intersection_points:
        return p_proj

    # get closest intersection point
    intersection_points = np.vstack(intersection_points)
    intersection_dists = np.linalg.norm(intersection_points - p[np.newaxis, :], axis = 1)
    p_clip = intersection_points[np.argmin(intersection_dists)]

    return p_clip
