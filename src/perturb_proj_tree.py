import numpy as np
from numba import jit
from projection import project_point_to_triangle, bounding_sphere, corner_point
from alpha_shape import alpha_shape_border

@jit(nopython = True)
def _query(query_point, query_radius, curr_node, thickness, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf):
    # project a point onto its nearest triangles and find the nearest projection location
    nearest = (None, np.inf)

    if is_leaf[curr_node]:
        if np.linalg.norm(query_point - center[curr_node]) <= query_radius:
            # project the point at the leaf node
            proj_point = project_point_to_triangle(query_point, triangle[curr_node], thickness = thickness)
            proj_dist = np.linalg.norm(query_point - proj_point)
            nearest = (proj_point, proj_dist)
    else:
        dist = np.linalg.norm(query_point - center[curr_node])

        if dist > radius_lo[curr_node] + query_radius: # query and partition spheres are completely not overlapping
            nearest = _query(query_point, query_radius, outside_node[curr_node], thickness, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf)
        elif dist < radius_hi[curr_node] - query_radius: # query and partition spheres are completely overlapping
            nearest = _query(query_point, query_radius, inside_node[curr_node], thickness, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf)
        else:
            # must examine both subtrees as the border of the query sphere overlaps the border of the partition sphere
            nearest_inside = _query(query_point, query_radius, inside_node[curr_node], thickness, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf)
            nearest_outside = _query(query_point, query_radius, outside_node[curr_node], thickness, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf)

            if nearest_inside[1] < nearest_outside[1]:
                nearest = nearest_inside
            else:
                nearest = nearest_outside

    return nearest

@jit(nopython = True)
def _project(x_perturb, perturb, max_radius, thickness, root, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf):
    epsilon = 1e-8
    distances = np.sqrt(np.sum(perturb ** 2, axis = 1))
    x_proj = []

    for i in range(len(x_perturb)):
        if np.abs(distances[i]) < epsilon: # points that are not perturbed are also not projected
            x_proj.append(x_perturb[i])
        else:
            # query radius = the perturbation distance
            # + maximum radius of all triangle circumcircles
            # + thickness of each triangle
            nearest_point, nearest_dist = _query(x_perturb[i], distances[i] + max_radius + thickness, root, thickness, center, radius_lo, radius_hi, inside_node, outside_node, triangle, is_leaf)

            if nearest_point is None:
                x_proj.append(x_perturb[i] - perturb[i])
            else:
                x_proj.append(nearest_point)

    return x_proj

@jit(nopython = True)
def _calc_tri_center(border_points, border_tri):
    triangles = []
    tri_center = []
    max_radius = 0.0

    for i in range(len(border_tri)):
        # get the minimum bounding sphere of each triangle
        tri = border_points[border_tri[i]]
        center, radius = bounding_sphere(tri)
        triangles.append(tri)
        tri_center.append(center)
        max_radius = max(max_radius, radius)

    return triangles, tri_center, max_radius

# each triangle is represented as a point in the tree
class PerturbProjTree:
    def __init__(self, x, alpha_std = 0.0, thickness = 0.0):
        self.thickness = thickness

        # construct the bounding triangles of the points
        border_points, border_tri = alpha_shape_border(x, alpha_std = alpha_std)
        triangles, tri_center, self.max_radius = _calc_tri_center(border_points, border_tri)
        triangles = np.array(triangles)
        tri_center = np.vstack(tri_center)

        self.center = np.empty((len(triangles) * 2, 3))
        self.radius_lo = np.empty(len(triangles) * 2)
        self.radius_hi = np.empty(len(triangles) * 2)
        self.inside_node = np.empty(len(triangles) * 2, dtype = int)
        self.outside_node = np.empty(len(triangles) * 2, dtype = int)
        self.triangle = np.empty((len(triangles) * 2, 3, 3))
        self.is_leaf = np.empty(len(triangles) * 2, dtype = bool)
        self.curr_idx = 0

        self.root = self.build(triangles, tri_center)

    def build(self, curr_triangles, curr_tri_center):
        if len(curr_triangles) == 0:
            print("Bad stuff happened when partitioning!!!")
            return None

        if len(curr_triangles) == 1:
            self.center[self.curr_idx] = curr_tri_center[0]
            self.triangle[self.curr_idx] = curr_triangles[0]
            self.is_leaf[self.curr_idx] = True
            self.curr_idx += 1
            return self.curr_idx - 1

        # pick corner point to partition with
        partition_center = corner_point(curr_tri_center)

        # get distances from each triangle's point to the partition point
        distances = np.linalg.norm(curr_tri_center - partition_center[np.newaxis, :], axis = 1)

        # pick the middle point to for the partition radius
        lo = len(distances) // 2
        hi = lo - 1
        # sort by negative distances so all triangle points with the same distance
        # as the picked mid distance will be to the right in the partition array
        partition = np.argpartition(-distances, (hi, lo))
        partition_radius_lo = distances[partition[lo]]
        partition_radius_hi = distances[partition[hi]]

        inside_idx = partition[lo:]
        outside_idx = partition[:lo]

        inside_node = self.build(curr_triangles[inside_idx], curr_tri_center[inside_idx])
        outside_node = self.build(curr_triangles[outside_idx], curr_tri_center[outside_idx])

        self.center[self.curr_idx] = partition_center
        self.radius_lo[self.curr_idx] = partition_radius_lo
        self.radius_hi[self.curr_idx] = partition_radius_hi
        self.inside_node[self.curr_idx] = inside_node
        self.outside_node[self.curr_idx] = outside_node
        self.is_leaf[self.curr_idx] = False
        self.curr_idx += 1

        return self.curr_idx - 1

    def project(self, x_perturb, perturb):
        res = _project(x_perturb, perturb, self.max_radius, self.thickness, self.root, self.center, self.radius_lo, self.radius_hi, self.inside_node, self.outside_node, self.triangle, self.is_leaf)
        return np.vstack(res)
