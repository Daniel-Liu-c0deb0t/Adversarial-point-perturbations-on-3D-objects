import numpy as np
from projection import project_point_to_triangle, bounding_sphere, corner_point
from alpha_shape import alpha_shape_border
from collections import namedtuple

Node = namedtuple("Node", ("center", "radius_lo", "radius_hi", "inside_node", "outside_node"))
Leaf = namedtuple("Leaf", ("center", "triangle"))

# each triangle is represented as a point in the tree
class PerturbProjTree:
    def __init__(self, x, alpha_std = 0.0, thickness = 0.0):
        self.thickness = thickness

        # construct the bounding triangles of the points
        border_points, border_tri = alpha_shape_border(x, alpha_std = alpha_std)
        triangles = []
        tri_center = []
        self.max_radius = 0.0

        for tri in border_tri:
            # get the minimum bounding sphere of each triangle
            tri = border_points[tri]
            center, radius = bounding_sphere(tri)
            triangles.append(tri)
            tri_center.append(center)
            self.max_radius = max(self.max_radius, radius)

        triangles = np.array(triangles)
        tri_center = np.vstack(tri_center)

        self.root = self.build(triangles, tri_center)

    def build(self, curr_triangles, curr_tri_center):
        if len(curr_triangles) == 0:
            return None

        if len(curr_triangles) == 1:
            return Leaf(curr_tri_center[0], curr_triangles[0])

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

        return Node(partition_center, partition_radius_lo, partition_radius_hi, inside_node, outside_node)

    def project(self, x_perturb, perturb):
        distances = np.linalg.norm(perturb, axis = 1)
        x_proj = []
        avg_proj_count = 0.0

        for point, dist in zip(x_perturb, distances):
            if np.isclose(dist, 0.0): # points that are not perturbed are also not projected
                x_proj.append(point)
            else:
                self.projection_count = 0
                # query radius = the perturbation distance
                # + maximum radius of all triangle circumcircles
                # + thickness of each triangle
                nearest_point, nearest_dist = self.query(point, dist + self.max_radius + self.thickness, self.root)
                x_proj.append(nearest_point)
                avg_proj_count += self.projection_count / float(len(x_perturb))

        print("Average points projected:", avg_proj_count)

        return np.vstack(x_proj)

    def query(self, query_point, query_radius, curr_node):
        # project a point onto its nearest triangles and find the nearest projection location
        nearest = (None, float("inf"))

        if type(curr_node) == Leaf:
            if np.linalg.norm(query_point - curr_node.center) <= query_radius:
                # project the point at the leaf node
                proj_point = project_point_to_triangle(query_point, curr_node.triangle, thickness = self.thickness)
                proj_dist = np.linalg.norm(query_point - proj_point)
                nearest = (proj_point, proj_dist)
                self.projection_count += 1
        elif type(curr_node) == Node:
            dist = np.linalg.norm(query_point - curr_node.center)

            if dist > curr_node.radius_lo + query_radius: # query and partition spheres are completely not overlapping
                nearest = self.query(query_point, query_radius, curr_node.outside_node)
            elif dist < curr_node.radius_hi - query_radius: # query and partition spheres are completely overlapping
                nearest = self.query(query_point, query_radius, curr_node.inside_node)
            else:
                # must examine both subtrees as the border of the query sphere overlaps the border of the partition sphere
                nearest_inside = self.query(query_point, query_radius, curr_node.inside_node)
                nearest_outside = self.query(query_point, query_radius, curr_node.outside_node)

                if nearest_inside[1] < nearest_outside[1]:
                    nearest = nearest_inside
                else:
                    nearest = nearest_outside

        return nearest
