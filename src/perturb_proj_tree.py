import numpy as np
from projection import project_point_to_triangle
from alpha_shape import alpha_shape_border
from collections import namedtuple

Node = namedtuple("Node", ("center", "radius", "inside_node", "outside_node"))
Leaf = namedtuple("Leaf", ("bucket"))

class PerturbProjTree:
    def __init__(self, x, alpha_std = 0.0, thickness = 0.0):
        self.thickness = thickness

        border_points, border_tri = alpha_shape_border(x, alpha_std = alpha_std)
        triangles = []
        tri_center = []
        tri_radius = []

        for tri in border_tri:
            tri = border_points[tri]
            center, radius = bounding_sphere(tri)
            triangles.append(tri)
            tri_center.append(center)
            tri_radius.append(radius + thickness)

        triangles = np.vstack(triangles)
        tri_center = np.vstack(tri_center)
        tri_radius = np.vstack(tri_radius)

        self.root = self.build(triangles, tri_center, tri_radius)

    def build(self, curr_triangles, curr_tri_center, curr_tri_radius):
        if not curr_triangles:
            return None

        if len(curr_triangles) == 1:
            return Leaf(curr_triangles)

        partition_center = curr_tri_center[np.random.randint(len(curr_tri_center))]

        distances_center = np.linalg.norm(curr_tri_center - partition_center[np.newaxis, :], axis = 1)
        distances = distances_center + curr_tri_radius

        mid = len(distances) // 2
        partition = np.argpartition(-distances, mid)
        partition_radius = distances[partition[mid]]

        inside_idx = partition[mid:]
        outside_idx = partition[:mid]
        both_inside_outside = np.nonzero(distances_center[outside_idx] - curr_tri_radius[outside_idx] <= partition_radius)
        inside_idx = np.concatenate((inside_idx, both_inside_outside))

        if len(inside_idx) == len(curr_triangles):
            return Leaf(curr_triangles)

        inside_node = self.build(curr_triangles[inside_idx], curr_tri_center[inside_idx], curr_tri_radius[inside_idx])
        outside_node = self.build(curr_triangles[outside_idx], curr_tri_center[outside_idx], curr_tri_radius[outside_idx])

        return Node(partition_center, partition_radius, inside_node, outside_node)
