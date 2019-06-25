import numpy as np
from numba import jit
from projection import cross, norm

@jit(nopython = True)
def farthest_point_sampling(triangles, initial_points, num_points, kappa = 3):
    sampled_points = sample_points(triangles, kappa * num_points)
    curr_points = []
    dists = np.full(len(sampled_points), np.inf)

    for i in range(len(initial_points)):
        dists = np.minimum(dists, norm(sampled_points - initial_points[i].reshape((1, -1))))

    for i in range(num_points):
        curr_points.append(sampled_points[np.argmax(dists)])
        dists = np.minimum(dists, norm(sampled_points - curr_points[-1].reshape((1, -1))))

    return np.array(curr_points)

@jit(nopython = True)
def sample_points(triangles, num_points):
    points = []
    prefix_areas = []
    areas = []

    for i in range(len(triangles)):
        area = np.linalg.norm(cross(triangles[i][2] - triangles[i][0], triangles[i][1] - triangles[i][0])) / 2.0
        areas.append(area)

        if i == 0:
            prefix_areas.append(area)
        else:
            prefix_areas.append(prefix_areas[i - 1] + area)

    total_area = prefix_areas[-1]

    for i in range(num_points):
        rand = np.random.uniform(high = total_area)
        idx = binary_search(prefix_areas, rand)

        a = triangles[idx][0]
        b = triangles[idx][1]
        c = triangles[idx][2]

        r1 = np.random.random()
        r2 = np.random.random()

        if r1 + r2 >= 1.0:
            r1 = 1 - r1
            r2 = 1 - r2

        point = a + r1 * (c - a) + r2 * (b - a)
        points.append(point)

    return np.array(points)

@jit(nopython = True)
def binary_search(a, b):
