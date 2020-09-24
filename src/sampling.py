import numpy as np
from numba import jit
from projection import cross, norm

@jit(nopython = True)
def _init_seed():
    np.random.seed(1234)

_init_seed()

@jit(nopython = True)
def binary_search(a, b):
    lo = 0
    hi = len(a) - 1

    while lo < hi:
        m = (lo + hi - 1) // 2

        if a[m] < b:
            lo = m + 1
        else:
            hi = m

    if a[hi] == b:
        return min(hi + 1, len(a) - 1)
    else:
        return hi

@jit(nopython = True)
def sample_points(triangles, num_points):
    prefix_areas = []

    for i in range(len(triangles)):
        area = np.linalg.norm(cross(triangles[i][2] - triangles[i][0], triangles[i][1] - triangles[i][0])) / 2.0

        if i == 0:
            prefix_areas.append(area)
        else:
            prefix_areas.append(prefix_areas[i - 1] + area)

    prefix_areas = np.array(prefix_areas)
    total_area = prefix_areas[-1]
    points = np.empty((num_points, 3))

    for i in range(num_points):
        rand = np.random.uniform(0.0, total_area)
        idx = binary_search(prefix_areas, rand)

        a = triangles[idx][0]
        b = triangles[idx][1]
        c = triangles[idx][2]

        r1 = np.random.random()
        r2 = np.random.random()

        if r1 + r2 >= 1.0:
            r1 = 1.0 - r1
            r2 = 1.0 - r2

        point = a + r1 * (c - a) + r2 * (b - a)
        points[i] = point

    return points

@jit(nopython = True)
def farthest_point(sampled_points, initial_points, num_points):
    curr_points = np.empty((num_points, 3))
    dists = np.full(len(sampled_points), np.inf)

    if initial_points is None:
        dists = np.minimum(dists, norm(sampled_points - sampled_points[0].reshape((1, -1))))
        curr_points[0] = sampled_points[0]
        start_idx = 1
    else:
        for i in range(len(initial_points)):
            dists = np.minimum(dists, norm(sampled_points - initial_points[i].reshape((1, -1))))

        start_idx = 0

    for i in range(start_idx, num_points):
        curr_points[i] = sampled_points[np.argmax(dists)]
        dists = np.minimum(dists, norm(sampled_points - curr_points[i].reshape((1, -1))))

    return curr_points

@jit(nopython = True)
def farthest_point_idx(sampled_points, initial_points, num_points):
    curr_points = np.empty(num_points, dtype = np.int64)
    dists = np.full(len(sampled_points), np.inf)

    if initial_points is None:
        dists = np.minimum(dists, norm(sampled_points - sampled_points[0].reshape((1, -1))))
        curr_points[0] = 0
        start_idx = 1
    else:
        for i in range(len(initial_points)):
            dists = np.minimum(dists, norm(sampled_points - initial_points[i].reshape((1, -1))))

        start_idx = 0

    for i in range(start_idx, num_points):
        curr_points[i] = np.argmax(dists)
        dists = np.minimum(dists, norm(sampled_points - sampled_points[curr_points[i]].reshape((1, -1))))

    return curr_points

@jit(nopython = True)
def farthest_point_sampling(triangles, initial_points, num_points, kappa):
    sampled_points = sample_points(triangles, kappa * num_points)
    return farthest_point(sampled_points, initial_points, num_points)

@jit(nopython = True)
def gaussian_rbf(norm, shape):
    return np.exp(-((shape * norm) ** 2))

@jit(nopython = True)
def radial_basis(sampled_points, initial_points, num_points, shape):
    probs = []
    total_prob = 0.0
    curr_points = np.empty((num_points, 3))

    for i in range(len(sampled_points)):
        prob = -np.inf

        for j in range(len(initial_points)):
            prob = max(prob, gaussian_rbf(np.linalg.norm(sampled_points[i] - initial_points[j]), shape))

        probs.append(prob)
        total_prob += prob

    for i in range(num_points):
        rand = np.random.uniform(0.0, total_prob)
        sum_prob = 0.0

        for j in range(len(sampled_points)):
            sum_prob += probs[j]

            if rand < sum_prob or j == len(sampled_points) - 1:
                curr_points[i] = sampled_points[j]
                total_prob -= probs[j]
                probs[j] = 0.0
                break

    return curr_points

@jit(nopython = True)
def radial_basis_sampling(triangles, initial_points, num_points, kappa, num_farthest, shape):
    if num_farthest is None:
        sampled_points = sample_points(triangles, kappa * num_points)
        radial_basis_points = radial_basis(sampled_points, initial_points, kappa // 2 * num_points, shape)
        return farthest_point(radial_basis_points, None, num_points)
    else:
        sampled_points = sample_points(triangles, kappa * num_points)
        radial_basis_points = radial_basis(sampled_points, initial_points, num_points - num_farthest, shape)
        initial_points = np.concatenate((initial_points, radial_basis_points))
        return np.concatenate((farthest_point(sampled_points, initial_points, num_farthest), radial_basis_points))

@jit(nopython = True)
def sample_on_line_segments(x, x_perturb, sigma):
    small_perturb = 0.003
    norms = norm(x_perturb - x)
    prefix = []

    for i in range(len(norms)):
        if i == 0:
            prefix.append(norms[i])
        else:
            prefix.append(prefix[i - 1] + norms[i])

    total_prob = prefix[-1]
    count = np.zeros(len(norms))

    for i in range(sigma):
        rand = np.random.uniform(0.0, total_prob)
        idx = binary_search(prefix, rand)
        count[idx] += 1.0

    x_sample = np.empty((sigma, 3))
    idx = 0

    for i in range(len(norms)):
        for j in range(count[i]):
            x_sample[idx] = x[i] + (x_perturb[i] - x[i]) * j / count[i] + small_perturb * np.random.randn(3)
            idx += 1

    return x_sample
