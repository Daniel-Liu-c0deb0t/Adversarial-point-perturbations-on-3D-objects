import numpy as np
from perturb_proj_tree import PerturbProjTree
from alpha_shape import alpha_shape_border
from sampling import farthest_point_sampling, radial_basis_sampling, sample_on_line_segments

def iter_l2_attack_n_proj(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    tau = params["tau"]

    epsilon = epsilon / float(n)
    tree = PerturbProjTree(x, thickness = tau)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb
        x_perturb = tree.project(x_perturb, perturb)

    return x_perturb

def mom_l2_attack_n_proj(model, x, y, params):
    epsilon = params["epsilon"]
    mu = params["mu"]
    n = params["n"]
    tau = params["tau"]

    epsilon = epsilon / float(n)
    tree = PerturbProjTree(x, thickness = tau)
    x_perturb = x
    grad = np.zeros(x.shape)

    for i in range(n):
        curr_grad = model.grad_fn(x_perturb, y)
        grad = mu * grad + curr_grad / np.mean(np.abs(curr_grad))
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb
        x_perturb = tree.project(x_perturb, perturb)

    return x_perturb

def mom_l2_attack(model, x, y, params):
    epsilon = params["epsilon"]
    mu = params["mu"]
    n = params["n"]

    epsilon = epsilon / float(n)
    x_perturb = x
    grad = np.zeros(x.shape)

    for i in range(n):
        curr_grad = model.grad_fn(x_perturb, y)
        grad = mu * grad + curr_grad / np.mean(np.abs(curr_grad))
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    return x_perturb

def iter_l2_attack_1_proj(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    tau = params["tau"]

    epsilon = epsilon / float(n)
    tree = PerturbProjTree(x, thickness = tau)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    x_perturb = tree.project(x_perturb, x_perturb - x)

    return x_perturb

def iter_l2_attack(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    return x_perturb

def normal_jitter(model, x, y, params):
    epsilon = params["epsilon"]
    tau = params["tau"]

    tree = PerturbProjTree(x, thickness = tau)
    perturb = np.random.normal(size = x.shape)
    perturb = epsilon * perturb / np.sqrt(np.sum(perturb ** 2))
    x_perturb = x + perturb
    x_perturb = tree.project(x_perturb, perturb)

    return x_perturb

def iter_l2_attack_1_sampling(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    k = params["k"]
    kappa = params["kappa"]
    tri_all_points = params["tri_all_points"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
    perturbed = x_perturb[sort_idx[k:]]

    border_points, border_triangles = alpha_shape_border(x_perturb if tri_all_points else perturbed)

    triangles = []

    for tri in border_triangles:
        triangles.append(border_points[tri])

    sampled = farthest_point_sampling(np.array(triangles), perturbed, k, kappa)

    x_sample = np.empty((len(x_perturb), 3))

    for i in range(len(sort_idx)):
        if i < k:
            x_sample[sort_idx[i]] = sampled[i]
        else:
            x_sample[sort_idx[i]] = x_perturb[sort_idx[i]]

    return x_sample

def iter_l2_attack_n_sampling(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    k = params["k"]
    kappa = params["kappa"]
    tri_all_points = params["tri_all_points"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

        sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
        perturbed = x_perturb[sort_idx[k:]]

        border_points, border_triangles = alpha_shape_border(x_perturb if tri_all_points else perturbed)

        triangles = []

        for tri in border_triangles:
            triangles.append(border_points[tri])

        sampled = farthest_point_sampling(np.array(triangles), perturbed, k, kappa)

        x_sample = np.empty((len(x_perturb), 3))

        for i in range(len(sort_idx)):
            if i < k:
                x_sample[sort_idx[i]] = sampled[i]
            else:
                x_sample[sort_idx[i]] = x_perturb[sort_idx[i]]

        x_perturb = x_sample

    return x_perturb

def iter_l2_attack_1_sampling_all(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    k = params["k"]
    kappa = params["kappa"]
    tri_all_points = params["tri_all_points"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
    perturbed = x_perturb[sort_idx[k:]]

    border_points, border_triangles = alpha_shape_border(x_perturb if tri_all_points else perturbed)

    triangles = []

    for tri in border_triangles:
        triangles.append(border_points[tri])

    sampled = farthest_point_sampling(np.array(triangles), None, len(x_perturb), kappa)

    return sampled

def iter_l2_attack_1_sampling_rbf(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    k = params["k"]
    kappa = params["kappa"]
    num_farthest = params["num_farthest"]
    shape = params["shape"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
    perturbed = x_perturb[sort_idx[k:]]

    border_points, border_triangles = alpha_shape_border(x_perturb)

    triangles = []

    for tri in border_triangles:
        triangles.append(border_points[tri])

    sampled = radial_basis_sampling(np.array(triangles), perturbed, k, kappa, num_farthest, shape)

    x_sample = np.empty((len(x_perturb), 3))

    for i in range(len(sort_idx)):
        if i < k:
            x_sample[sort_idx[i]] = sampled[i]
        else:
            x_sample[sort_idx[i]] = x_perturb[sort_idx[i]]

    return x_sample

def iter_l2_attack_n_sampling_rbf(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    k = params["k"]
    kappa = params["kappa"]
    num_farthest = params["num_farthest"]
    shape = params["shape"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

        sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
        perturbed = x_perturb[sort_idx[k:]]

        border_points, border_triangles = alpha_shape_border(x_perturb)

        triangles = []

        for tri in border_triangles:
            triangles.append(border_points[tri])

        sampled = radial_basis_sampling(np.array(triangles), perturbed, k, kappa, num_farthest, shape)

        x_sample = np.empty((len(x_perturb), 3))

        for i in range(len(sort_idx)):
            if i < k:
                x_sample[sort_idx[i]] = sampled[i]
            else:
                x_sample[sort_idx[i]] = x_perturb[sort_idx[i]]

        x_perturb = x_sample

    return x_perturb

def iter_l2_attack_top_k(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    top_k = params["top_k"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
    x_max = np.empty((len(x_perturb), 3))

    for i in range(len(sort_idx)):
        if i < len(sort_idx) - top_k:
            x_max[sort_idx[i]] = x[sort_idx[i]]
        else:
            x_max[sort_idx[i]] = x_perturb[sort_idx[i]]

    return x_max

def iter_l2_adversarial_sticks(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]
    top_k = params["top_k"]
    sigma = params["sigma"]

    epsilon = epsilon / float(n)
    tree = PerturbProjTree(x)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    sort_idx = np.argsort(np.linalg.norm(x_perturb - x, axis = 1))
    perturbed_idx = sort_idx[-top_k:]
    perturbed = x_perturb[perturbed_idx]
    x_proj = tree.project(perturbed, perturbed - x[perturbed_idx])
    x_sample = sample_on_line_segments(x_proj, perturbed, sigma)
    x_max = np.empty((len(x_perturb), 3))

    for i in range(len(sort_idx)):
        if i < sigma:
            x_max[sort_idx[i]] = x_sample[i]
        elif i < len(sort_idx) - top_k:
            x_max[sort_idx[i]] = x[sort_idx[i]]
        else:
            x_max[sort_idx[i]] = x_perturb[sort_idx[i]]

    return x_max

def iter_l2_attack_fft(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]

    epsilon = epsilon / float(n)
    x_perturb = np.fft.fft2(x)

    for i in range(n):
        grad = model.grad_freq_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad * np.conj(grad)))
        x_perturb = x_perturb + perturb

    return np.real(np.fft.ifft2(x_perturb))

def iter_l2_attack_sinks(model, x, y, params):
    nu = params["nu"]
    epsilon = params["epsilon"]
    n = params["n"]
    num_sinks = params["num_sinks"]

    sinks = np.random.randn(num_sinks, 3)
    sinks = sinks / np.linalg.norm(sinks, axis = 1, keepdims = True)
    sink_coeff = np.zeros(num_sinks)

    for i in range(n):
        grad = model.grad_sink_fn(x, y, sinks, sink_coeff, epsilon)
        sink_coeff = sink_coeff + nu * grad

    return model.x_perturb_sink_fn(x, sinks, sink_coeff, epsilon)
