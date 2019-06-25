import numpy as np
from perturb_proj_tree import PerturbProjTree
from alpha_shape import alpha_shape_border
from sampling import farthest_point_sampling

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

    x_perturb = tree.project(x_perturb, perturb)

    return x_perturb

def iter_l2_attack_1_sampling(model, x, y, params):
    epsilon = params["epsilon"]
    n = params["n"]

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    zero_grad_mask = np.all(np.isclose(x_perturb, 0.0), axis = 1)
    not_perturbed_count = np.sum(zero_grad_mask)
    perturbed = x_perturb[np.logical_not(zero_grad_mask)]

    border_points, border_triangles = alpha_shape_border(perturbed)

    triangles = []

    for tri in border_triangles:
        triangles.append(border_points[tri])

    sampled = farthest_point_sampling(np.array(triangles), perturbed, not_perturbed_count, kappa = 3)

    idx = 0
    x_sample = []

    for i in range(len(x_perturb)):
        if zero_grad_mask[i]:
            x_sample.append(sampled[idx])
            idx += 1
        else:
            x_sample.append(x_perturb[i])

    return np.vstack(x_sample)
