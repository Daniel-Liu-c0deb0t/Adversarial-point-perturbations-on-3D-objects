import numpy as np
from perturb_proj_tree import PerturbProjTree

def iter_l2_attack_n_proj(grad_fn, x, epsilon, n, tau):
    epsilon = epsilon / n
    tree = PerturbProjTree(x, thickness = tau)
    x_perturb = x

    for i in range(n):
        grad = grad_fn(x_perturb)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb
        x_perturb = tree.project(x_perturb, perturb)

    return x_perturb

def iter_l2_attack_1_proj(grad_fn, x, epsilon, n, tau):
    epsilon = epsilon / n
    tree = PerturbProjTree(x, thickness = tau)
    x_perturb = x

    for i in range(n):
        grad = grad_fn(x_perturb)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    x_perturb = tree.project(x_perturb, perturb)

    return x_perturb
