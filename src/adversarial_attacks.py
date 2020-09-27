import numpy as np
from perturb_proj_tree import PerturbProjTree
from alpha_shape import alpha_shape_border
from sampling import farthest_point_sampling, farthest_point_idx, radial_basis_sampling, sample_on_line_segments

def iter_l2_attack_n_proj(model, x, y, params):
    epsilon = float(params["epsilon"])
    n = int(params["n"])
    tau = float(params["tau"])

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
    epsilon = float(params["epsilon"])
    n = int(params["n"])

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        grad = model.grad_fn(x_perturb, y)
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    return x_perturb

def iter_l2_attack_dropout(model, x, y, params):
    epsilon = float(params["epsilon"])
    n = int(params["n"])

    epsilon = epsilon / float(n)
    x_perturb = x

    for i in range(n):
        mask_idx = np.random.choice(len(x_perturb), 500, replace = False)
        mask = np.zeros(x.shape)
        mask[mask_idx, :] = 1.0
        grad = model.grad_fn(x_perturb * mask, y) * mask
        perturb = epsilon * grad / np.sqrt(np.sum(grad ** 2))
        x_perturb = x_perturb + perturb

    return x_perturb

def normal_jitter(model, x, y, params):
    epsilon = float(params["epsilon"])
    tau = float(params["tau"])

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
    epsilon = float(params["epsilon"])
    n = int(params["n"])
    k = int(params["k"])
    kappa = int(params["kappa"])
    tri_all_points = str(params["tri_all_points"]) == "True"

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
    epsilon = float(params["epsilon"])
    n = int(params["n"])
    top_k = int(params["top_k"])
    sigma = int(params["sigma"])

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
    eta = float(params["eta"])
    mu = float(params["mu"])
    lambda_ = float(params["lambda_"])
    n = int(params["n"])
    num_sinks = int(params["num_sinks"])

    dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis = 2)
    dists = np.where(np.eye(len(x)) > 0.0, np.full(dists.shape, np.inf), dists)
    avg_min_dist = np.mean(np.min(dists, axis = 1))
    mu = mu * avg_min_dist

    grad = model.grad_fn(x, y)
    sort_idx = np.argsort(np.linalg.norm(grad, axis = 1))
    x_grad = x + 2.0 * avg_min_dist * grad / (1e-6 + np.linalg.norm(grad, axis = 1, keepdims = True))
    candidates = x_grad[sort_idx[-(num_sinks * 7):]]
    perturbed_idx = farthest_point_idx(candidates, candidates[-1][np.newaxis, :], num_sinks - 1)
    sink_source = np.append(candidates[perturbed_idx], candidates[-1][np.newaxis, :], axis = 0)
    
    pred_idx = np.argmax(model.pred_fn(x))
    
    lo = 0.0
    hi = float(lambda_)
    res = x
    
    # binary search for lambda
    for _ in range(20):
        mid = (lo + hi) / 2.0
        
        model.reset_sink_fn(sink_source)

        for i in range(n):
            model.train_sink_fn(x, y, sink_source, mu, mid, eta)
        
        x_perturb = model.x_perturb_sink_fn(x, sink_source, mu, mid)
        
        if pred_idx == np.argmax(model.pred_fn(x_perturb)):
            hi = mid
        else:
            lo = mid
            res = x_perturb

    return res

def chamfer_attack(model, x, y, params):
    eta = float(params["eta"])
    alpha = float(params["alpha"])
    lambda_ = float(params["lambda_"])
    n = int(params["n"])
    
    pred_idx = np.argmax(model.pred_fn(x))
    
    lo = 0.0
    hi = float(alpha)
    res = x
    
    # binary search for alpha
    for _ in range(20):
        mid = (lo + hi) / 2.0
        
        model.reset_chamfer_fn(x + 0.0001 * np.random.normal(size = x.shape))

        for i in range(n):
            model.train_chamfer_fn(x, y, mid, lambda_, eta)
        
        x_perturb = model.x_perturb_chamfer_fn(x)
        
        if pred_idx == np.argmax(model.pred_fn(x_perturb)):
            hi = mid
        else:
            lo = mid
            res = x_perturb

    return res

def iter_l2_adversarial_sticks2(model, x, y, params):
    n = int(params["n"])
    top_k = int(params["top_k"])
    sigma = int(params["sigma"])
    eta = float(params["eta"])
    alpha = float(params["alpha"])
    lambda_ = float(params["lambda_"])
    density = float(params["density"])

    sort_idx = np.argsort(np.linalg.norm(model.grad_fn(x, y), axis = 1))
    perturbed_idx = sort_idx[-top_k:]
    mask = np.zeros(x.shape)
    mask[perturbed_idx, :] = 1.0

    tree = PerturbProjTree(x)

    pred_idx = np.argmax(model.pred_fn(x))
    
    lo = 0.0
    hi = float(alpha)
    res = x

    # binary search for alpha
    for _ in range(20):
        mid = (lo + hi) / 2.0
        
        model.reset_sticks_fn(x + 0.0001 * np.random.normal(size = x.shape))

        for i in range(n):
            model.train_sticks_fn(x, y, mask, mid, lambda_, eta)
        
        x_perturb = model.x_perturb_sticks_fn(x, mask)
        
        if pred_idx == np.argmax(model.pred_fn(x_perturb)):
            hi = mid
        else:
            lo = mid
            res = x_perturb

    dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis = 2)
    dists = np.where(np.eye(len(x)) > 0.0, np.full(dists.shape, np.inf), dists)
    avg_min_dist = np.mean(np.min(dists, axis = 1))

    perturbed = res[perturbed_idx]
    x_proj = tree.project(perturbed, perturbed - x[perturbed_idx])
    sticks_len = np.sum(np.linalg.norm(perturbed - x_proj, axis = 1))
    num_resample = min(int(sticks_len / avg_min_dist * density), sigma)
    x_sample = sample_on_line_segments(x_proj, perturbed, num_resample)
    x_unperturbed = x[sort_idx[:-top_k]]
    x_farthest_idx = farthest_point_idx(x_unperturbed, x[perturbed_idx], len(x) - top_k - num_resample)
    x_max = np.concatenate((x_sample, x_unperturbed[x_farthest_idx], perturbed))

    return x_max

def chamfer_dropout(model, x, y, params):
    n = int(params["n"])
    eta = float(params["eta"])
    alpha = float(params["alpha"])
    lambda_ = float(params["lambda_"])

    pred_idx = np.argmax(model.pred_fn(x))
    
    lo = 0.0
    hi = float(alpha)
    res = x

    # binary search for alpha
    for _ in range(20):
        mid = (lo + hi) / 2.0

        x_perturb = x + 0.0001 * np.random.normal(size = x.shape)
        
        for i in range(n):
            model.reset_dropout_fn(x_perturb)
            mask_idx = np.random.choice(len(x), 500, replace = False)
            mask = np.zeros(x.shape)
            mask[mask_idx, :] = 1.0

            for j in range(10):
                model.train_dropout_fn(x, x_perturb, y, mask, mid, lambda_, eta)
        
            x_perturb = model.x_perturb_dropout_fn(x_perturb, np.ones(x.shape)

        if pred_idx == np.argmax(model.pred_fn(x_perturb)):
            hi = mid
        else:
            lo = mid
            res = x_perturb

    return res
