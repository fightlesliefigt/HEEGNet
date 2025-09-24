import torch

from ...utils.utils_common import EPS, darcosh


def _ldot(u, v, keepdim=False, dim=-1):
    m = u * v
    if keepdim:
        ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
    else:
        ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
    return ret

def frechet_hyperboloid_forward(X, w, K=-1.0, max_iter=1000, rtol=1e-6, atol=1e-6, verbose=False):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        w (tensor): weights of shape [..., points]
        K (float): curvature (must be negative)
    Returns
    -------
        frechet mean (tensor): shape [..., dim]
    """
    mu = X[..., 0, :].clone()

    mu_prev = mu
    iters = 0
    for ith in range(max_iter):
        inner = K * _ldot(X, mu.unsqueeze(-2), keepdim=True)
        u = (w.unsqueeze(-1) * darcosh(inner) * X).sum(dim=-2)
        mu = u / (K * _ldot(u, u, keepdim=True)).sqrt()

        dist = (mu - mu_prev).norm(dim=-1)
        prev_dist = mu_prev.norm(dim=-1)
        # print(f'{ith+1}: {dist}')
        if (dist < atol).all() or (dist / prev_dist < rtol).all():
            break

        mu_prev = mu
        iters += 1

    if verbose:
        print(iters)

    return mu
