import torch
import torch.nn.functional as F
import numpy as np

from utils_hyperbolic import projection, minkowski_ip, minkowski_ip2


def emd1D(u_values, v_values, u_weights=None, v_weights=None,p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    zero = torch.zeros(1, dtype=dtype, device=device)
    
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    if p == 1:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
    if p == 2:
        return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
    return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)


def sliced_cost(Xs, Xt, v, u_weights=None, v_weights=None, p=1):
    n, _ = Xs.shape
    m, _ = Xt.shape
    
    device = Xs.device

    n_proj, d = v.shape

    x0 = torch.zeros((1,d), device=device)
    x0[0,0] = 1
        
    ## Projection geodesic
    #Xps_geod = projection(Xs, x0, v)
    #Xpt_geod = projection(Xt, x0, v)
    
    #Xps = torch.arcsinh(torch.sum(Xps_geod[:,None]*v[None], axis=-1)).reshape(-1,n_proj) 
    #Xpt = torch.arcsinh(torch.sum(Xpt_geod[:,None]*v[None], axis=-1)).reshape(-1,n_proj) 

    ## Get coordinates on R
    # Xps = torch.sign(torch.sum(Xps_geod[:,None]*v[None], axis=-1)).reshape(-1, n_proj) \
    #        * torch.arccosh(torch.clamp(-minkowski_ip(Xps_geod.reshape(-1, d), x0), min=1+1e-5)).reshape(-1, n_proj)
    # Xpt = torch.sign(torch.sum(Xpt_geod[:,None]*v[None], axis=-1)).reshape(-1, n_proj) \
    #        * torch.arccosh(torch.clamp(-minkowski_ip(Xpt_geod.reshape(-1, d), x0), min=1+1e-5)).reshape(-1, n_proj)

    
    
    ip_x0_Xs = minkowski_ip(x0, Xs)
    ip_v_Xs = minkowski_ip2(v, Xs)
    
    ip_x0_Xt = minkowski_ip(x0, Xt)
    ip_v_Xt = minkowski_ip2(v, Xt)
    
    Xps = torch.arctanh(-ip_v_Xs/ip_x0_Xs)
    Xpt = torch.arctanh(-ip_v_Xt/ip_x0_Xt)
    
    return torch.mean(emd1D(Xps.T,Xpt.T,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p))


def hyper_sliced_wasserstein(Xs, Xt, num_projections, device,
                       u_weights=None, v_weights=None, p=2):
    n, d = Xs.shape

    # Random projection directions, shape (d-1, num_projections)
    vs = np.random.normal(size=(num_projections, d-1))
    vs = F.normalize(torch.from_numpy(vs), p=2, dim=-1).type(Xs.dtype).to(device)
    vs = F.pad(vs, (1,0))
    
    return sliced_cost(Xs, Xt, v=vs,
                       u_weights=u_weights,
                       v_weights=v_weights,
                       p=p)
