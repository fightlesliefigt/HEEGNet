import torch
import torch.distributions as D
import torch.nn.functional as F
import numpy as np

from utils_hyperbolic import parallelTransport, expMap


def sampleWrappedNormal(mu, Sigma, n):
    device = mu.device
    
    d = len(mu)
    normal = D.MultivariateNormal(torch.zeros((d-1,), device=device),Sigma)
    x0 = torch.zeros((1,d), device=device)
    x0[0,0] = 1
    
    ## Sample in T_x0 H
    v_ = normal.sample((n,))
    v = F.pad(v_, (1,0))
    
    ## Transport to T_\mu H and project on H
    u = parallelTransport(v, x0, mu)    
    y = expMap(u, mu)
    
    return y 