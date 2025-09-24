import torch
from .hyperboloid import Hyperboloid

#TODO don't need to include this
def hyperboloid_to_ball(x, K):
    return x[..., 1:] / (1 + ((-K)**0.5) * x[..., :1])

def ball_to_hyperboloid(x, K):
    R = 1/ ((-K) **0.5)
    xnormsq = x.norm(dim=-1, keepdim=True).pow(2)
    first_part = R * (1 - K * xnormsq) / (1 + K * xnormsq)
    sec_part = (2 * x) / (1 + K * xnormsq)

    return torch.cat((first_part, sec_part), dim=-1)
