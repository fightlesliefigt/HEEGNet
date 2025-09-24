import torch

from .backward import frechet_ball_backward, frechet_hyperboloid_backward
from .forward import frechet_ball_forward, frechet_hyperboloid_forward

from ..utils.utils_common import TOLEPS

class FrechetMeanBall(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, K,max_iter):
        mean = frechet_ball_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype],max_iter=max_iter)
        ctx.save_for_backward(x, mean, w, K)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        X, mean, w, K = ctx.saved_tensors
        dx, dw, dK = frechet_ball_backward(X, mean, grad_output, w, K)
        return dx, dw, dK, None

class FrechetMeanHyperboloid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, K, max_iter):
        mean = frechet_hyperboloid_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype],max_iter=max_iter)
        ctx.save_for_backward(x, mean, w, K)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        X, mean, w, K = ctx.saved_tensors
        dx, dw, dK = frechet_hyperboloid_backward(X, mean, grad_output, w, K)
        return dx, dw, dK, None


