import math
from typing import Tuple
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.types import Number
import torch.nn as nn
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel, Sphere
from . import functionals


class CovariancePool(nn.Module):
    def __init__(self, alpha = None, unitvar = False):
        super().__init__()
        self.pooldim = -1
        self.chandim = -2
        self.alpha = alpha
        self.unitvar = unitvar
    
    def forward(self, X : Tensor) -> Tensor:
        X0 = X - X.mean(dim=self.pooldim, keepdim=True)
        if self.unitvar:
            X0 = X0 / X0.std(dim=self.pooldim, keepdim=True)
            X0.nan_to_num_(0)

        C = (X0 @ X0.transpose(-2, -1)) / X0.shape[self.pooldim]
        if self.alpha is not None:
            Cd = C.diagonal(dim1=self.pooldim, dim2=self.pooldim-1)
            Cd += self.alpha
        return C


class BiMap(nn.Module):
    def __init__(self, shape : Tuple[int, ...] or torch.Size, W0 : Tensor = None, manifold='stiefel', **kwargs):
        super().__init__()

        if manifold == 'stiefel':
            assert(shape[-2] >= shape[-1])
            mf = Stiefel()
        elif manifold == 'sphere':
            mf = Sphere()
            shape = list(shape)
            shape[-1], shape[-2] = shape[-2], shape[-1]
        else:
            raise NotImplementedError()

        # add constraint (also initializes the parameter to fulfill the constraint)
        self.W = ManifoldParameter(torch.empty(shape, **kwargs), manifold=mf)

        # optionally initialize the weights (initialization has to fulfill the constraint!)
        if W0 is not None:
            self.W.data = W0 # e.g., self.W = torch.nn.init.orthogonal_(self.W)
        else:
            self.reset_parameters()
    
    def forward(self, X : Tensor) -> Tensor:
        if isinstance(self.W.manifold, Sphere):
            return self.W @ X @ self.W.transpose(-2,-1)
        else:
            return self.W.transpose(-2,-1) @ X @ self.W

    @torch.no_grad()
    def reset_parameters(self):
        if isinstance(self.W.manifold, Stiefel):
            # uniform initialization on stiefel manifold after theorem 2.2.1 in Chikuse (2003): statistics on special manifolds
            W = torch.rand(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            self.W.data = W @ functionals.sym_invsqrtm.apply(W.transpose(-1,-2) @ W)
        elif isinstance(self.W.manifold, Sphere):
            W = torch.empty(self.W.shape, dtype=self.W.dtype, device=self.W.device)
            # kaiming initialization std2uniformbound * gain * fan_in
            bound = math.sqrt(3) * 1. / W.shape[-1]
            W.uniform_(-bound, bound)
            # constraint has to be satisfied
            self.W.data = W / W.norm(dim=-1, keepdim=True)
        else:
            raise NotImplementedError()


class ReEig(nn.Module):
    def __init__(self, threshold : Number = 1e-4):
        super().__init__()
        self.threshold = Tensor([threshold])

    def forward(self, X : Tensor) -> Tensor:
        return functionals.sym_reeig.apply(X, self.threshold)


class LogEig(nn.Module):
    def __init__(self, ndim, tril=True):
        super().__init__()

        self.tril = tril
        if self.tril:
            ixs_lower = torch.tril_indices(ndim,ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.ixs = torch.cat((ixs_diag[None,:].tile((2,1)), ixs_lower), dim=1)
        self.ndim = ndim

    def forward(self, X : Tensor) -> Tensor:
        return self.embed(functionals.sym_logm.apply(X))

    def embed(self, X : Tensor) -> Tensor:
        if self.tril:
            x_vec = X[...,self.ixs[0],self.ixs[1]]
            x_vec[...,self.ndim:] *= math.sqrt(2)
        else:
            x_vec = X.flatten(start_dim=-2)
        return x_vec
