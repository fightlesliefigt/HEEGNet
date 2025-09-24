import torch
from .frechetmean.frechet import FrechetMeanHyperboloid

from ..base import Hyperbolic
from .utils import math_hyperboloid as math

class Hyperboloid(Hyperbolic):
    """
    The hyperboloid model is known as the Lorentz model.
    \mathbb{H}_K^n= \left\{x \in \mathbb{R}^{n+1} | \|x\|_{\mathcal{L}}^2=\frac{1}{K}, x_1>0 \right\},
    where \|x\|_{\mathcal{L}}^2=\sum_{i=2}^{n+1} x_i^2 - x_1^2, $K<0$ is the negative curvature
    0 denotes the lorentz origin \Lzero = [sqrt(-1/K), 0, ..., 0]^\top

    References
    ----------
        [a] Fully Hyperbolic Convolutional Neural Networks for Computer Vision
        [b] Mixed-curvature Variational Autoencoders, with R=\sqrt(-1/K)
        [c] Differentiating through the Fréchet Mean
        [d] Riemannian Batch Normalization: A Gyro Approach 2025
    """
    def __init__(self, K=-1.0):
        super(__class__, self).__init__(K=K)
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-4, dim=-1):
        """<x, x>_L = 1 / K and x[..., 0] > 0."""
        quad_form = self.ldot(x,x)
        expected = 1.0 / self.K
        ok1 = torch.allclose(quad_form, expected, atol=atol, rtol=rtol)
        ok2 = (x[..., 0] > 0).all()
        ok = ok1 and ok2

        if not ok:
            if not ok1:
                reason = f"⟨x,x⟩_L ≠ 1/K = {expected.item():.4f}"
            elif not ok2:
                reason = "x[..., 0] ≤ 0: first component must be positive"
            else:
                reason = "Unknown manifold violation"
        else:
            reason = None

        return ok, reason

    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5, dim=-1):
        """
        Check whether a vector u lies in the tangent space T_x M at point x:
        <x, u>_L = 0

        Args:
            x (Tensor): base point on the manifold of shape [..., n+1]
            u (Tensor): tangent vector of same shape
            atol (float): absolute tolerance
            rtol (float): relative tolerance
            dim (int): dimension to contract over

        Returns:
            (bool, str or None): True if u ∈ T_x M; otherwise, False and reason
        """
        inner = self.ldot(x, u, keepdim=True, dim=dim)
        ok = torch.allclose(inner, torch.zeros_like(inner), atol=atol, rtol=rtol)
        reason = None if ok else "⟨x, u⟩_L ≠ 0: u is not in the tangent space at x"
        return ok, reason

    def random_tangent_origin(self, *size, mean=0, std=1,scale=1):
        n = size[-1] # intrinsic dimension
        # Sample spatial part in R^n
        spatial = torch.randn(*size[:-1], n, device=self.K.device, dtype=self.K.dtype) * std + mean
        # Pad zero in the time coordinate
        tangent = torch.cat([
            torch.zeros(*size[:-1], 1, device=self.K.device, dtype=self.K.dtype),  # time part = 0
            spatial
        ], dim=-1)
        return tangent * scale

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1] - 1
        else:
            return sh - 1

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1] + 1
        else:
            return dim + 1

    def zero(self, *shape):
        x = torch.zeros(*shape, dtype=self.K.dtype, device=self.K.device)
        x[..., 0] = 1 / (-self.K).sqrt()
        return x

    def zero_tan(self, *shape):
        return torch.zeros(*shape, dtype=self.K.dtype, device=self.K.device)

    def zero_like(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = 1 / (-self.K).sqrt()
        return y

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return torch.cat((torch.zeros_like(x[..., 0]).unsqueeze(-1), x), dim=-1)

    def ldot(self, u, v, keepdim=False, dim=-1):
        return math.ldot(u, v, keepdim, dim)

    def calc_time(self, space):
        # Compute time component x0 = sqrt(||space||^2 - 1/K)
        return math.calc_time(space,self.K)

    def projx(self, x):
        """-x_0^2+\sum_{i=1}^n x_i^2=\frac{1}{K} \quad \Rightarrow \quad x_0=\sqrt{\frac{1}{-K}+\sum_{i=1}^n x_i^2}"""
        return math.projx(x,self.K)

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        return math.add_time(space,self.K)

    def proju(self, x, u):
        return math.proju(x,u,self.K)

    def proju0(self, u):
        return math.proju0(u, self.K)

    def egrad2rgrad(self, x, u):
        return math.egrad2rgrad(x, u, self.K)

    def inner(self, x, u, v, keepdim=False):
        return math.ldot(u, v, keepdim=keepdim)

    def inner0(self, u, v, keepdim=False):
        return math.ldot(u, v, keepdim=keepdim)

    def inner_v_0(self, u,dim=-1, keepdim=False):
        """<v, \Lzero> _L """
        return math.inner_v_0(u, keepdim=keepdim, dim=dim)

    def exp(self, x, u, project=False):
        """ Note that [b, c] didn't use self.projx(res), while [a] did"""
        res = math.exp(x, u, self.K)
        if project:
            return self.projx(res)
        else:
            return res

    def exp0(self, u,project=False):
        res = math.exp0(u, self.K)
        if project:
            return self.projx(res)
        else:
            return res

    def log(self, x, y):
        return math.log(x, y,self.K)

    def log0(self, x):
        return math.log0(x, self.K)

    def dist(self, x, y, squared=False, keepdim=False):
        dist = math.dist(x, y, self.K, keepdim=keepdim)
        return dist.pow(2) if squared else dist

    def dist0(self, x, squared=False, keepdim=False):
        dist = math.dist0(x, self.K, keepdim=keepdim)
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        return math.transp(x, y, u, self.K)

    def transpfrom0(self, x, u):
        return math.transpfrom0(x, u, self.K)

    def gyroscalarprod(self, x, r):
        return math.gyroscalarprod(x, r, self.K)

    def gyroinv(self, x):
        return math.gyroinv(x)

    def _gyroadd_via_riem(self, x, y):
        """Calculating gyroadd via definition"""
        u = self.log0(y)
        v = self.transpfrom0(x, u)
        return self.exp(x, v)

    def gyroadd(self, x, y):
        """Closed-form"""
        return math.gyroadd_radius(x, y, self.K)

    def frechet_mean(self,x,max_iter=1000,w=None):
        """Alg. 3: Differentiating through the Fréchet Mean"""
        if w is None:
            w = torch.ones(x.shape[:-1]).to(x)
        return FrechetMeanHyperboloid.apply(x, w, self.K,max_iter)