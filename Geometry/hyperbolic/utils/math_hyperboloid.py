import torch
from .utils_common import EPS, sinhdiv, divsinh

# =======================
# Low-level TorchScript ops
# =======================

@torch.jit.script
def _ldot(u: torch.Tensor, v: torch.Tensor, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
    # Lorentz inner product: <u, v>_L = -u0*v0 + sum_{i>0} u_i*v_i
    m = u * v
    if keepdim:
        # subtract double time component with keepdim=True
        return torch.sum(m, dim=dim, keepdim=True) - 2.0 * m[..., 0:1]
    else:
        return torch.sum(m, dim=dim, keepdim=False) - 2.0 * m[..., 0]

@torch.jit.script
def _calc_time(space: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.sum(space * space, dim=-1, keepdim=True) - 1.0 / K).clamp(eps)

@torch.jit.script
def _projx(x: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    space = x[..., 1:]
    t = _calc_time(space, K, eps)
    return torch.cat([t, space], dim=-1)

@torch.jit.script
def _add_time(space: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    t = _calc_time(space, K, eps)
    return torch.cat([t, space], dim=-1)

@torch.jit.script
def _proju(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    ldot = _ldot(x, u, True)
    return u - K * ldot * x

@torch.jit.script
def _proju0(u: torch.Tensor) -> torch.Tensor:
    zero_time = torch.zeros_like(u[..., :1])
    return torch.cat([zero_time, u[..., 1:]], dim=-1)

@torch.jit.script
def _egrad2rgrad(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    time_one = torch.ones_like(x[..., :1])
    space_zeros = torch.zeros_like(x[..., 1:])
    scaling = torch.cat([time_one, space_zeros], dim=-1)
    u = u - 2.0 * x[..., :1] * scaling
    return _proju(x,u,K)

@torch.jit.script
def _inner_v_0(u: torch.Tensor, K: torch.Tensor, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
    # <u, L_zero>_L = -u_0 / sqrt(-K)
    # u.narrow(dim, 0, 1) picks the time component along the given dim
    res = -u.narrow(dim, 0, 1) / torch.sqrt(-K)
    if keepdim is False:
        res = res.squeeze(dim)
    return res

@torch.jit.script
def _exp(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    un = _ldot(u, u, keepdim=True).sqrt()
    alpha = (un * (-K).sqrt()).clamp(min=eps)
    res = x * torch.cosh(alpha) + sinhdiv(alpha) * u
    return res

@torch.jit.script
def _exp0(u: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    sqrtK = torch.sqrt(-K)
    u_s = u[..., 1:]
    u_s_norm = torch.norm(u_s, p=2, dim=-1, keepdim=True)
    theta = (u_s_norm * sqrtK).clamp(min=eps)

    res = torch.ones_like(u)
    res[..., :1] = torch.cosh(theta) / sqrtK
    res[..., 1:] = sinhdiv(theta) * u_s
    return res

@torch.jit.script
def _log(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    beta = (K * _ldot(x, y, keepdim=True) ).clamp(min=1.0 + eps)
    num = torch.acosh(beta)
    u = divsinh(num) * (y - beta * x)
    return u

@torch.jit.script
def _log0(x: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    sqrtK = torch.sqrt(-K)
    x_t = x[..., :1]
    x_s = x[..., 1:]
    dom = (torch.norm(x_s, p=2, dim=-1, keepdim=True)*sqrtK).clamp(eps)
    theta = (sqrtK * x_t).clamp_min(1.0 + eps)
    scale = torch.acosh(theta) / dom
    res = torch.zeros_like(x)
    res[..., 1:] = scale * x_s
    return res

@torch.jit.script
def _dist(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor, eps: float, keepdim: bool = False) -> torch.Tensor:
    beta = (K * _ldot(x, y, keepdim=keepdim)).clamp(min=1.0 + eps)
    res = torch.acosh(beta)
    return res / torch.sqrt(-K)

@torch.jit.script
def _dist0(x: torch.Tensor, K: torch.Tensor, eps: float, keepdim: bool = False) -> torch.Tensor:
    sqrtK = torch.sqrt(-K)
    if keepdim:
        xt = x[..., :1]
    else:
        xt = x[..., 0]
    theta = (sqrtK * xt).clamp_min(1.0 + eps)
    return torch.acosh(theta) / sqrtK

@torch.jit.script
def _transp(x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    xy = _ldot(x, y, keepdim=True)
    uy = _ldot(u, y, keepdim=True)
    denom = (1.0 + K * xy).clamp_min(eps)
    return u - (K * uy / denom) * (x + y)

@torch.jit.script
def _transpfrom0(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    sqrtK = torch.sqrt(-K)
    spatial_x = x[..., 1:]
    spatial_u = u[..., 1:]
    dot = torch.sum(spatial_x * spatial_u, dim=-1, keepdim=True)  # <x_s, u_s>_Euclid
    denom = (1.0 + sqrtK * x[..., :1]).clamp_min(eps)
    factor = (K * dot) / denom
    correction = torch.zeros_like(u)
    correction[..., :1] = x[..., :1] + 1.0 / sqrtK
    correction[..., 1:] = spatial_x
    return u - factor * correction

@torch.jit.script
def _gyroscalarprod(x: torch.Tensor, r: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    sqrtK = torch.sqrt(-K)
    xt = x[..., 0:1]
    xs = x[..., 1:]
    xs_norm = torch.norm(xs, p=2, dim=-1, keepdim=True).clamp_min(eps)
    theta = torch.acosh((sqrtK * xt).clamp_min(1.0 + eps)) # 注意数值域要 ≥ 1
    rt = r * theta
    out = torch.zeros_like(x)
    out[..., 0:1] = torch.cosh(rt)
    out[..., 1:] = (torch.sinh(rt) / xs_norm) * xs
    return out / sqrtK

@torch.jit.script
def _gyroinv(x: torch.Tensor) -> torch.Tensor:
    xt = x[..., :1]
    xs = x[..., 1:]
    return torch.cat([xt, -xs], dim=-1)

@torch.jit.script
def _gyroadd_radius(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor, eps: float) -> torch.Tensor:
    sqrt_absK = torch.sqrt(torch.abs(K))

    x_t = x[..., :1]
    y_t = y[..., :1]
    x_s = x[..., 1:]
    y_s = y[..., 1:]

    a = 1.0 + sqrt_absK * x_t
    b = 1.0 + sqrt_absK * y_t

    n_x = (x_s * x_s).sum(dim=-1, keepdim=True)
    n_y = (y_s * y_s).sum(dim=-1, keepdim=True)
    s_xy = (x_s * y_s).sum(dim=-1, keepdim=True)

    # D, N
    D = (a * a) * (b * b) - 2.0 * K * a * b * s_xy + (K * K) * n_x * n_y
    N = (a * a) * n_y + 2.0 * a * b * s_xy + (b * b) * n_x

    denom = D + K * N
    # sign-preserving stabilization
    sign = torch.sign(denom)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom_safe = denom + sign * eps

    # z_t
    z_t = ((D - K * N) / denom_safe) / sqrt_absK

    # z_s
    A_x = a * (b * b) - 2.0 * K * b * s_xy - K * a * n_y
    A_y = b * (a * a + K * n_x)
    coef = 2.0 / denom_safe
    z_s = coef * (A_x * x_s + A_y * y_s)

    return torch.cat((z_t, z_s), dim=-1)


# =======================
# Public base-compatible wrappers
# =======================

def ldot(u: torch.Tensor, v: torch.Tensor, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
    return _ldot(u, v, keepdim, dim)

def inner_v_0(u: torch.Tensor, K: torch.Tensor, keepdim: bool = False, dim: int = -1) -> torch.Tensor:
    return _inner_v_0(u, K, keepdim, dim)

def calc_time(space: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _calc_time(space, K, EPS[space.dtype])

def projx(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _projx(x, K, EPS[x.dtype])

def add_time(space: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _add_time(space, K, EPS[space.dtype])

def proju(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _proju(x, u, K)

def proju0(u: torch.Tensor) -> torch.Tensor:
    return _proju0(u)

def egrad2rgrad(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _egrad2rgrad(x, u, K)

def exp(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _exp(x, u, K, EPS[x.dtype])

def exp0(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _exp0(u, K, EPS[u.dtype])

def log(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _log(x, y, K, EPS[x.dtype])

def log0(x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _log0(x, K, EPS[x.dtype])

def dist(x: torch.Tensor, y: torch.Tensor, K: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return _dist(x, y, K, EPS[x.dtype], keepdim)

def dist0(x: torch.Tensor, K: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    return _dist0(x, K, EPS[x.dtype], keepdim)


def transp(x: torch.Tensor, y: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _transp(x, y, u, K, float(EPS[x.dtype]))

def transpfrom0(x: torch.Tensor, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _transpfrom0(x, u, K, float(EPS[x.dtype]))

def gyroscalarprod(x: torch.Tensor, r: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    return _gyroscalarprod(x, r, K, EPS[x.dtype])

def gyroinv(x: torch.Tensor) -> torch.Tensor:
    return _gyroinv(x)
def gyroadd_radius(x: torch.Tensor, y: torch.Tensor, K) -> torch.Tensor:
    return _gyroadd_radius(x, y, K, EPS[x.dtype])

