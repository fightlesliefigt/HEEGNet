import abc

import torch

EPS = {torch.float32: 1e-4, torch.float64: 1e-8}


class Manifold(torch.nn.Module):
    # metaclass = abc.ABCMeta
    name = property(lambda self: self.__class__.__name__)
    def __init__(self,):
        super().__init__()


    @abc.abstractmethod
    def _check_point_on_manifold(self, x, atol=1e-5, rtol=1e-5, dim=-1):
        """Check whether a point x lies in the manifold"""
        pass


    def _check_vector_on_tangent(self, x, u, atol=1e-5, rtol=1e-5, dim=-1):
        """ Check whether a vector is a valid tangent vector."""
        pass

    @abc.abstractmethod
    def random_normal(self, *size, mean=0, std=1,scale=1):
        """
        Sample from Gaussian in tangent space at origin and map to the manifold.
        size contains the manifold dimension
        """
        tangent = self.random_tangent_origin(*size, mean=mean, std=std, scale=scale)
        return self.exp0(tangent, project=True)

    @abc.abstractmethod
    def random_tangent_origin(self, *size, mean=0, std=1,scale=1):
        """Sample from Gaussian in tangent space at origin and map to the manifold."""
        pass

    @abc.abstractmethod
    def zero(self, *shape):
        """shape contains manifold dimension"""
        pass

    @abc.abstractmethod
    def zero_like(self, x):
        pass

    @abc.abstractmethod
    def zero_tan(self, *shape):
        pass

    @abc.abstractmethod
    def zero_tan_like(self, x):
        pass

    @abc.abstractmethod
    def inner(self, x, u, v, keepdim=False):
        pass

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u):
        pass

    def proju0(self, u):
        return self.proju(self.zero_like(u), u)

    @abc.abstractmethod
    def projx(self, x):
        pass

    def egrad2rgrad(self, x, u):
        return self.proju(x, u)

    @abc.abstractmethod
    def exp(self, x, u):
        pass

    def exp0(self, u):
        return self.exp(self.zero_like(u), u)

    @abc.abstractmethod
    def log(self, x, y):
        pass

    def log0(self, y):
        return self.log(self.zero_like(y), y)

    def geodesic(self, x,y,t):
        """Geodesic from x to y"""
        return self.exp(x, t* self.log(x, y))
        
    def dist(self, x, y, squared=False, keepdim=False):
        return self.norm(x, self.log(x, y), squared, keepdim)

    def dist0(self, x, dim=-1, squared=False, keepdim=False):
        return self.norm(x, self.log0(x), squared, keepdim)

    def pdist(self, x, squared=False):
        assert x.ndim == 2
        n = x.shape[0]
        m = torch.triu_indices(n, n, 1, device=x.device)
        return self.dist(x[m[0]], x[m[1]], squared=squared, keepdim=False)

    @abc.abstractmethod
    def transp(self, x, y, u):
        pass

    def transpfrom0(self, x, u):
        return self.transp(self.zero_like(x), x, u)
    
    def transpto0(self, x, u):
        return self.transp(x, self.zero_like(x), u)

    def gyroadd(self, x, y):
        u = self.log0(y)
        v = self.transpfrom0(x, u)
        return self.exp(x, v)

    def gyroscalarprod(self, x, r):
        return self.exp0(x, r * self.log0(x))

    def gyroinv(self, x):
        return self.exp0(- self.log0(x))

    @abc.abstractmethod
    def gyration(self, u, v, w):
        # ⊖ (u ⊕ v) ⊕ (u ⊕ (v ⊕ w))
        u_plus_v = self.gyroadd(u, v)
        v_plus_w = self.gyroadd(v, w)
        return self.gyroadd(self.gyroinv(u_plus_v), self.gyroadd(u,v_plus_w))

    def gyrotrans(self, x, y,translate='Left'):
        """Gyro translation"""
        if translate=='Left':
            res=self.gyroadd(x,y)
        elif translate=='Right':
            res = self.gyroadd(y,x)
        return res

    @abc.abstractmethod
    def sh_to_dim(self, shape):
        """ambient space dim to intrinsic dim"""
        pass

    @abc.abstractmethod
    def dim_to_sh(self, dim):
        """intrinsic dim to ambient space dim"""
        pass

    @abc.abstractmethod
    def squeeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def unsqueeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def __str__(self):
        return f'{self.name}'

    def frechet_variance(self, x, mu, w=None):
        """
        Args
        ----
            x (tensor): points of shape [..., points, dim]
            mu (tensor): mean of shape [..., dim]
            w (tensor): weights of shape [..., points]

            where the ... of the three variables line up
        
        Returns
        -------
            tensor of shape [...]
        """
        distance = self.dist(x, mu.unsqueeze(-2), squared=True)
        if w is None:
            return distance.mean(dim=-1)
        else:
            return (distance * w).sum(dim=-1)

    def karcher_mean(self,x,max_iter=1000,w=None,batchdim=[0]):
        # currently only support [bs,n]
        if w is None:
            w = torch.ones(x.shape[0],1).to(x)/x.shape[0]
        init_point = x[tuple(0 for _ in batchdim)]  # Dynamically selects first elements along batchdim
        for ith in range(max_iter):
            tan_data = self.log(init_point,x)
            tan_mean = (tan_data * w).sum(dim=batchdim)
            condition = tan_mean.norm(dim=batchdim)
            # print(f'{ith+1}: {condition}')
            if torch.all(condition<=1e-4):
                # th.sum(condition>=0.1)
                # print('early stop')
                break
            init_point = self.exp(init_point,tan_mean)
        return init_point

    def frechet_mean(self,x,max_iter=1000,w=None):
        return self.karcher_mean(x,max_iter=max_iter,w=w,batchdim=[0])

class ConstantCurvature(Manifold):

    def __init__(self, K=-1.0):
        super().__init__()
        K_tensor = K if isinstance(K, torch.Tensor) else torch.tensor(K, dtype=torch.float64)
        self.register_buffer("K", K_tensor)

class Hyperbolic(ConstantCurvature):

    def __init__(self, K=-1.0):
        super().__init__(K=K)
        assert K < 0

