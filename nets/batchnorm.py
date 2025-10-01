from builtins import NotImplementedError
from enum import Enum
from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.types import Number

from geoopt.tensor import ManifoldParameter, ManifoldTensor
from . import functionals
from Geometry.hyperbolic import Hyperboloid

class BatchNormTestStatsMode(Enum):
    BUFFER = 'buffer'
    REFIT = 'refit'
    ADAPT = 'adapt'


class BatchNormDispersion(Enum):
    NONE = 'mean'
    SCALAR = 'scalar'
    VECTOR = 'vector'


class BatchNormTestStatsInterface:
    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        pass

# %% base classes

class BaseBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self, eta = 1.0, eta_test = 0.1, test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER):
        super().__init__()
        self.eta = eta
        self.eta_test = eta_test
        self.test_stats_mode = test_stats_mode

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        self.test_stats_mode = mode


class SchedulableBatchNorm(BaseBatchNorm):
    def set_eta(self, eta = None, eta_test = None):
        if eta is not None:
            self.eta = eta
        if eta_test is not None:
            self.eta_test = eta_test


class BaseDomainBatchNorm(nn.Module, BatchNormTestStatsInterface):
    def __init__(self):
        super().__init__()
        self.batchnorm = torch.nn.ModuleDict()

    def set_test_stats_mode(self, mode : BatchNormTestStatsMode):
        for bn in self.batchnorm.values():
            if isinstance(bn, BatchNormTestStatsInterface):
                bn.set_test_stats_mode(mode)

    def add_domain_(self, layer : BaseBatchNorm, domain : Tensor):
        self.batchnorm[str(domain.item())] = layer

    def get_domain_obj(self, domain : Tensor):
        return self.batchnorm[domain.item()]

    @torch.no_grad()
    def initrunningstats(self, X, domain):
        self.batchnorm[str(domain.item())].initrunningstats(X)

    def forward_domain_(self, X, domain):
        res = self.batchnorm[str(domain.item())](X)
        return res

    def forward(self, X, d):
        du = d.unique()

        X_normalized = torch.empty_like(X)
        res = [(self.forward_domain_(X[d==domain], domain),torch.nonzero(d==domain))
                for domain in du]
        X_out, ixs = zip(*res)
        X_out, ixs = torch.cat(X_out), torch.cat(ixs).flatten()
        X_normalized[ixs] = X_out
        
        return X_normalized


# %% SPD manifold implementation

class SPDBatchNormImpl(BaseBatchNorm):
    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim : int, 
                 eta = 1., eta_test = 0.1,
                 karcher_steps : int = 1, learn_mean = True, learn_std = True, 
                 dispersion : BatchNormDispersion = BatchNormDispersion.SCALAR, 
                 eps = 1e-5, mean = None, std = None, **kwargs):
        super().__init__(eta, eta_test)
        # the last two dimensions are used for SPD manifold


        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.batchdim = batchdim
        self.karcher_steps = karcher_steps
        self.eps = eps
        self.manifold = Hyperboloid()
        
        init_mean = self.manifold.zero(shape[-1])
        init_var = torch.ones((1))

        self.register_buffer('running_mean', init_mean)
        self.register_buffer('running_var', init_var)
        self.register_buffer('running_mean_test', init_mean)
        self.register_buffer('running_var_test', init_var)

        self.mean = mean
        
        if self.dispersion is not BatchNormDispersion.NONE:
            if std is not None:
                self.std = std
            else:
                if self.learn_std:
                    self.std = nn.parameter.Parameter(init_var.clone())
                else:
                    self.std = init_var.clone()


    @torch.no_grad()
    def initrunningstats(self, X):
        self.running_mean = self.manifold.frechet_mean(X,max_iter=100)
        self.running_mean_test = self.running_mean.clone()

        if self.dispersion is BatchNormDispersion.SCALAR:
            self.running_var = self.manifold.frechet_variance(X, self.running_mean_test)
            self.running_var_test = self.running_var.clone()

    def forward(self, X):
        bs, h, w, c = X.shape
        X = X.view(-1, c)   
        if self.training:
            # compute the Karcher flow for the current batch
            batch_mean = self.manifold.frechet_mean(X, max_iter=100)
            # update the running mean
            #rm = functionals.spd_2point_interpolation(self.running_mean, batch_mean, self.eta)
            rm = self.manifold.geodesic(self.running_mean, batch_mean,self.eta)
            if self.dispersion is BatchNormDispersion.SCALAR:
                batch_var = self.manifold.frechet_variance(X,batch_mean)
                rv = (1. - self.eta) * self.running_var + self.eta * batch_var

        else:
            if self.test_stats_mode == BatchNormTestStatsMode.BUFFER:
                pass # nothing to do: use the ones in the buffer
            elif self.test_stats_mode == BatchNormTestStatsMode.REFIT:
                self.initrunningstats(X)
            elif self.test_stats_mode == BatchNormTestStatsMode.ADAPT:              
                pass
            rm = self.running_mean_test
            if self.dispersion is BatchNormDispersion.SCALAR:
                rv = self.running_var_test

        # rescale to desired dispersion
        if self.dispersion is BatchNormDispersion.SCALAR:
            inv_input_mean = self.manifold.gyroinv(rm)
            Xn = self.manifold.gyrotrans(inv_input_mean,X)
            factor = 1 / (rv + self.eps).sqrt()
            Xn = self.manifold.gyroscalarprod(Xn,factor)

        else:
            inv_input_mean = self.manifold.gyroinv(rm)
            Xn = self.manifold.gyrotrans(inv_input_mean,X)

        if self.training:
            with torch.no_grad():
                self.running_mean = rm.clone()
                self.running_mean_test = self.manifold.geodesic(self.running_mean_test, batch_mean,self.eta_test)
                if self.dispersion is not BatchNormDispersion.NONE:
                    self.running_var = rv.clone()
                    batch_var_test = self.manifold.frechet_variance(X,batch_mean)
                    self.running_var_test = (1. - self.eta_test) * self.running_var_test + self.eta_test * batch_var_test
        Xn = Xn.view(bs, h, w, c)
        return Xn


class SPDBatchNorm(SPDBatchNormImpl):
    """
    Batch normalization on the SPD manifold.
    
    Class implements [Brooks et al. 2019, NIPS] (dispersion= ``BatchNormDispersion.NONE``) 
    and [Kobler et al. 2022, ICASSP] (dispersion= ``BatchNormDispersion.SCALAR``).
    By default dispersion = ``BatchNormDispersion.SCALAR``.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass. Use another batch normailzation variant.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=1.0, eta_test=eta, **kwargs)


class SPDBatchReNorm(SPDBatchNormImpl):
    """
    Batch re normalization on the SPD manifold [Kobler et al. 2022, ICASSP].
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=0.1, **kwargs):
        if 'dispersion' not in kwargs.keys():
            kwargs['dispersion'] = BatchNormDispersion.SCALAR
        if 'eta_test' in kwargs.keys():
            raise RuntimeError('This parameter is ignored in this subclass.')
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta, **kwargs)


class AdaMomSPDBatchNorm(SPDBatchNormImpl,SchedulableBatchNorm):
    """
    Adaptive momentum batch normalization on the SPD manifold [proposed].

    The momentum terms can be controlled via a momentum scheduler.
    """
    def __init__(self, shape: Tuple[int, ...] or torch.Size, 
                 batchdim: int,
                 eta=1.0, eta_test=0.1, **kwargs):
        super().__init__(shape=shape, batchdim=batchdim, 
                         eta=eta, eta_test=eta_test, **kwargs)


class DomainSPDBatchNormImpl(BaseDomainBatchNorm):
    """
    Domain-specific batch normalization on the SPD manifold [proposed]

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    """

    domain_bn_cls = None # needs to be overwritten by subclasses

    def __init__(self, shape : Tuple[int,...] or torch.Size, batchdim :int,
                 learn_mean : bool = True, learn_std : bool = True,
                 dispersion : BatchNormDispersion = BatchNormDispersion.NONE,
                 test_stats_mode : BatchNormTestStatsMode = BatchNormTestStatsMode.BUFFER,
                 eta = 1., eta_test = 0.1, domains : Tensor = Tensor([]), **kwargs):
        super().__init__()

        if dispersion == BatchNormDispersion.VECTOR:
            raise NotImplementedError()

        self.dispersion = dispersion
        self.learn_mean = learn_mean
        self.learn_std = learn_std

        manifold= Hyperboloid()  
        init_mean = manifold.zero(shape[-1])  
        #print('init_mean',init_mean.shape, init_mean)

        # if self.learn_mean:
        #     self.mean = ManifoldParameter(init_mean, manifold=Hyperboloid())  #! Hyperbolic
        # else:
        #     self.mean = ManifoldTensor(init_mean, manifold=Hyperboloid()) #! Hyperbolic
        self.mean= init_mean
        #####
        if self.dispersion is BatchNormDispersion.SCALAR:
            init_var = torch.ones((1))           #! Hyperbolic variance initialization
            if self.learn_std:
                self.std = nn.parameter.Parameter(init_var.clone())
                #print('std',self.std.shape, self.std)
            else:
                self.std = init_var.clone()
        else:
            self.std = None
        
        cls = type(self).domain_bn_cls
        for domain in domains:
            self.add_domain_(cls(shape=shape, batchdim=batchdim, 
                                learn_mean=learn_mean,learn_std=learn_std, dispersion=dispersion,
                                mean=self.mean, std=self.std, eta=eta, eta_test=eta_test, **kwargs),
                            domain)

        self.set_test_stats_mode(test_stats_mode)

class AdaMomDomainSPDBatchNorm(DomainSPDBatchNormImpl):
    """
    Combines domain-specific batch normalization on the SPD manifold
    with adaptive momentum batch normalization [Yong et al. 2020, ECCV].

    Keeps running stats for each domain. Scaling and bias parameters are shared across domains.
    The momentum terms can be controlled with a momentum scheduler.
    """

    domain_bn_cls = AdaMomSPDBatchNorm
