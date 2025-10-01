import torch
import torch.nn as nn

from Geometry.hyperbolic import Hyperboloid
from nets.GyroBNBase import GyroBNBase

class GyroBNH1D(GyroBNBase):
    """
    GyroBN layer for hyperbolic spaces (Poincare, Hyperboloid, or Klein models).

    Note: This implementation only supports input of shape [batch_size, dim].

    Args:
        shape (list[int]): Manifold dimensions, e.g., [d] for vectors in L^d, which are represented as vectors in R^{d+1}.
        model (str, default='Poincare'): Type of hyperbolic model. One of {"Poincare", "Hyperboloid", "Klein"}.
        K (float, default=-1.0): Negative curvature of the hyperbolic space.
        max_iter (int, default=1000): Maximum iterations for computing the Fréchet mean.
    """

    def __init__(self, shape, model='Hyperboloid', K=-1.0, max_iter=1000, momentum=0.1, translate="Left",
                 track_running_stats=True, init_1st_batch=False,eps=1e-6):
        super().__init__(shape=shape, batchdim=[0], momentum=momentum, translate=translate,
                         track_running_stats=track_running_stats, init_1st_batch=init_1st_batch, eps=eps)
        self.model = model;self.K=K;self.max_iter=max_iter
        self.get_manifold()
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()

    def get_manifold(self):
        classes = {
            "Hyperboloid": Hyperboloid,
        }
        self.manifold = classes[self.model](K=self.K)

    def set_parameters(self):
        mean_shape, var_shape = self._get_param_shapes()
        self.weight = nn.Parameter(self.manifold.zero_tan(mean_shape))
        self.shift = nn.Parameter(torch.ones(*var_shape))

    def register_running_stats(self):
        """
        In [a], they use the following for the variance:
            Initialization:
                self.updates = 0
            updates:
                self.running_var = (1 - 1 / self.updates) * self.running_var + batch_var / self.updates
                self.updates += 1
        But we follow the standard Euclidean BN.
        [a] Differentiating through the Fréchet Mean
        """
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            running_mean = self.manifold.zero(mean_shape)
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)

    def forward(self, x):
        if self.training:
            print('input shape',x.shape)
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
            print(f"Input mean shape: {input_mean.shape}, Input variance shape: {input_var}")
            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
        else:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
        output = self.normalization(x,input_mean,input_var)
        return output

    def normalization(self,x,input_mean,input_var):
        # set parameters
        print(f"Weight shape: {self.weight.shape}")
        weight = self.manifold.exp0(self.weight)
        print(f"Weight shape: {weight.shape}")
        # centering
        inv_input_mean = self.manifold.gyroinv(input_mean)
        if torch.isnan(inv_input_mean).any():
            print("Warning: input_mean contains NaN values!")    
            print("Number of NaN:", torch.isnan(inv_input_mean).sum().item()) 
        print(f"Inverse input mean shape: {inv_input_mean.shape},value: {inv_input_mean}")
        x_center = self.manifold.gyrotrans(inv_input_mean,x,translate=self.translate)
        print(f"Centered x shape: {x_center.shape}")
        # shifting
        factor = self.shift / (input_var + self.eps).sqrt()
        print(f"Factor shape: {factor.shape},factor: {factor}")
        x_scaled = self.manifold.gyroscalarprod(x_center,factor)
        print(f"Scaled x shape: {x_scaled.shape}")
        # biasing
        x_normed = self.manifold.gyrotrans(weight, x_scaled,translate=self.translate)
        return x_normed

class GyroBNH2D(GyroBNBase):
    """
    GyroBN layer for hyperbolic spaces (Poincare, Hyperboloid, or Klein models).

    Note: This implementation only supports input of shape [batch_size, dim].

    Args:
        shape (list[int]): Manifold dimensions, e.g., [d] for vectors in L^d, which are represented as vectors in R^{d+1}.
        model (str, default='Poincare'): Type of hyperbolic model. One of {"Poincare", "Hyperboloid", "Klein"}.
        K (float, default=-1.0): Negative curvature of the hyperbolic space.
        max_iter (int, default=1000): Maximum iterations for computing the Fréchet mean.
    """

    def __init__(self, shape, model='Hyperboloid', K=-1.0, max_iter=1000, momentum=0.1, translate="Left",
                 track_running_stats=True, init_1st_batch=False,eps=1e-6):
        super().__init__(shape=shape, batchdim=[0], momentum=momentum, translate=translate,
                         track_running_stats=track_running_stats, init_1st_batch=init_1st_batch, eps=eps)
        self.model = model;self.K=K;self.max_iter=max_iter
        self.get_manifold()
        self.set_parameters()
        if self.track_running_stats:
            self.register_running_stats()

    def get_manifold(self):
        classes = {
            "Hyperboloid": Hyperboloid,
        }
        self.manifold = classes[self.model](K=self.K)

    def set_parameters(self):
        mean_shape, var_shape = self._get_param_shapes()
        self.weight = nn.Parameter(self.manifold.zero_tan(mean_shape))
        self.shift = nn.Parameter(torch.ones(*var_shape))

    def register_running_stats(self):
        """
        In [a], they use the following for the variance:
            Initialization:
                self.updates = 0
            updates:
                self.running_var = (1 - 1 / self.updates) * self.running_var + batch_var / self.updates
                self.updates += 1
        But we follow the standard Euclidean BN.
        [a] Differentiating through the Fréchet Mean
        """
        if self.init_1st_batch:
            self.running_mean = None
            self.running_var = None
        else:
            mean_shape, var_shape = self._get_param_shapes()
            running_mean = self.manifold.zero(mean_shape)
            running_var = torch.ones(*var_shape)
            self.register_buffer('running_mean', running_mean)
            self.register_buffer('running_var', running_var)

    def forward(self, x):
        bs, h, w,c = x.shape
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        if nan_count > 0 or inf_count > 0:
            print(f"Warning: x contains {nan_count} NaN values and {inf_count} inf values")
            print(f"Total elements in x: {x.numel()}")       
        x = x.view(-1,c)   #reshape to [bs, h*w, c]       
        if self.training:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
            if self.track_running_stats:
                self.updating_running_statistics(input_mean, input_var)
        elif self.track_running_stats:
            input_mean = self.running_mean
            input_var = self.running_var
        else:
            input_mean = self.manifold.frechet_mean(x,max_iter=self.max_iter)
            input_var = self.manifold.frechet_variance(x, input_mean)
        # if input_var > 0.35:
        #     input_var = 0.35
        output = self.normalization(x,input_mean,input_var)
        # ok, reason = self.manifold._check_point_on_manifold(output)
        # if not ok:
        #     print(f"Warning: output is not on the manifold, input_var:{input_var}")
        # else:
        #     print(f"Output is on the manifold, input_var:{input_var}")
        output = output.view(bs, h, w, c)
        return output

    def normalization(self,x,input_mean,input_var):
        # set parameters
        weight = self.manifold.exp0(self.weight)
        # centering
        inv_input_mean = self.manifold.gyroinv(input_mean)
        x_center = self.manifold.gyrotrans(inv_input_mean,x,translate=self.translate)
        # shifting
        factor = self.shift / (input_var + self.eps).sqrt()
        x_scaled = self.manifold.gyroscalarprod(x_center,factor)
        # biasing
        x_normed = self.manifold.gyrotrans(weight, x_scaled,translate=self.translate)
        return x_normed