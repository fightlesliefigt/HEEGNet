import torch as th
import torch.nn as nn

class GyroBNBase(nn.Module):
    """
    Base class for Gyrogroup Batch Normalization (GyroBN) on manifolds.

    Args:
        shape (list[int]): Shape of the input excluding batch dimensions (e.g., [d] or [c, d, d]).
        batchdim (list[int], default=[0]): Indices of batch dimensions to normalize over.
        momentum (float, default=0.1): Momentum for updating running stats.
        translate (str, default='Left'): Use 'Left' or 'Right' gyrotranslation for centering/biasing.
        track_running_stats (bool, default=True): Maintain running mean/var across batches.
        init_1st_batch (bool, default=False): Use the first batch to initialize running stats.
        eps (float, default=1e-6): Small value added to variance for stability.

    Note:
        Subclasses must implement:
            - get_manifold(): define self.manifold
            - set_parameters(): define self.weight and self.shift
            - register_running_stats(): register running_mean and running_var buffers
    """
    def __init__(self,shape, batchdim=[0], momentum=0.1, translate='Left',
                 track_running_stats=True, init_1st_batch=False,eps=1e-6):
        super().__init__()
        self.shape=shape
        self.batchdim=batchdim
        self.momentum = momentum
        self.eps=eps
        self.translate=translate
        self.track_running_stats = track_running_stats
        self.init_1st_batch = init_1st_batch

        # Require customized set_parameters, register_running_stats, get_manifold

        # # Handle channel vs non-channel case
        # if len(self.shape) > 2:
        #     # --- running statistics ---
        #     self.register_buffer("running_mean", th.eye(self.shape[-1]).repeat(*self.shape[:-2], 1, 1))
        #     self.register_buffer("running_var", th.ones(*shape[:-2], 1, 1))
        #     # --- parameters ---
        #     self.shift = nn.Parameter(th.ones(*shape[:-2], 1, 1))
        # else:
        #     # --- running statistics ---
        #     self.register_buffer("running_mean", th.eye(self.shape[-1]))
        #     self.register_buffer("running_var", th.ones(1))
        #     # --- parameters ---
        #     self.shift = nn.Parameter(th.ones(1))

    def _get_param_shapes(self):
        shape_prefix = list(self.shape[:-1])
        mean_shape = shape_prefix + [self.manifold.dim_to_sh(self.shape[-1])]
        var_shape = shape_prefix + [1] if shape_prefix else [1]
        return mean_shape, var_shape

    def set_parameters(self):
        raise NotImplementedError

    def register_running_stats(self):
        raise NotImplementedError

    def get_manifold(self):
        raise NotImplementedError

    def updating_running_statistics(self, batch_mean, batch_var):
        if self.running_mean is None:
            self.running_mean = batch_mean.detach()
        else:
            self.running_mean = self.manifold.geodesic(self.running_mean, batch_mean,self.momentum)

        if self.running_var is None:
            self.running_var = batch_var.detach()
        else:
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

    def __repr__(self):
        attributes = []
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, th.Tensor):
                attributes.append(f"{key}.shape={tuple(value.shape)}")  # only show Tensor shape
            else:
                attributes.append(f"{key}={value}")

        # register_buffer
        for name, buffer in self.named_buffers(recurse=False):
            attributes.append(f"{name}={buffer.item() if buffer.numel() == 1 else tuple(buffer.shape)}")

        return f"{self.__class__.__name__}({', '.join(attributes)}) \n {self.manifold}"
