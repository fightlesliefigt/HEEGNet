import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
import torch.nn.functional as F

class LorentzAct(nn.Module):
    """ Implementation of a general Lorentz Activation on space components. 
    """
    def __init__(self, activation, manifold: CustomLorentz):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.activation = activation # e.g. torch.relu

    def forward(self, x):
        return self.manifold.lorentz_activation(x, self.activation)
    

class LorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.lorentz_relu(x)
    
class LorentzeLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzeLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.lorentz_elu(x)


class LorentzGlobalAvgPool2d(torch.nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold: CustomLorentz, keep_dim=False):
        super(LorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = self.manifold.centroid(x)
        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x

class LorentzAvgPool2d(nn.Module):
    def __init__(self, manifold: CustomLorentz,kernel_size, stride=None):
        super(LorentzAvgPool2d, self).__init__()
        self.manifold = manifold
        # 处理参数
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
    
    def forward(self, x):
        # input shape: [batch_size, height, width, channels]
        batch_size, height, width, channels = x.shape

        # output dimensions
        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1

        # initialize output tensor
        output = torch.zeros(batch_size, out_height * out_width, channels,
                            dtype=x.dtype, device=x.device)

        # average pooling
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.kernel_size[1]

                # average window
                window = x[:, h_start:h_end, w_start:w_end, :]
                window = window.reshape(batch_size, -1, channels)
                idx = i * out_width + j
                output[:, idx, :] = self.manifold.centroid(window)
        output = output.view(batch_size, out_height, out_width, channels)
        return output




 