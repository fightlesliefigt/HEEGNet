import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected,LorentzGroupedFullyConnected

class LorentzConv1d(nn.Module):
    """ Implements a fully hyperbolic 1D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, bias: Same as nn.Conv1d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt()) 

        patches = x.unfold(1, self.kernel_size, self.stride)
        # Lorentz direct concatenation of features within patches
        patches_time = patches.narrow(2, 0, 1)
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=(-2,-1), keepdim=True) - ((self.kernel_size - 1) * self.manifold.k))
        patches_time_rescaled = patches_time_rescaled.view(bsz, patches.shape[1], -1)

        patches_space = patches.narrow(2, 1, patches.shape[2]-1).reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)

        return out


class LorentzConv2d(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) +1
        #print('lin_features', lin_features)
        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1])) #!初始代码
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]
        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        x = x.permute(0, 3, 1, 2)
        patches = self.unfold(x)  # batch_size, channels × kernel_h × kernel_w, windows (windows的个数是根据stride和padding计算的)
        patches = patches.permute(0, 2, 1) # batch_size, windows, channels × kernel_h × kernel_w
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))
        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) # No need, but seems to improve runtime??    
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)
        out = self.linearized_kernel(patches_pre_kernel)  
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out
    #在h,w上做画分, 在h,w上进行一个group卷积

class LorentzConvGroup2d(nn.Module):
    """ Implements a fully hyperbolic 2D grouped convolutional layer using the Lorentz model.
    
    Groups are only applied to spatial channels, while the time channel is shared across all groups.
    
    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d
        groups: Number of groups for grouped convolution (applied only to spatial channels)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            groups=1,
            LFC_normalize=False
    ):
        super(LorentzConvGroup2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.groups = groups

        if (out_channels-1) % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        if (in_channels-1) % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        
        self.group_in_channels = (in_channels - 1) // groups 
        self.group_out_channels = (out_channels - 1) // groups

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]



        lin_features = self.group_in_channels  * self.kernel_size[0] * self.kernel_size[1]

        self.linearized_kernel = LorentzGroupedFullyConnected(
            manifold,
            lin_features, 
            self.group_out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1])) #!初始代码
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]
        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        # 分离时间和空间维度
        x_space = x[..., 1:]  # 空间维度 [bsz, h, w, in_channels-1]
        
        # 转换为 PyTorch 卷积格式 (batch, channels, height, width)
        x_space = x_space.permute(0, 3, 1, 2)  # [bsz, in_channels-1, h, w]
        
        outputs = []
        
        for i in range(self.groups):
            # 计算当前组的通道范围
            start_ch = i * self.group_in_channels
            end_ch = (i + 1) * self.group_in_channels
            
            # 提取当前组的空间通道
            x_group = x_space[:, start_ch:end_ch, :, :]  # [bsz, group_in_channels, h, w]
            
            # 对当前组进行unfold操作
            patches = self.unfold(x_group)  # [bsz, group_in_channels * kernel_h * kernel_w, windows]
            patches = patches.permute(0, 2, 1)  # [bsz, windows, group_in_channels * kernel_h * kernel_w]
            
            # 通过群组全连接层处理
            out = self.linearized_kernel(patches)  # [bsz, windows, group_out_channels]
            outputs.append(out)
        
        # 合并所有组的输出
        out_space = torch.cat(outputs, dim=-1)  # [bsz, windows, out_channels-1]
        
        # 添加时间维度
        out = self.manifold.add_time(out_space)  # [bsz, windows, out_channels]
        
        # 重新调整形状
        out = out.view(bsz, h_out, w_out, self.out_channels)
        
        return out