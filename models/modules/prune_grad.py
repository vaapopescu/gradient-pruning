from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import pdb

_PRUNING_PERCENTAGE = 70

def random_zeros(x, percent=_PRUNING_PERCENTAGE):
    perm = torch.randperm(x.numel())
    number_of_zeros = int(x.numel() * percent / 100)

    mask = perm[:number_of_zeros].cuda()
    x.flatten()[mask] = 0.0
    
    return x

def smaller_zeros(x, percent=_PRUNING_PERCENTAGE):
    k = max(1, int(x.numel() * percent / 100))
    kth_value, _ = torch.kthvalue(x.flatten().detach().cpu().abs(), k)
    print('k ', k, 'kth ', kth_value)
    x[x.abs() <= kth_value.cuda()] = 0.0

    return x

def statistical_smaller_zeros(x, percent=_PRUNING_PERCENTAGE):
    mask_size = 1000
    k = max(1, int(mask_size * percent / 100))

    perm = torch.randperm(x.numel())
    mask = perm[:mask_size].cuda()
   
    stats = x.flatten()[mask].detach().cpu().abs()
    
    kth_value, _ = torch.kthvalue(stats, k)
    x[x.abs() <= kth_value.cuda()] = 0.0

    return x


class UniformZeroGrad(InplaceFunction):
    @staticmethod
    def forward(ctx, input, percent=_PRUNING_PERCENTAGE):
        ctx.percent = percent
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            grad_input = grad_output.detach().clone()
            
            if grad_output.numel() > 2000:
                grad_input = smaller_zeros(grad_input, ctx.percent)

        return grad_input, None, None, None, None, None, None, None



def conv2d_prune_grad(input, weight, bias=None, stride=1, padding=0, dilation=1, 
                    groups=1, percent=_PRUNING_PERCENTAGE):
    out1 = F.conv2d(input, weight, bias,
                    stride, padding, dilation, groups)
    out2 = prune_grad(out1, percent=percent)
    return out2

def linear_prune_grad(input, weight, bias=None, percent=_PRUNING_PERCENTAGE):
    out1 = F.linear(input, weight, bias)
    out2 = prune_grad(out1, percent=percent)
    return out2

def bn_prune_grad(x, weight, bias, running_mean, running_var, training, 
                 momentum, eps, percent):
    out1 = F.batch_norm(x, running_mean, running_var, weight,
                        bias, training, momentum, eps)
    out2 = prune_grad(out1, percent=percent)
    return out2

def prune_grad(x, percent):
    return UniformZeroGrad().apply(x, percent)


class PConv2d(nn.Conv2d):
    """docstring for ZConv2d."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 percent=_PRUNING_PERCENTAGE):
        super(PConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.percent = percent


    def forward(self, input):
        output = conv2d_prune_grad(input, self.weight, self.bias, self.stride, 
                                  self.padding, self.dilation, self.groups, self.percent)
        return output


class PLinear(nn.Linear):
    """docstring for ZLinear."""
    def __init__(self, in_features, out_features, bias=True,
                 percent=_PRUNING_PERCENTAGE):
        super(PLinear, self).__init__(in_features, out_features, bias)
        self.percent = percent

    def forward(self, input):
        output = linear_prune_grad(input, self.weight, self.bias, self.percent)
        return output


class PBN(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, affine=True, eps=1e-5, 
                 track_running_stats=True, percent=_PRUNING_PERCENTAGE):
        super(PBN, self).__init__(num_features, momentum, affine, eps, track_running_stats)
        self.percent = percent

    def forward(self, x):
        output = bn_prune_grad(x, self.weight, self.bias, self.running_mean, self.running_var,
                              self.training, self.momentum, self.eps, self.percent)
        return output


if __name__ == '__main__':
    x = torch.rand(2, 3, 2, 2)
    percent = 40
    x_z = smaller_zeros(x, percent)

    pdb.set_trace()
    print(x)
    print(x_z)




