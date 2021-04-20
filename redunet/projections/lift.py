import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions



class Lift(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, init_mode='gaussian1.0', stride=1, trainable=False, relu=True, seed=0):
        super(Lift, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.init_mode = init_mode
        self.stride = stride
        self.trainable = trainable
        self.relu = relu
        self.seed = seed

    def set_weight(self, init_mode, size, trainable):
        torch.manual_seed(self.seed)
        if init_mode == 'gaussian0.1':
            p = distributions.normal.Normal(loc=0, scale=0.1)
        elif init_mode == 'gaussian1.0':
            p = distributions.normal.Normal(loc=0, scale=1.)  
        elif init_mode == 'gaussian5.0':
            p = distributions.normal.Normal(loc=0, scale=5.)  
        elif init_mode == 'uniform0.1':
            p = distributions.uniform.Uniform(-0.1, 0.1)
        elif init_mode == 'uniform0.5':
            p = distributions.uniform.Uniform(-0.5, 0.5)
        else:
            raise NameError(f'No such kernel: {init_mode}')
        kernel = p.sample(size).type(torch.float)
        self.kernel = nn.Parameter(kernel, requires_grad=trainable)


class Lift1D(Lift):
    def __init__(self, in_channel, out_channel, kernel_size, init_mode='gaussian1.0', stride=1, trainable=False, relu=True, seed=0):
        super(Lift1D, self).__init__(in_channel, out_channel, kernel_size, init_mode, stride, trainable, relu, seed)
        self.size = (out_channel, in_channel, kernel_size)
        self.set_weight(init_mode, self.size, trainable)
    
    def forward(self, Z):
        Z = F.pad(Z, (0, self.kernel_size-1), 'circular')
        out = F.conv1d(Z, self.kernel, stride=self.stride)
        if self.relu:
            return F.relu(out)
        return out


class Lift2D(Lift):
    def __init__(self, in_channel, out_channel, kernel_size, init_mode='gaussian1.0', stride=1, trainable=False, relu=True, seed=0):
        super(Lift2D, self).__init__(in_channel, out_channel, kernel_size, init_mode, stride, trainable, relu, seed)
        self.size = (out_channel, in_channel, kernel_size, kernel_size)
        self.set_weight(init_mode, self.size, trainable)

    def forward(self, Z):
        kernel = self.kernel.to(Z.device)
        Z = F.pad(Z, (0, self.kernel_size-1, 0, self.kernel_size-1), 'circular')
        out = F.conv2d(Z, kernel, stride=self.stride)
        if self.relu:
            return F.relu(out)
        return out
