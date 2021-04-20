from .layers.vector import Vector
from .layers.fourier1d import Fourier1D
from .layers.fourier2d import Fourier2D
from .redunet import ReduNet




def ReduNetVector(num_classes, num_layers, d, eta, eps, lmbda):
    redunet = ReduNet(
        *[Vector(eta, eps, lmbda, num_classes, d) for _ in range(num_layers)]
    )
    return redunet

def ReduNet1D(num_classes, num_layers, channels, timesteps, eta, eps, lmbda):
    redunet = ReduNet(
        *[Fourier1D(eta, eps, lmbda, num_classes, (channels, timesteps)) for _ in range(num_layers)]
    )
    return redunet

def ReduNet2D(num_classes, num_layers, channels, height, width, eta, eps, lmbda):
    redunet = ReduNet(
        *[Fourier2D(eta, eps, lmbda, num_classes, (channels, height, width)) for _ in range(num_layers)]
    )
    return redunet