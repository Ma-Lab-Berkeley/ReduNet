from redunet import *



def lift2d(channels, layers, num_classes, seed=0):
    net = ReduNet(
        Lift2D(1, channels, 9, seed=seed),
        *[Fourier2D(eta=0.5, 
                    eps=0.1, 
                    lmbda=500, 
                    num_classes=num_classes, 
                    dimensions=(channels, 28, 28)
                    ) for _ in range(layers)],
    )
    return net
