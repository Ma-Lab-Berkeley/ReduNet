from redunet import *



def flatten(layers, num_classes):
    net = ReduNet(
        *[Vector(eta=0.5, 
                    eps=0.1, 
                    lmbda=500, 
                    num_classes=num_classes, 
                    dimensions=784
                    ) for _ in range(layers)],
    )
    return net