import torch
import torch.nn as nn
from loss import compute_mcr2
from .layers.redulayer import ReduLayer



class ReduNet(nn.Sequential):
    # ReduNet Architecture. This class inherited from nn.Seqeuential class, 
    # hence can be used for stacking layers of torch.nn.Modules. 
    
    def __init__(self, *modules):
        super(ReduNet, self).__init__(*modules)
        self._init_loss()

    def init(self, inputs, labels):
        # Initialize the network. Using inputs and labels, it constructs 
        # the parameters E and Cs throughout each ReduLayer. 
        with torch.no_grad():
            return self.forward(inputs,
                                labels,
                                init=True,
                                loss=True)

    def update(self, inputs, labels, tau=0.1):
        # Update the network parameters E and Cs by
        # performing a moving average. 
        with torch.no_grad():
            return self.forward(inputs, 
                                labels, 
                                tau=tau,
                                update=True,
                                loss=True)

    def zero(self):
        # Set every network parameters E and Cs to a zero matrix.
        with torch.no_grad():
            for module in self:
                if isinstance(module, ReduLayer):
                    module.zero()
        return self

    def batch_forward(self, inputs, batch_size=1000, loss=False, cuda=True, device=None):
        # Perform forward pass in batches.
        outputs = []
        for i in range(0, inputs.shape[0], batch_size):
            print('batch:', i, end='\r')
            batch_inputs = inputs[i:i+batch_size]
            if device is not None:
                batch_inputs = batch_inputs.to(device)
            elif cuda:
                batch_inputs = batch_inputs.cuda()
            batch_outputs = self.forward(batch_inputs, loss=loss)
            outputs.append(batch_outputs.cpu())
        return torch.cat(outputs)

    def forward(self,
                inputs, 
                labels=None,
                tau=0.1, 
                init=False,
                update=False,
                loss=False):

        self._init_loss()
        self._inReduBlock = False
        
        for layer_i, module in enumerate(self):
            # preprocess for redunet layers
            if self._isEnterReduBlock(layer_i, module):
                inputs = module.preprocess(inputs)
                self._inReduBlock = True

            # If init is set to True, then initialize 
            # layer using inputs and labels
            if init and self._isReduLayer(module):
                module.init(inputs, labels)

            # If update is set to True, then initialize 
            # layer using inputs and labels
            if update and self._isReduLayer(module):
                module.update(inputs, labels, tau)

            # Perform a forward pass
            if self._isReduLayer(module):
                inputs, preds = module(inputs, return_y=True)
            else:
                inputs = module(inputs)

            # compute loss for redunet layer
            if loss and isinstance(module, ReduLayer):
                losses = compute_mcr2(inputs, preds, module.eps)
                self._append_loss(layer_i, *losses)

            # postprocess for redunet layers
            if self._isExitReduBlock(layer_i, module):
                inputs = module.postprocess(inputs)
                self._inReduBlock = False
        return inputs


    def get_loss(self):
        return self.losses

    def _init_loss(self):
        self.losses = {'layer': [], 'loss_total':[], 'loss_expd': [], 'loss_comp': []}

    def _append_loss(self, layer_i, loss_total, loss_expd, loss_comp):
        self.losses['layer'].append(layer_i)
        self.losses['loss_total'].append(loss_total)
        self.losses['loss_expd'].append(loss_expd)
        self.losses['loss_comp'].append(loss_comp)
        print(f"{layer_i} | {loss_total:.6f} {loss_expd:.6f} {loss_comp:.6f}")

    def _isReduLayer(self, module):
        return isinstance(module, ReduLayer)

    def _isEnterReduBlock(self, _, module):
        # my first encounter of ReduLayer
        if not self._inReduBlock and self._isReduLayer(module):
            return True
        return False

    def _isExitReduBlock(self, layer_i, _):
        # I am in ReduBlock and I am the last layer of the network
        if len(self) - 1 == layer_i and self._inReduBlock: \
            return True
        # I am in ReduBlock and I am the last ReduLayer
        if self._inReduBlock and not self._isReduLayer(self[layer_i+1]):
            return True
        return False
