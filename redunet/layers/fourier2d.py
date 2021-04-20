import time
import torch
import torch.nn as nn
import torch.nn.functional as tF
from itertools import product
from torch.fft import fft2, ifft2
import numpy as np
from opt_einsum import contract

import functional as F
from ..multichannel_weight import MultichannelWeight
from .vector import Vector



class Fourier2D(Vector):
    def __init__(self, eta, eps, lmbda, num_classes, dimensions):
        super(Fourier2D, self).__init__(eta, eps, lmbda, num_classes)
        assert len(dimensions) == 3, 'dimensions should have tensor dim = 3'
        self.channels, self.height, self.width = dimensions
        self.gam = nn.Parameter(torch.ones(num_classes) / num_classes, requires_grad=False)
        self.E = MultichannelWeight(*dimensions, dtype=torch.complex64)
        self.Cs = nn.ModuleList([MultichannelWeight(*dimensions, dtype=torch.complex64) 
                                        for _ in range(num_classes)])

    def nonlinear(self, Cz):
        norm = torch.linalg.norm(Cz.flatten(start_dim=2), axis=2).clamp(min=1e-8)
        pred = tF.softmax(-self.lmbda * norm, dim=0).unsqueeze(2)
        y = torch.argmax(pred, axis=0) #TODO: for non argmax case
        gam = self.gam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pred = pred.unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(gam * Cz * pred, axis=0)
        return out, y

    def compute_E(self, V):
        m, C, H, W = V.shape
        alpha = C / (m * self.eps)
        I = torch.eye(C, device=V.device).unsqueeze(-1).unsqueeze(-1)
        pre_inv = I + alpha * contract('ji...,jk...->ik...', V, V.conj())
        E = torch.empty_like(pre_inv, dtype=torch.complex64)
        for h, w in product(range(H), range(W)):
            E[:, :, h, w] = alpha * torch.inverse(pre_inv[:, :, h, w])
        return E
    
    def compute_Cs(self, V, y):
        m, C, H, W = V.shape
        I = torch.eye(C, device=V.device).unsqueeze(-1).unsqueeze(-1)
        Cs = torch.empty((self.num_classes, C, C, H, W), dtype=torch.complex64)
        for j in range(self.num_classes):
            V_j = V[(y == int(j))]
            m_j = V_j.shape[0]
            alpha_j = C / (m_j * self.eps)
            pre_inv = I + alpha_j * contract('ji...,jk...->ik...', V_j, V_j.conj())
            for h, w in product(range(H), range(W)):
                Cs[j, :, :, h, w] = alpha_j * torch.inverse(pre_inv[:, :, h, w])
        return Cs

    def preprocess(self, X):
        Z = F.normalize(X)
        return fft2(Z, norm='ortho', dim=(2, 3))

    def postprocess(self, X):
        Z = ifft2(X, norm='ortho', dim=(2, 3))
        return F.normalize(Z).real
