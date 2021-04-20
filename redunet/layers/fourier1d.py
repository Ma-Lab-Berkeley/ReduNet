import torch
import torch.nn as nn
import torch.nn.functional as tF
from torch.fft import fft, ifft
from opt_einsum import contract
import numpy as np

import functional as F
from ..multichannel_weight import MultichannelWeight
from .vector import Vector



class Fourier1D(Vector):
    def __init__(self, eta, eps, lmbda, num_classes, dimensions):
        super(Fourier1D, self).__init__(eta, eps, lmbda, num_classes)
        assert len(dimensions) == 2, 'dimensions should have tensor dim = 2'
        self.channels, self.timesteps = dimensions
        self.gam = nn.Parameter(torch.ones(num_classes) / num_classes, requires_grad=False)
        self.E = MultichannelWeight(self.channels, self.timesteps, dtype=torch.complex64)
        self.Cs = nn.ModuleList([MultichannelWeight(self.channels, self.timesteps, dtype=torch.complex64) 
                                        for _ in range(num_classes)])

    def nonlinear(self, Bz):
        norm = torch.linalg.norm(Bz.reshape(Bz.shape[0], Bz.shape[1], -1), axis=2)
        norm = torch.clamp(norm, min=1e-8)
        pred = tF.softmax(-self.lmbda * norm, dim=0).unsqueeze(2)
        y = torch.argmax(pred, axis=0) #TODO: for non argmax case
        gam = self.gam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pred = pred.unsqueeze(-1)
        out = torch.sum(gam * Bz * pred, axis=0)
        return out, y

    def compute_E(self, V):
        m, C, T = V.shape
        alpha = C / (m * self.eps)
        I = torch.eye(C, device=V.device).unsqueeze(-1)
        pre_inv = I + alpha * contract('ji...,jk...->ik...', V, V.conj())
        E = torch.empty_like(pre_inv)
        for t in range(T):
            E[:, :, t] = alpha * torch.inverse(pre_inv[:, :, t])
        return E
    
    def compute_Cs(self, V, y):
        m, C, T = V.shape
        I = torch.eye(C, device=V.device).unsqueeze(-1)
        Cs = torch.empty((self.num_classes, C, C, T), dtype=torch.complex64)
        for j in range(self.num_classes):
            V_j = V[(y == int(j))]
            m_j = V_j.shape[0]
            alpha_j = C / (m_j * self.eps)
            pre_inv = I + alpha_j * contract('ji...,jk...->ik...', V_j, V_j.conj())
            for t in range(T):
                Cs[j, :, :, t] = alpha_j * torch.inverse(pre_inv[:, :, t])
        return Cs

    def preprocess(self, X):
        Z = F.normalize(X)
        return fft(Z, norm='ortho', dim=2)

    def postprocess(self, X):
        Z = ifft(X, norm='ortho', dim=2)
        return F.normalize(Z).real
