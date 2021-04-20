import torch
import torch.nn as nn
import torch.nn.functional as tF

import functional as F
from .redulayer import ReduLayer


class Vector(ReduLayer):
    def __init__(self, eta, eps, lmbda, num_classes, dimensions=None):
        super(Vector, self).__init__()
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda
        self.num_classes = num_classes
        self.d = dimensions

        if self.d is not None: #NOTE: initilaized in child objectss
            self.gam = nn.Parameter(torch.ones(num_classes) / num_classes)
            self.E = nn.Linear(self.d, self.d, bias=False)
            self.Cs = nn.ModuleList([nn.Linear(self.d, self.d, bias=False) for _ in range(num_classes)])

    def forward(self, Z, return_y=False):
        expd = self.E(Z)
        comp = torch.stack([C(Z) for C in self.Cs])
        clus, y_approx = self.nonlinear(comp)
        Z = Z + self.eta * (expd - clus)
        Z = F.normalize(Z)
        if return_y:
            return Z, y_approx
        return Z

    def nonlinear(self, Cz):
        norm = torch.linalg.norm(Cz.reshape(Cz.shape[0], Cz.shape[1], -1), axis=2)
        norm = torch.clamp(norm, min=1e-8)
        pred = tF.softmax(-self.lmbda * norm, dim=0).unsqueeze(2)
        y = torch.argmax(pred, axis=0) #TODO: for non argmax case
        gam = self.gam.unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(gam * Cz * pred, axis=0)
        return out, y

    def compute_gam(self, X, y):
        m = X.shape[0]
        m_j = [torch.nonzero(y==j).size()[0] for j in range(self.num_classes)]
        gam = (torch.tensor(m_j).float() / m).flatten()
        return gam

    def compute_E(self, X):
        m, d = X.shape
        Z = X.T
        I = torch.eye(d, device=X.device)
        c = d / (m * self.eps)
        E = c * torch.inverse(I + c * Z @ Z.T)
        return E

    def compute_Cs(self, X, y):
        m, d = X.shape
        Z = X.T
        I = torch.eye(d, device=X.device)
        Cs = torch.zeros((self.num_classes, d, d))
        for j in range(self.num_classes):
            idx = (y == int(j))
            Z_j = Z[:, idx]
            m_j = Z_j.shape[1]
            c_j = d / (m_j * self.eps)
            Cs[j] = c_j * torch.inverse(I + c_j * Z_j @ Z_j.T)
        return Cs

    def preprocess(self, X):
        return F.normalize(X)

    def postprocess(self, X):
        return F.normalize(X)

