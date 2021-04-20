import torch
import torch.nn as nn
from opt_einsum import contract


class MaximalCodingRateReduction(nn.Module): #TODO: fix this
    def __init__(self, eps=0.1, gam=1.):
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gam = gam

    def discrimn_loss(self, Z):
        m, d = Z.shape
        I = torch.eye(d).to(Z.device)
        c = d / (m * self.eps)
        return logdet(c * covariance(Z) + I) / 2.

    def compress_loss(self, Z, Pi):
        loss_comp = 0.
        for j in y.unique():
            Z_j = Z[(y == int(j))[:, 0]]
            m_j = Z_j.shape[0]
            c_j = d / (m_j * eps)
            logdet_j = logdet(I + c_j * Z_j.T @ Z_j)
            loss_comp += logdet_j * m_j / (2 * m)
        return loss_comp

    def forward(self, Z, y):
        Pi = y # TODO: change this to prob distribution
        loss_discrimn = self.discrimn_loss(Z)
        loss_compress = self.compress_loss(Z, Pi)
        return loss_discrimn - self.gam * loss_compress


def compute_mcr2(Z, y, eps):
    if len(Z.shape) == 2:
        loss_func = compute_loss_vec
    elif len(Z.shape) == 3:
        loss_func = compute_loss_1d
    elif len(Z.shape) == 4:
        loss_func = compute_loss_2d
    return loss_func(Z, y, eps)
        
def compute_loss_vec(Z, y, eps):
    m, d = Z.shape
    I = torch.eye(d).to(Z.device)
    c = d / (m * eps)
    loss_expd = logdet(c * covariance(Z) + I) / 2.
    loss_comp = 0.
    for j in y.unique():
        Z_j = Z[(y == int(j))[:, 0]]
        m_j = Z_j.shape[0]
        c_j = d / (m_j * eps)
        logdet_j = logdet(I + c_j * Z_j.T @ Z_j)
        loss_comp += logdet_j * m_j / (2 * m)
    loss_expd, loss_comp = loss_expd.item(), loss_comp.item()
    return loss_expd - loss_comp, loss_expd, loss_comp

def compute_loss_1d(V, y, eps):
    m, C, T = V.shape
    I = torch.eye(C).unsqueeze(-1).to(V.device)
    alpha = C / (m * eps)
    cov = alpha * covariance(V) + I
    loss_expd = logdet(cov.permute(2, 0, 1)).sum() / (2 * T)
    loss_comp = 0.
    for j in y.unique():
        V_j = V[y==int(j)]
        m_j = V_j.shape[0]
        alpha_j = C / (m_j * eps) 
        cov_j = alpha_j * covariance(V_j) + I
        loss_comp += m_j / m * logdet(cov_j.permute(2, 0, 1)).sum() / (2 * T)
    loss_expd, loss_comp = loss_expd.real.item(), loss_comp.real.item()
    return loss_expd - loss_comp, loss_expd, loss_comp

def compute_loss_2d(V, y, eps):
    m, C, H, W = V.shape
    I = torch.eye(C).unsqueeze(-1).unsqueeze(-1).to(V.device)
    alpha = C / (m * eps)
    cov = alpha * covariance(V) + I
    loss_expd = logdet(cov.permute(2, 3, 0, 1)).sum() / (2 * H * W)
    loss_comp = 0.
    for j in y.unique():
        V_j = V[(y==int(j))[:, 0]]
        m_j = V_j.shape[0]
        alpha_j = C / (m_j * eps) 
        cov_j = alpha_j * covariance(V_j) + I
        loss_comp += m_j / m * logdet(cov_j.permute(2, 3, 0, 1)).sum() / (2 * H * W)
    loss_expd, loss_comp = loss_expd.real.item(), loss_comp.real.item()
    return loss_expd - loss_comp, loss_expd, loss_comp

def covariance(X):
    return contract('ji...,jk...->ik...', X, X.conj())

def logdet(X):
    sgn, logdet = torch.linalg.slogdet(X)
    return sgn * logdet