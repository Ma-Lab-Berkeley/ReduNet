import time
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.distributions as distributions
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate



def get_features(net, trainloader, verbose=True, n=None, device='cpu'):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        up_to (int): pass through network up to a certain layer
        flatten (bool): if True, flatten output
        verbose (bool): shows loading staus bar
    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    total = 0
    with torch.no_grad():
        for _, (batch_imgs, batch_lbls) in enumerate(train_bar):
            batch_imgs = batch_imgs.to(device)
            batch_features = net(batch_imgs)
            features.append(batch_features.cpu().detach())
            labels.append(batch_lbls)
            total += len(batch_features)
            if n is not None and  total > n:
                break
    return torch.cat(features)[:n], torch.cat(labels)[:n]

def get_samples(dataset, num_samples, shuffle=False, batch_idx=0, seed=0, method='uniform'):
    if method == 'uniform':
        np.random.seed(seed)
        dataloader = DataLoader(dataset, batch_size=dataset.data.shape[0])
        X, y = next(iter(dataloader))
        if shuffle: # ensure you sample different samples
            idx_arr = np.random.choice(X.shape[0], y.shape[0], replace=False)
            X, y = X[idx_arr], y[idx_arr]
        if num_samples is not None:
            X, y = get_n_each(X, y, num_samples, batch_idx)
        X, y = X.float(), y.long()
        if len(X.shape) == 3:
            X = X.unsqueeze(1)
        return X.float(), y.long()
    elif method == 'first':
        num_classes = torch.unique(dataset.targets).size()[0]
        return next(iter(DataLoader(dataset, batch_size=num_classes*num_samples)))

def normalize(X, p=2):
    if isinstance(X, torch.Tensor):
        norm = torch.linalg.norm(X.flatten(1), ord=p, axis=1)
        norm = norm.clip(min=1e-8)
        for _ in range(len(X.shape)-1):
            norm = norm.unsqueeze(-1)
        return X / norm
    elif isinstance(X, np.ndarray):
        norm = np.linalg.norm(X.reshape(X.shape[0], -1), ord=p, axis=1)
        norm = np.clip(norm, a_min=1e-8, a_max=None)
        for _ in range(len(X.shape)-1):
            norm = np.expand_dims(norm, -1)
        return X / norm
    else:
        raise TypeError('Input array not instances of torch.Tensor or np.ndarray')

def get_n_each(X, y, n=None, batch_idx=0):
    classes = torch.unique(y)
    _X, _y = [], []
    for c in classes:
        idx_class = (y == c)
        X_class, y_class = X[idx_class], y[idx_class]
        if n is not None:
            X_class = torch.roll(X_class, -batch_idx*n, dims=0)
            y_class = torch.roll(y_class, -batch_idx*n, dims=0)
        _X.append(X_class[:n])
        _y.append(y_class[:n])
    return torch.cat(_X), torch.cat(_y)

def translate(data, labels, stride=7, n=None):
    if len(data.shape) == 3:
        return translate1d(data, labels, n=n, stride=stride)
    if len(data.shape) == 4:
        return translate2d(data, labels, n=n, stride=stride)
    raise ValueError('translate not available.')
        

def translate1d(data, labels, n=None, stride=1):
    m, _, T = data.shape
    data_new = []
    if n is None:
        shifts = range(0, T, stride)
    else:
        shifts = range(-n*stride, (n+1)*stride, stride)
    for t in shifts:
        data_new.append(torch.roll(data, t, dims=(2)))
    nrepeats = len(range(0, T, stride))
    return (torch.cat(data_new), 
            labels.repeat(nrepeats))

def translate2d(data, labels, n=None, stride=1):
    m, _, H, W = data.shape
    if n is None:
        shifts_horizontal = range(0, H, stride)
        shifts_vertical = range(0, H, stride)
    else:
        shifts_horizontal = range(-n*stride, (n+1)*stride, stride)
        shifts_vertical = range(-n*stride, (n+1)*stride, stride)
    data_new = []
    for h in shifts_horizontal:
        for w in shifts_vertical:
            data_new.append(torch.roll(data, (h, w), dims=(2, 3)))
    nrepeats = len(shifts_vertical) * len(shifts_horizontal)
    return (torch.cat(data_new), 
            labels.repeat(nrepeats))

def cart2polar(images_cart, channels, timesteps):
    m, C, H, W = images_cart.shape
    mid_pt = int(H // 2)
    R = torch.linspace(0, mid_pt, channels).long()
    thetas = torch.linspace(0, 360, timesteps).float()
    images_polar = []
    for theta in thetas:
        image_rotated = TF.rotate(images_cart, theta.item())
        images_polar.append(image_rotated[:, :, mid_pt, R])
    return torch.cat(images_polar, axis=1).transpose(1, 2)

def step_lr(epochs, init, gamma, steps):
    """learning rate decay
    epochs: total number of epochs
    gamma: multiplicative decay
    step: decay at which steps
    init: initial learning rate
    """
    rates = np.ones(epochs) * init
    for step in steps:
        rates[step:] = rates[step:] * gamma
    return rates

def corrupt_labels(trainset, num_classes, ratio, seed):
    """Corrupt labels in trainset.
    
    Parameters:
        trainset (torch.data.dataset): trainset where labels is stored
        ratio (float): ratio of labels to be corrupted. 0 to corrupt no labels; 
                            1 to corrupt all labels
        seed (int): random seed for reproducibility
        
    Returns:
        trainset (torch.data.dataset): trainset with updated corrupted labels
        
    """

    np.random.seed(seed)
    train_labels = np.asarray(trainset.targets)
    n_train = len(train_labels)
    n_rand = int(len(trainset.data)*ratio)
    randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
    train_labels[randomize_indices] = np.random.choice(np.arange(num_classes), size=n_rand, replace=True)
    trainset.targets = torch.tensor(train_labels).int()
    return trainset

def to_cpu(*gpu_vars):
    cpu_vars = []
    for var in gpu_vars:
        cpu_vars.append(var.detach().cpu())
    return cpu_vars
