import time
import os
from os import listdir
from os.path import join, isfile, isdir, expanduser
from tqdm import tqdm

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import functional as F
from redunet import *


def load_architecture(data, arch, seed=0):
    if data == 'mnist2d':
        if arch == 'lift2d_channels35_layers5':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=35, layers=5, num_classes=10, seed=seed)
        if arch == 'lift2d_channels35_layers10':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=35, layers=5, num_classes=10, seed=seed)
        if arch == 'lift2d_channels35_layers20':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=35, layers=20, num_classes=10, seed=seed)
        if arch == 'lift2d_channels55_layers5':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=55, layers=5, num_classes=10, seed=seed)
        if arch == 'lift2d_channels55_layers10':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=55, layers=5, num_classes=10, seed=seed)
        if arch == 'lift2d_channels55_layers20':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=55, layers=20, num_classes=10, seed=seed)
    if data == 'mnist2d+2class':
        if arch == 'lift2d_channels35_layers5':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=35, layers=5, num_classes=2, seed=seed)
        if arch == 'lift2d_channels35_layers10':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=35, layers=5, num_classes=2, seed=seed)
        if arch == 'lift2d_channels35_layers20':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=35, layers=20, num_classes=2, seed=seed)
        if arch == 'lift2d_channels55_layers5':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=55, layers=5, num_classes=2, seed=seed)
        if arch == 'lift2d_channels55_layers10':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=55, layers=5, num_classes=2, seed=seed)
        if arch == 'lift2d_channels55_layers20':
            from architectures.mnist.lift2d import lift2d
            return lift2d(channels=55, layers=20, num_classes=2, seed=seed)
    if data == 'mnistvector':
        if arch == 'layers50':
            from architectures.mnist.flatten import flatten
            return flatten(layers=50, num_classes=10)
        if arch == 'layers20':
            from architectures.mnist.flatten import flatten
            return flatten(layers=20, num_classes=10)
        if arch == 'layers10':
            from architectures.mnist.flatten import flatten
            return flatten(layers=10, num_classes=10)
        if arch == 'layers5':
            from architectures.mnist.flatten import flatten
            return flatten(layers=5, num_classes=10)
    if data == 'mnistvector_2class':
        if arch == 'layers50':
            from architectures.mnist.flatten import flatten
            return flatten(layers=50, num_classes=2)
        if arch == 'layers20':
            from architectures.mnist.flatten import flatten
            return flatten(layers=20, num_classes=2)
        if arch == 'layers10':
            from architectures.mnist.flatten import flatten
            return flatten(layers=10, num_classes=2)
        if arch == 'layers5':
            from architectures.mnist.flatten import flatten
            return flatten(layers=5, num_classes=2)
    raise NameError('Cannot find architecture: {}.')

def load_dataset(choice, data_dir='./data/'):
    if choice == 'mnist2d':
        from datasets.mnist import mnist2d_10class
        return mnist2d_10class(data_dir)
    if choice == 'mnist2d_2class':
        from datasets.mnist import mnist2d_2class
        return mnist2d_2class(data_dir)
    if choice =='mnistvector':
        from datasets.mnist import mnistvector_10class
        return mnistvector_10class(data_dir)
    raise NameError(f'Dataset {choice} not found.')

