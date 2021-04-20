import argparse
import os

import torch
import torch.nn as nn


from redunet import *
import evaluate
import load as L
import functional as F
import utils
import plot



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='choice of dataset')
parser.add_argument('--arch', type=str, required=True, help='choice of architecture')
parser.add_argument('--samples', type=int, required=True, help="number of samples per update")
parser.add_argument('--tail', type=str, default='', help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/', help='base directory for saving.')
parser.add_argument('--data_dir', type=str, default='./data/', help='base directory for saving.')
args = parser.parse_args()

## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## Model Directory
model_dir = os.path.join(args.save_dir, 
                         'forward',
                         f'{args.data}+{args.arch}',
                         f'samples{args.samples}'
                         f'{args.tail}')
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

## Data
trainset, testset, num_classes = L.load_dataset(args.data, data_dir=args.data_dir)
X_train, y_train = F.get_samples(trainset, args.samples)
X_train, y_train = X_train.to(device), y_train.to(device)

## Architecture
net = L.load_architecture(args.data, args.arch)
net = net.to(device)

## Training
with torch.no_grad():
    Z_train = net.init(X_train, y_train)
    losses_train = net.get_loss()
    X_train, Z_train = F.to_cpu(X_train, Z_train)

## Saving
utils.save_loss(model_dir, 'train', losses_train)
utils.save_ckpt(model_dir, 'model', net)

## Plotting
plot.plot_loss_mcr(model_dir, 'train')

print(model_dir)
