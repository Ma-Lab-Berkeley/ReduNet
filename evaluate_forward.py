import argparse
import os

import torch
import torch.nn as nn

from redunet import *
import evaluate
import functional as F
import load as L
import utils
import plot



parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, help='model directory')
parser.add_argument('--loss', default=False, action='store_true', help='set to True if plot loss')
parser.add_argument('--trainsamples', type=int, default=None, help="number of train samples in each class")
parser.add_argument('--testsamples', type=int, default=None, help="number of train samples in each class")
parser.add_argument('--translatetrain', default=False, action='store_true', help='set to True if translation train samples')
parser.add_argument('--translatetest', default=False, action='store_true', help='set to True if translation test samples')
parser.add_argument('--batch_size', type=int, default=100, help='batch size for evaluation')
args = parser.parse_args()

## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

## Setup
eval_dir = os.path.join(args.model_dir, 
                        f'trainsamples{args.trainsamples}'
                        f'_testsamples{args.testsamples}'
                        f'_translatetrain{args.translatetrain}'
                        f'_translatetest{args.translatetest}')
params = utils.load_params(args.model_dir)

## Data
trainset, testset, num_classes = L.load_dataset(params['data'], data_dir=params['data_dir'])
X_train, y_train = F.get_samples(trainset, args.trainsamples)
X_test, y_test = F.get_samples(testset, args.testsamples)
if args.translatetrain:
    X_train, y_train = F.translate(X_train, y_train, stride=7)
if args.translatetest:
    X_test, y_test = F.translate(X_test, y_test, stride=7)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

## Architecture
net = L.load_architecture(params['data'], params['arch'])
net = utils.load_ckpt(args.model_dir, 'model', net)
net = net.to(device)

## Forward
with torch.no_grad():
    print('train')
    Z_train = net.batch_forward(X_train, batch_size=args.batch_size, loss=args.loss, device=device)
    X_train, y_train, Z_train = F.to_cpu(X_train, y_train, Z_train)
    utils.save_loss(eval_dir, f'train', net.get_loss())

    print('test')
    Z_test = net.batch_forward(X_test, batch_size=args.batch_size, loss=args.loss, device=device)
    X_test, y_test, Z_test = F.to_cpu(X_test, y_test, Z_test)
    utils.save_loss(eval_dir, f'test', net.get_loss())

## Normalize
X_train = F.normalize(X_train.flatten(1))
X_test = F.normalize(X_test.flatten(1))
Z_train = F.normalize(Z_train.flatten(1))
Z_test = F.normalize(Z_test.flatten(1))

# Evaluate
evaluate.evaluate(eval_dir, 'knn', Z_train, y_train, Z_test, y_test)
#evaluate.evaluate(eval_dir, 'nearsub', Z_train, y_train, Z_test, y_test, num_classes=num_classes, n_comp=10)

# Plot
plot.plot_loss_mcr(eval_dir, 'train')
plot.plot_loss_mcr(eval_dir, 'test')
plot.plot_heatmap(eval_dir, 'X_train', X_train, y_train, num_classes)
plot.plot_heatmap(eval_dir, 'X_test', X_test, y_test, num_classes)
plot.plot_heatmap(eval_dir, 'Z_train', Z_train, y_train, num_classes)
plot.plot_heatmap(eval_dir, 'Z_test', Z_test, y_test, num_classes)

