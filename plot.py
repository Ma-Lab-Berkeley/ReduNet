import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import functional as F
import utils

def plot_loss_mcr(model_dir, name):
    file_dir = os.path.join(model_dir, 'loss', f'{name}.csv')
    data = pd.read_csv(file_dir)
    loss_total = data['loss_total'].ravel()
    loss_expd = data['loss_expd'].ravel()
    loss_comp = data['loss_comp'].ravel()
    num_iter = np.arange(len(loss_total))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True)
    ax.plot(num_iter, loss_total, label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_expd, label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_comp, label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    fig.tight_layout()

    loss_dir = os.path.join(model_dir, 'figures', 'loss_mcr')
    os.makedirs(loss_dir, exist_ok=True)
    file_name = os.path.join(loss_dir, f'{name}.png')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))

def plot_loss(model_dir):
    """Plot cross entropy loss. """
    ## extract loss from csv
    file_dir = os.path.join(model_dir, 'losses.csv')
    data = pd.read_csv(file_dir)
    epochs = data['epoch'].ravel()
    loss = data['loss'].ravel()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    ax.plot(epochs, loss, #label=r'Loss', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.set_title("Loss")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    ## create saving directory
    loss_dir = os.path.join(model_dir, 'figures', 'loss')
    os.makedirs(loss_dir, exist_ok=True)
    file_name = os.path.join(loss_dir, 'loss.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(loss_dir, 'loss.pdf')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))

def plot_csv(model_dir, filename):
    df = pd.read_csv(os.path.join(model_dir, f'{filename}.csv'))
    colnames = df.columns
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for colname in colnames[1:]:
        ax.plot(df[colnames[0]], df[colname], marker='x', label=colname)
    ax.set_xlabel(colnames[0])
    ax.set_ylabel(filename)
    ax.legend()

    csv_dir = os.path.join(model_dir, 'figures', 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    savepath = os.path.join(csv_dir, f'{filename}.png')
    fig.savefig(savepath)
    print('Plot saved to: {}'.format(savepath))

def plot_loss_ce(model_dir, filename='loss_ce'):
    df = pd.read_csv(os.path.join(model_dir, f'{filename}.csv'))
    colnames = df.columns
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for colname in colnames[1:]:
        ax.plot(np.arange(df.shape[0]), df['loss_ce'], label=colname)
    ax.set_xlabel(colnames[0])
    ax.set_ylabel(filename)
    ax.legend()

    csv_dir = os.path.join(model_dir, 'figures', 'loss_ce')
    os.makedirs(csv_dir, exist_ok=True)
    savepath = os.path.join(csv_dir, f'loss_ce.png')
    fig.savefig(savepath)
    print('Plot saved to: {}'.format(savepath))


def plot_acc(model_dir):
    """Plot training and testing accuracy"""
    ## extract loss from csv
    file_dir = os.path.join(model_dir, 'acc.csv')
    data = pd.read_csv(file_dir)
    epochs = data['epoch'].ravel()
    acc_train = data['acc_train'].ravel()
    acc_test = data['acc_test'].ravel()
    # epoch,acc_train,acc_test

    ## Theoretical Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    ax.plot(epochs, acc_train, label='train', color='green', alpha=0.8)
    ax.plot(epochs, acc_test, label='test', color='red', alpha=0.8)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    ## create saving directory
    acc_dir = os.path.join(model_dir, 'figures', 'acc')
    os.makedirs(acc_dir, exist_ok=True)
    file_name = os.path.join(acc_dir, 'accuracy.png')
    plt.savefig(file_name, dpi=400)
    print("Plot saved to: {}".format(file_name))
    file_name = os.path.join(acc_dir, 'accuracy.pdf')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))

def plot_heatmap(model_dir, name, features, labels, num_classes):
    """Plot heatmap of cosine simliarity for all features. """
    features_sort, _ = utils.sort_dataset(features, labels, 
                            classes=num_classes, stack=False)
    features_sort_ = np.vstack(features_sort)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)

    # plt.rc('text', usetex=False)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman'] #+ plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, len(labels), num_classes+1))
    ax.set_yticks(np.linspace(0, len(labels), num_classes+1))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()

    save_dir = os.path.join(model_dir, 'figures', 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"{name}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_transform(model_dir, inputs, outputs, name):
    fig, ax = plt.subplots(ncols=2)
    inputs = inputs.permute(1, 2, 0)
    outputs = outputs.permute(1, 2, 0)
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
    ax[0].imshow(inputs)
    ax[0].set_title('inputs')
    ax[1].imshow(outputs)
    ax[1].set_title('outputs')
    save_dir = os.path.join(model_dir, 'figures', 'images')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'{name}.png')
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_channel_image(model_dir, features, name):
    def normalize(x):
        out = x - x.min()
        out = out / (out.max() - out.min())
        return out
    fig, ax = plt.subplots()
    ax.imshow(normalize(features), cmap='gray')
    save_dir = os.path.join(model_dir, 'figures', 'images')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'{name}.png')
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()


def plot_nearest_image(model_dir, image, nearest_images, values, name, grid_size=(4, 4)):
    fig, ax = plt.subplots(*grid_size, figsize=(10, 10))
    idx = 1
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if i == 0 and j == 0:
                ax[i, j].imshow(image)
            else:
                ax[i, j].set_title(values[idx-1])
                ax[i, j].imshow(nearest_images[idx-1])
                idx += 1
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.setp(ax[0, 0].spines.values(), color='red', linewidth=2)
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, 'figures', 'nearest_image')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_image(model_dir, image, name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if image.shape[2] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, 'figures', 'image')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

def save_image(image, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if image.shape[2] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()