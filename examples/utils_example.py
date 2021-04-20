import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_2d(data, noise, samples, shuffle=False):
    if data == 1:
        centers = [(1, 0), (0, 1)]
    elif data == 2:
        centers = [(np.cos(np.pi/3), np.sin(np.pi/3)), (1 ,0)]
    elif data == 3:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4)), (1 ,0)]
    elif data == 4:
        centers = [(np.cos(3*np.pi/4), np.sin(3*np.pi/4)), (1 ,0)]
    elif data == 5:
        centers = [(np.cos(2*np.pi/3), np.sin(2*np.pi/3)), (1 ,0)]
    elif data == 6:
        centers = [(np.cos(3*np.pi/4), np.sin(3*np.pi/4)), (np.cos(4*np.pi/3), np.sin(4*np.pi/3)), (1 ,0)]
    elif data == 7:
        centers = [(np.cos(3*np.pi/4), np.sin(3*np.pi/4)), 
                   (np.cos(4*np.pi/3), np.sin(4*np.pi/3)), 
                   (np.cos(np.pi/4), np.sin(np.pi/4))]
    elif data == 8:
        centers = [(np.cos(np.pi/6), np.sin(np.pi/6)), 
                   (np.cos(np.pi/2), np.sin(np.pi/2)), 
                   (np.cos(3*np.pi/4), np.sin(3*np.pi/4)),
                   (np.cos(5*np.pi/4), np.sin(5*np.pi/4)),
                   (np.cos(7*np.pi/4), np.sin(7*np.pi/4)),
                   (np.cos(3*np.pi/2), np.sin(3*np.pi/2))]
    else:
        raise NameError('data not found.')

    data = []
    targets = []
    for lbl, center in enumerate(centers):
        X = np.random.normal(loc=center, scale=noise, size=(samples, 2))
        y = np.repeat(lbl, samples).tolist()
        data.append(X)
        targets += y
    data = np.concatenate(data)
    data = data / np.linalg.norm(data, axis=1, ord=2, keepdims=True)
    targets = np.array(targets)

    if shuffle:
        idx_arr = np.random.choice(np.arange(len(data)), len(data), replace=False)
        data, targets = data[idx_arr], targets[idx_arr]

    data = torch.tensor(data).float()
    targets = torch.tensor(targets).long()
    return data, targets, len(centers)

def generate_3d(data, noise, samples, shuffle=False):
    if data == 1:
        centers = [(1, 0, 0), 
                   (0, 1, 0), 
                   (0, 0, 1)]
    elif data == 2:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4), 1),
                   (np.cos(2*np.pi/3), np.sin(2*np.pi/3), 1),
                   (np.cos(np.pi), np.sin(np.pi), 1)]
    elif data == 3:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4), 1),
                   (np.cos(2*np.pi/3), np.sin(2*np.pi/3), 1),
                   (np.cos(5*np.pi/6), np.cos(5*np.pi/6), 1)]
    else:
        raise NameError('Data not found.')

    X, Y = [], []
    for c, center in enumerate(centers):
        _X = np.random.normal(center, scale=(noise, noise, noise), size=(samples, 3))
        _Y = np.ones(samples, dtype=np.int32) * c
        X.append(_X)
        Y.append(_Y)
    X = np.vstack(X)
    X = X / np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    Y = np.hstack(Y)
    
    if shuffle:
        idx_arr = np.random.choice(np.arange(len(X)), len(X), replace=False)
        X, Y = X[idx_arr], Y[idx_arr]

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).long()
    return X, Y, 3


def plot_2d(inputs, labels, outputs):
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    for c in labels.unique():
        ax[0].scatter(inputs[:, 0], inputs[:, 1], c=labels)
        ax[0].set_ylim([-1.1, 1.1])
        ax[0].set_xlim([-1.1, 1.1])
        ax[0].set_title('X')
        ax[1].scatter(outputs[:, 0], outputs[:, 1], c=labels)
        ax[1].set_ylim([-1.1, 1.1])
        ax[1].set_xlim([-1.1, 1.1])
        ax[1].set_title('Z')
    fig.tight_layout()
    plt.show()
    plt.close() 


def plot_3d(Z, y, title=''):
    colors = np.array(['green', 'blue', 'red'])
    colors = np.array(['forestgreen', 'royalblue', 'brown'])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors[y], cmap=plt.cm.Spectral, s=200.0)
#     Z, _ = F.get_n_each(Z, y, 1)
#     for c in np.unique(y):
#         ax.quiver(0.0, 0.0, 0.0, Z[c, 0], Z[c, 1], Z[c, 2], length=1.0, normalize=True, arrow_length_ratio=0.05, color='black')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.5)
    ax.xaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.yaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.set_title(title)
    # [tick.label.set_fontsize(24) for tick in ax.xaxis.get_major_ticks()] 
    # [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
    # [tick.label.set_fontsize(24) for tick in ax.zaxis.get_major_ticks()]
    ax.view_init(20, 15)
    plt.tight_layout()
    plt.show()
    plt.close()



def plot_loss_mcr(data):
    loss_total = data['loss_total']
    loss_expd = data['loss_expd']
    loss_comp = data['loss_comp']
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
    plt.show()
    plt.close()