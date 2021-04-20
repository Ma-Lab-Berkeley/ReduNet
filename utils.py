import os
import json
import numpy as np
import pandas as pd
import torch



def sort_dataset(data, labels, classes, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    if type(classes) == int:
        classes = np.arange(classes)
    sorted_data = []
    sorted_labels = []
    for c in classes:
        idx = (labels == c)
        data_c = data[idx]
        labels_c = labels[idx]
        sorted_data.append(data_c)
        sorted_labels.append(labels_c)
    if stack:
        if isinstance(data, np.ndarray):
            sorted_data = np.vstack(sorted_data)
            sorted_labels = np.hstack(sorted_labels)
        else:
            sorted_data = torch.stack(sorted_data)
            sorted_labels = torch.cat(sorted_labels)
    return sorted_data, sorted_labels

def save_params(model_dir, params, name='params', name_prefix=None):
    """Save params to a .json file. Params is a dictionary of parameters."""
    if name_prefix:
        model_dir = os.path.join(model_dir, name_prefix)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    path = os.path.join(model_dir, "params.json")
    with open(path, 'r') as f:
        _dict = json.load(f)
    return _dict

def update_params(model_dir, dict_):
    params = load_params(model_dir)
    for key in dict_.keys():
        params[key] = dict_[key]
    save_params(model_dir, params)
    return params

def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, f'{filename}.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def append_csv(model_dir, filename, entries):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, f'{filename}.csv')
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def save_loss(model_dir, name, loss_dict):
    save_dir = os.path.join(model_dir, "loss")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "{}.csv".format(name))
    pd.DataFrame(loss_dict).to_csv(file_path)

def save_features(model_dir, name, features, labels, layer=None):
    save_dir = os.path.join(model_dir, "features")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{name}_features.npy"), features)
    np.save(os.path.join(save_dir, f"{name}_labels.npy"), labels)

def save_ckpt(model_dir, name, net):
    """Save PyTorch checkpoint to model_dir/checkpoints/ directory in model directory. """
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints',
        '{}.pt'.format(name)))

def load_ckpt(model_dir, name, net, eval_=True):
    """Load checkpoint from model directory. Checkpoints should be stored in 
    `model_dir/checkpoints/'.
    """
    ckpt_path = os.path.join(model_dir, 'checkpoints', f'{name}.pt')
    print('Loading checkpoint: {}'.format(ckpt_path))
    state_dict = torch.load(ckpt_path)
    net.load_state_dict(state_dict)
    del state_dict
    if eval_:
        net.eval()
    return net

