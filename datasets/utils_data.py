import numpy as np
import torch



def filter_class(dataset, classes):
    data, labels = dataset.data, dataset.targets
    if type(labels) == list:
        labels = torch.tensor(labels)
    data_filter = []
    labels_filter = []
    for _class in classes:
        idx = labels == _class
        data_filter.append(data[idx])
        labels_filter.append(labels[idx])
    if type(dataset.data) == np.ndarray:
        dataset.data = np.vstack(data_filter)
        dataset.targets = np.hstack(labels_filter)
    elif type(dataset.data) == torch.Tensor:
        dataset.data = torch.cat(data_filter)
        dataset.targets = torch.cat(labels_filter)
    else:
        raise TypeError('dataset.data type neither np.ndarray nor torch.Tensor')
    return dataset, len(classes)