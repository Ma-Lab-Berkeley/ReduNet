import numpy as np
import scipy.stats as sps
import torch

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import functional as F
import utils



def evaluate(eval_dir, method, train_features, train_labels, test_features, test_labels, **kwargs):
    if method == 'svm':
        acc_train, acc_test = svm(train_features, train_labels, test_features, test_labels)
    elif method == 'knn':
        acc_train, acc_test = knn(train_features, train_labels, test_features, test_labels, **kwargs)
    elif method == 'nearsub':
        acc_train, acc_test = nearsub(train_features, train_labels, test_features, test_labels, **kwargs)
    elif method == 'nearsub_pca':
        acc_train, acc_test = knn(train_features, train_labels, test_features, test_labels, **kwargs)
    acc_dict = {'train': acc_train, 'test': acc_test}
    utils.save_params(eval_dir, acc_dict, name=f'acc_{method}')

def svm(train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}, {}".format(acc_train, acc_test))
    return acc_train, acc_test

# def knn(train_features, train_labels, test_features, test_labels, k=5):
#     sim_mat = train_features @ train_features.T
#     topk = torch.from_numpy(sim_mat).topk(k=k, dim=0)
#     topk_pred = train_labels[topk.indices]
#     test_pred = torch.tensor(topk_pred).mode(0).values.detach()
#     acc_train = compute_accuracy(test_pred.numpy(), train_labels)

#     sim_mat = train_features @ test_features.T
#     topk = torch.from_numpy(sim_mat).topk(k=k, dim=0)
#     topk_pred = train_labels[topk.indices]
#     test_pred = torch.tensor(topk_pred).mode(0).values.detach()
#     acc_test = compute_accuracy(test_pred.numpy(), test_labels)
#     print("kNN: {}, {}".format(acc_train, acc_test))
#     return acc_train, acc_test

def knn(train_features, train_labels, test_features, test_labels, k=5):
    sim_mat = train_features @ train_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc_train = compute_accuracy(test_pred, train_labels)

    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc_test = compute_accuracy(test_pred, test_labels)
    print("kNN: {}, {}".format(acc_train, acc_test))
    return acc_train, acc_test

# # TODO: 1. implement pytorch version 2. suport batches
# def nearsub(train_features, train_labels, test_features, test_labels, num_classes, n_comp=10, return_pred=False):
#     train_scores, test_scores = [], []
#     classes = np.arange(num_classes)
#     features_sort, _ = utils.sort_dataset(train_features, train_labels, 
#                                           classes=classes, stack=False)           
#     fd = features_sort[0].shape[1]
#     if n_comp >= fd:
#         n_comp = fd - 1
#     for j in classes:
#         svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
#         subspace_j = np.eye(fd) - svd.components_.T @ svd.components_
#         train_j = subspace_j @ train_features.T
#         test_j = subspace_j @ test_features.T
#         train_scores_j = np.linalg.norm(train_j, ord=2, axis=0)
#         test_scores_j = np.linalg.norm(test_j, ord=2, axis=0)
#         train_scores.append(train_scores_j)
#         test_scores.append(test_scores_j)
#     train_pred = np.argmin(train_scores, axis=0)
#     test_pred = np.argmin(test_scores, axis=0)
#     if return_pred:
#         return train_pred.tolist(), test_pred.tolist()
#     train_acc = compute_accuracy(classes[train_pred], train_labels)
#     test_acc = compute_accuracy(classes[test_pred], test_labels)
#     print('SVD: {}, {}'.format(train_acc, test_acc))
#     return train_acc, test_acc

def nearsub(train_features, train_labels, test_features, test_labels, 
            num_classes, n_comp=10, return_pred=False):
    train_scores, test_scores = [], []
    classes = np.arange(num_classes)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, 
                                          classes=classes, stack=False)           
    fd = features_sort[0].shape[1]
    for j in classes:
        _, _, V = torch.svd(features_sort[j])
        components = V[:, :n_comp].T
        subspace_j = torch.eye(fd) - components.T @ components
        train_j = subspace_j @ train_features.T
        test_j = subspace_j @ test_features.T
        train_scores_j = torch.linalg.norm(train_j, ord=2, axis=0)
        test_scores_j = torch.linalg.norm(test_j, ord=2, axis=0)
        train_scores.append(train_scores_j)
        test_scores.append(test_scores_j)
    train_pred = torch.stack(train_scores).argmin(0)
    test_pred = torch.stack(test_scores).argmin(0)
    if return_pred:
        return train_pred.numpy(), test_pred.numpy()
    train_acc = compute_accuracy(classes[train_pred], train_labels.numpy())
    test_acc = compute_accuracy(classes[test_pred], test_labels.numpy())
    print('SVD: {}, {}'.format(train_acc, test_acc))
    return train_acc, test_acc

def nearsub_pca(train_features, train_labels, test_features, test_labels, num_classes, n_comp=10):
    scores_pca = []
    classes = np.arange(num_classes)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, classes=classes, stack=False)           
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in np.arange(len(classes)):
        pca = PCA(n_components=n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)
        scores_pca.append(score_pca_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    acc_pca = compute_accuracy(classes[test_predict_pca], test_labels)
    print('PCA: {}'.format(acc_pca))
    return acc_pca

def argmax(train_features, train_labels, test_features, test_labels):
    train_pred = train_features.argmax(1)
    train_acc = compute_accuracy(train_pred, train_labels)
    test_pred = test_features.argmax(1)
    test_acc = compute_accuracy(test_pred, test_labels)
    return train_acc, test_acc

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    if type(y_pred) == torch.Tensor:
        n_wrong = torch.count_nonzero(y_pred - y_true).item()
    elif type(y_pred) == np.ndarray:
        n_wrong = np.count_nonzero(y_pred - y_true)
    else:
        raise TypeError("Not Tensor nor Array type.")
    n_samples = len(y_pred)
    return 1 - n_wrong / n_samples

def baseline(train_features, train_labels, test_features, test_labels):
    test_models = {'log_l2': SGDClassifier(loss='log', max_iter=10000, random_state=42),
                   'SVM_linear': LinearSVC(max_iter=10000, random_state=42),
                   'SVM_RBF': SVC(kernel='rbf', random_state=42),
                   'DecisionTree': DecisionTreeClassifier(),
                   'RandomForrest': RandomForestClassifier()}
    for model_name in test_models:
        test_model = test_models[model_name]
        test_model.fit(train_features, train_labels)
        score = test_model.score(test_features, test_labels)
        print(f"{model_name}: {score}")

def majority_vote(pred, true):
    pred_majority = sps.mode(pred, axis=0)[0].squeeze()
    return compute_accuracy(pred_majority, true)
