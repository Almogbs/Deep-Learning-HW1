import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders

class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """
        self.x_train, self.y_train = dataloader_utils.flatten(dl_train)
        self.n_classes = len(self.y_train.unique())

        return self
    
    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """
        dist_matrix = l2_dist(x_test, self.x_train)
        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            _, inx = dist_matrix[i].topk(self.k, largest=False)
            y_pred[i] = int(torch.mode(self.y_train[inx])[0])
    
        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """
    x1_sqrd = torch.sum(x1 ** 2, dim=1).reshape([-1, 1])
    x2_sqrd = torch.sum(x2 ** 2, dim=1).reshape([1, -1])

    return torch.sqrt(x1_sqrd + x2_sqrd - 2 * (x1.matmul(x2.T)))

def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    return int(torch.sum(y == y_pred)) / len(y)

def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """
    fold_size = int(len(ds_train) / num_folds)
    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        curr_accuracies = []
        for fold in range(num_folds):
            df_val = Subset(ds_train, list(range(fold * fold_size, (fold + 1) * fold_size)))
            df_train = Subset(ds_train, list(range(0, fold * fold_size)) + list(range((fold + 1) * fold_size, len(ds_train))))
            
            x_val, y_val = dataloader_utils.flatten(DataLoader(df_val, batch_size=4, shuffle=False, num_workers=2))
        
            model.train(DataLoader(df_train, batch_size=4, shuffle=False, num_workers=2))
            y_pred = model.predict(x_val)

            acc = accuracy(y_val, y_pred)

            curr_accuracies.append(acc)
        
        accuracies.append(curr_accuracies)
        
    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
