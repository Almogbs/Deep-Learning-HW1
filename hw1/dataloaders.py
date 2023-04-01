import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler, DataLoader, SubsetRandomSampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        i = 0
        while (i < len(self.data_source) - 1 - i):
            yield i
            yield len(self.data_source) - 1 - i
            i += 1
            
        if i == len(self.data_source) - 1 - i:
            yield i
                
    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    train_size = int((1 - validation_ratio) * len(dataset))

    dl_train = DataLoader(dataset, batch_size=batch_size,
         num_workers=num_workers,sampler=SubsetRandomSampler(indices=range(0, train_size)))

    dl_valid = DataLoader(dataset, batch_size=batch_size,
         num_workers=num_workers,sampler=SubsetRandomSampler(indices=range(train_size, len(dataset) )))

    return dl_train, dl_valid
