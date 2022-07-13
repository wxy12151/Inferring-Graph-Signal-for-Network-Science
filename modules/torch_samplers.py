#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:53:36 2021

@author: gardar
"""

import torch
from torch.utils.data import Sampler

# Implementation from https://github.com/pytorch/pytorch/issues/18317
class RandomBatchSampler(Sampler):
    """
    Yield a mini-batch of indices with random batch order

    Arguments:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, data_source, batch_size, drop_last):

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral "
                             "value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.data_source   = data_source
        self.batch_size    = batch_size
        self.drop_last     = drop_last
        self.fragment_size = len(data_source) % batch_size

    def __iter__(self):
        batch_indices = range(0, len(self.data_source) - self.fragment_size, self.batch_size)

        for batch_indices_idx in torch.randperm(len(self.data_source) // self.batch_size):
            yield list(range(batch_indices[batch_indices_idx], batch_indices[batch_indices_idx]+self.batch_size))

        if self.fragment_size > 0 and not self.drop_last:
            yield list(range(len(self.data_source) - self.fragment_size, len(self.data_source)))

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size