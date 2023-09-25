import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator, DatasetOperator


class MnistRotDataset(DatasetOperator):

    def __init__(self, z, tau1, tau2):
        super().__init__(tau1, tau2)
        self.z = z

class RotatedMnistDataGen(DataGenerator):
    def __init__(self, type, samples, data_seed, file1, file2):
        super().__init__(type, samples, data_seed)
        if type == "type12":
            file1 = file2
        if type == "type11":
            file2 = file1

        data1 = np.loadtxt(file1, delimiter=' ')
        np.random.shuffle(data1)
        data2 = np.loadtxt(file2, delimiter=' ')
        np.random.shuffle(data2)
        self.X = torch.tensor(data1[:, :-1].astype(np.float32))
        self.Y = torch.tensor(data2[:, :-1].astype(np.float32))

        total_samples = min(self.X.shape[0], self.Y.shape[0])
        self.z = torch.stack([self.X[:total_samples, ...], self.Y[:total_samples, ...]], dim=2)
        num_chunks = int(total_samples / samples)

        self.index_sets_seq = np.array_split(range(total_samples), num_chunks)

    def generate(self,seed, tau1, tau2) -> Dataset:
        ind = self.index_sets_seq[seed]
        return MnistRotDataset(self.z[ind,:,:], tau1, tau2)