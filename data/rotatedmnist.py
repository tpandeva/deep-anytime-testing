import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator
from PIL import Image


class MnistRotDataset(Dataset):

    def __init__(self, X, Y, samples):
        self.X = X
        self.Y = Y
        self.num_samples = samples

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x, mode='F')
        y = Image.fromarray(y, mode='F')
        return x, y

    def __len__(self):
        return self.num_samples

class RotatedMnistDataGen(DataGenerator):
    def __init__(self, type, samples,  file1, file2):
        super().__init__(type, samples)
        if type == "type12":
            file1 = file2
        if type == "type11":
            file2 = file1

        data1 = np.loadtxt(file1, delimiter=' ')
        np.random.shuffle(data1)
        data2 = np.loadtxt(file2, delimiter=' ')
        np.random.shuffle(data2)
        self.X = data1[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.Y = data2[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        total_samples = min(self.X.shape[0], self.Y.shape[0])
        num_chunks = int(total_samples / samples)
        self.index_sets_seq = np.array_split(range(total_samples), num_chunks)

    def generate(self,seed) -> Dataset:
        ind = self.index_sets_seq[seed]
        return MnistRotDataset(self.X[ind,:,:], self.Y[ind,:,:], len(ind))