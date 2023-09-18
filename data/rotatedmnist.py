import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator
from PIL import Image
from torchvision import transforms

class MnistRotDataset(Dataset):

    def __init__(self, z, samples, tau1, tau2):
        self.z = z
        self.num_samples = samples
        self.tau1 = tau1
        self.tau2 = tau2

    def __getitem__(self, idx):
        tau1_z, tau2_z = self.z[idx], self.z[idx].clone()
        if self.tau1 is not None:
            tau1_z = self.tau1(tau1_z)
        if self.tau2 is not None:
            tau2_z = self.tau2(tau2_z)
        return tau1_z, tau2_z

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
        self.X = torch.tensor(data1[:, :-1].astype(np.float32))
        self.Y = torch.tensor(data2[:, :-1].astype(np.float32))
        self.z = torch.stack([self.X, self.Y], dim=2)
        total_samples = min(self.X.shape[0], self.Y.shape[0])
        num_chunks = int(total_samples / samples)
        self.index_sets_seq = np.array_split(range(total_samples), num_chunks)

    def generate(self,seed, tau1, tau2) -> Dataset:
        ind = self.index_sets_seq[seed]
        return MnistRotDataset(self.z[ind,:,:], len(ind), tau1, tau2)