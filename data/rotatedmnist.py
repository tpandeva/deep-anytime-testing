import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator, DatasetOperator
import torchvision
from torchvision import transforms
from operators import RandomRotateImgOperator


class MnistRotDataset(DatasetOperator):

    def __init__(self, z, tau1, tau2):
        super().__init__(tau1, tau2)
        self.z = z


class RotatedMnistDataGen(DataGenerator):
    def __init__(self, type, samples, data_seed, p):
        super().__init__(type, samples, data_seed)
        if type == "type12":
            p = 0.5
        if type == "type11":
            p = 0.5
        if type == "type2":
            p = p

        transforms_MNIST = transforms.Compose(
            [
                transforms.ToTensor(),
                RandomRotateImgOperator([2 * torch.pi / r for r in [-4, 4, 2, 1]]),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ]
        )
        mnist = torchvision.datasets.MNIST("/var/scratch/tpandeva/deep-anytime-testing/data/mnist", train=True,
                                           transform=transforms_MNIST, download=False)
        loader = torch.utils.data.DataLoader(mnist, batch_size=len(mnist), shuffle=True)
        data = next(iter(loader))
        idx_test = data[1] == 6
        mnist6 = data[0][idx_test]
        idx_test = data[1] == 9
        mnist9 = data[0][idx_test]

        num_samples = min(mnist9.shape[0], mnist6.shape[0])
        mnist9 = mnist9[:num_samples]
        mnist6 = mnist6[:num_samples]
        half_size = int(p * num_samples)

        data6_ = mnist6[:half_size, :].clone()
        data9_ = mnist9[:half_size, :].clone()
        mnist6[:half_size, :] = data9_.clone()
        mnist9[:half_size, :] = data6_.clone()

        idx = torch.randperm(mnist6.shape[0])
        mnist6 = mnist6[idx, ...]
        idx = torch.randperm(mnist9.shape[0])
        mnist9 = mnist9[idx, ...]

        self.X = 1.0 * torch.flatten(mnist6, 1)
        self.Y = 1.0 * torch.flatten(mnist9, 1)

        total_samples = min(self.X.shape[0], self.Y.shape[0])
        self.z = torch.stack([self.X[:total_samples, ...], self.Y[:total_samples, ...]], dim=2)
        num_chunks = int(total_samples / samples)

        self.index_sets_seq = np.array_split(range(total_samples), num_chunks)

    def generate(self, seed, tau1, tau2) -> Dataset:
        ind = self.index_sets_seq[seed]
        return MnistRotDataset(self.z[ind, :, :], tau1, tau2)