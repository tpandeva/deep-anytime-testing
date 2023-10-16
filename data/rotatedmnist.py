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
    """
    Data generator for the Rotated MNIST dataset. This class preprocesses the MNIST dataset
    and rotates specific images to create a modified dataset.
    """

    def __init__(self, type, samples, data_seed, p, file_path):
        """
        Initialize the RotatedMnistDataGen object.

        Args:
        - type (str): Specifies the type of rotation applied.
        - samples (int): Number of samples to generate.
        - data_seed (int): Seed for random number generation.
        - p (float): Mixing parameter.
        - file_path (str): Path to the MNIST dataset or where it should be downloaded.
        """
        super().__init__(type, samples, data_seed)


        # Define transformations for the MNIST dataset
        transforms_MNIST = transforms.Compose(
            [
                transforms.ToTensor(),
                RandomRotateImgOperator([2 * torch.pi / r for r in [-4, 4, 2, 1]]),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ]
        )

        # Load the MNIST dataset
        mnist = torchvision.datasets.MNIST(file_path, train=True, transform=transforms_MNIST, download=True)
        loader = torch.utils.data.DataLoader(mnist, batch_size=len(mnist), shuffle=True)
        data = next(iter(loader))

        # Split the data based on labels 6 and 9
        idx_test6 = data[1] == 6
        mnist6 = data[0][idx_test6]

        idx_test9 = data[1] == 9
        mnist9 = data[0][idx_test9]

        # Ensure equal number of samples for both classes
        num_samples = min(mnist9.shape[0], mnist6.shape[0])
        mnist9 = mnist9[:num_samples]
        mnist6 = mnist6[:num_samples]
        p_size = int(p * num_samples)

        # Swap half of the images between the two classes
        data6_ = mnist6[:p_size, :].clone()
        data9_ = mnist9[:p_size, :].clone()
        mnist6[:p_size, :] = data9_.clone()
        mnist9[:p_size, :] = data6_.clone()

        # Shuffle the images
        idx6 = torch.randperm(mnist6.shape[0])
        mnist6 = mnist6[idx6, ...]

        idx9 = torch.randperm(mnist9.shape[0])
        mnist9 = mnist9[idx9, ...]

        # Flatten and store the image tensors
        self.X = 1.0 * torch.flatten(mnist6, 1)
        self.Y = 1.0 * torch.flatten(mnist9, 1)

        # Create subsets based on the 'samples' parameter
        total_samples = min(self.X.shape[0], self.Y.shape[0])
        self.z = torch.stack([self.X[:total_samples, ...], self.Y[:total_samples, ...]], dim=2)
        num_chunks = int(total_samples / samples)
        self.index_sets_seq = np.array_split(range(total_samples), num_chunks)

    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate a subset of the data based on the provided seed.

        Args:
        - seed (int): Seed to determine which subset of the data to use.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.

        Returns:
        - Dataset: A subset of the Rotated MNIST dataset.
        """
        ind = self.index_sets_seq[seed]
        return MnistRotDataset(self.z[ind, ...], tau1, tau2)
