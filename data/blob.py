
import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator, DatasetOperator
def sample_D_blobs(N, sigma_mx_2, r=3, d=2, rho=0.03, rs=None):
    """
    Generate Blob-D for testing type-II error (or test power).

    Parameters:
    - N (int): Number of samples.
    - sigma_mx_2 (array-like): List of covariance matrices for Y.
    - r (int, optional): Maximum value for the coordinates. Default is 3.
    - d (int, optional): Dimension of the space. Default is 2.
    - rho (float, optional): Diagonal covariance for X. Default is 0.03.
    - rs (int, optional): Random seed. Default is None.

    Returns:
    - X (np.ndarray): Generated samples with shape (N, d).
    - Y (np.ndarray): Generated samples with shape (N, d), with added correlation.
    """
    rs = np.random.default_rng(rs)
    mu = np.zeros(d)
    sigma = np.eye(d) * rho
    X = rs.multivariate_normal(mu, sigma, size=N)
    Y = rs.multivariate_normal(mu, np.eye(d), size=N)

    # Assign to blobs
    X += rs.integers(0, r, size=(N, d))

    locs = np.array(list(itertools.product(range(r), repeat=d)))
    k = r ** d

    # Divide indices into k groups
    ind = np.random.permutation(N)
    groups = np.array_split(ind, k)

    # Add correlation to Y
    for i, group in enumerate(groups):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        Y[group] = Y[group].dot(L.T) + locs[i]

    return X, Y

# Usage example:
# N = 100
# sigma_mx_2 = [np.eye(2) * i for i in range(9)]
# X, Y = sample_blobs_Q(N, sigma_mx_2)


class BlobData(DatasetOperator):
    """
    Dataset operator for generating multi-modal blob data.

    This class is designed for generating blob data.
    Note: Currently only supports 2-dimensional data (d=2).
    """

    def __init__(self, type, samples, r=3, d=2, rho=0.03, with_labels=False, seed=0, tau1=None, tau2=None):
        """
        Initialize the BlobData object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples in the dataset.
        - r (int): A parameter for generating blob data.
        - d (int): Dimensions of the data. Currently only supports 2.
        - rho (float): Density parameter for generating blob data.
        - with_labels (bool): Whether to add labels to the data.
        - seed (int): Random seed
        - tau1 (callable): Operator 1.
        - tau2 (callable): Operator 2.
        """
        super().__init__(tau1, tau2)

        sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
        sigma_mx_2 = np.zeros([r ** d, 2, 2])

        # Constructing the covariance matrix
        for i in range(r ** d):
            sigma_mx_2[i] = sigma_mx_2_standard
            if i < 4:
                sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
                sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
            if i == 4:
                sigma_mx_2[i][0, 1] = 0.00
                sigma_mx_2[i][1, 0] = 0.00
            if i > 4:
                sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
                sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)

        # Sample blob data based on the type specified
        if type == "type2":
            X, Y = sample_D_blobs(samples, sigma_mx_2, r, d, rho, rs=seed)
        elif type == "type11":
            Z, _ = sample_D_blobs(samples, sigma_mx_2, rs=seed)
            X = Z.copy()
            Z, _ = sample_D_blobs(samples, sigma_mx_2, rs=10000 * (seed + 1))
            Y = Z.copy()
        elif type == "type12":
            _, Z = sample_D_blobs(samples, sigma_mx_2, rs=seed)
            X = Z.copy()
            _, Z = sample_D_blobs(samples, sigma_mx_2, rs=10000 * (seed + 1))
            Y = Z.copy()

        # Convert numpy arrays to PyTorch tensors and store
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(Y).float()
        self.z = torch.stack([self.x, self.y], dim=2)

        # Optionally add labels to the data
        if with_labels:
            self.x = torch.concat((self.x, torch.ones((self.x.shape[0], 1))), dim=1)
            self.y = torch.concat((self.y, torch.zeros((self.y.shape[0], 1))), dim=1)
            self.z = torch.concat((self.x, self.y))
            idx = torch.randperm(self.z.shape[0])
            self.z = self.z[idx]
            self.z = torch.stack([self.z[:samples, :], self.z[samples:, :]], dim=2)


class BlobDataGen(DataGenerator):
    """
    Data generator for the BlobData dataset.

    This class is responsible for preparing the blob-like dataset and generating subsets based on a given number of samples.
    """

    def __init__(self, type, samples, data_seed, r, d, rho, with_labels):
        """
        Initialize the BlobDataGen object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate.
        - data_seed (int): Seed for random number generation.
        - r (int): A parameter for generating blob data.
        - d (int): Dimensions of the data. Currently only supports 2.
        - rho (float): Density parameter for generating blob data.
        - with_labels (bool): Whether to add labels to the data.
        """
        super().__init__(type, samples, data_seed)

        # Store parameters
        self.type, self.samples, self.data_seed, self.r, self.d, self.rho, self.with_labels = type, samples, data_seed, r, d, rho, with_labels

    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate a subset of the BlobData dataset.

        Args:
        - seed (int): Seed to determine which subset of the data to use.
        - tau1 (callable): Operator 1.
        - tau2 (callable): Operator 2.

        Returns:
        - Dataset: A subset of the BlobData dataset.
        """
        modified_seed = (self.data_seed + 1) * 100 + seed
        return BlobData(self.type, self.samples, self.r, self.d, self.rho, self.with_labels, modified_seed, tau1, tau2)
