import numpy as np
import torch
from torch.utils.data import Dataset

from .datagen import DatasetOperator, DataGenerator


def get_cit_data(d=20, a=3, n=5000, test='type1', seed=0, u=None, v=None):
    """Generate data for the PCR test.
     Code from https://github.com/shaersh/ecrt/
    :param d: dimension of the data
    :param a: parameter for the type 2 test
    :param n: number of samples
    :param test: type of the test
    :return: X, Y data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z_mu = np.zeros((d-1, 1)).ravel()
    Z_Sigma = np.eye(d-1)
    Z = np.random.multivariate_normal(Z_mu, Z_Sigma, n)
    if v is None: v = np.random.normal(0, 1, (d-1, 1))
    X_mu = Z @ v
    X = np.random.normal(X_mu, 1, (n, 1))
    if u is None: u = np.random.normal(0, 1, (d-1, 1))
    beta = np.ones((d, 1))
    if test == 'type2':
        Y_mu = (Z @ u) ** 2 + a * X
    elif test == 'type1':
        Y_mu = (Z @ u) ** 2
        beta[0] = 0
    Y = np.random.normal(Y_mu, 1, (n, 1))
    X = np.column_stack((X, Z))
    return X, Y, X_mu

def sample_X_given_Z(Z, X_mu):
    n, d = Z.shape
    X = np.random.normal(X_mu, 1, (n, 1))
    X = np.column_stack((X, Z))
    return X



class GaussianCIT(DatasetOperator):
    """
    Gaussian Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Gaussian CIT dataset.
    """

    def __init__(self, type, samples, seed, tau1, tau2, u, v):
        """
        Initialize the GaussianCIT object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples in the dataset.
        - seed (int): Random seed for reproducibility.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.
        - u (numpy.ndarray): A parameter for generating CIT data.
        - v (numpy.ndarray): Another parameter for generating CIT data.
        """
        super().__init__(tau1, tau2)

        # Retrieve data for Gaussian CIT
        X, Y, mu = get_cit_data(u=u, v=v, n=samples, test=type, seed=seed)

        # Create a sample from X given Z
        X_tilde = sample_X_given_Z(X[:, 1:].copy(), mu)

        # Convert numpy arrays to PyTorch tensors
        X = torch.from_numpy(X).to(torch.float32)
        Y = torch.from_numpy(Y).to(torch.float32)
        X_tilde = torch.from_numpy(X_tilde).to(torch.float32)

        # Concatenate tensors
        Z = torch.cat((X, Y), dim=1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1)

        # Stack the two tensors along a new dimension
        self.z = torch.stack([Z, Z_tilde], dim=2)


class GaussianCITGen(DataGenerator):
    """
    Gaussian CIT Data Generator class that extends the DataGenerator.

    This class is responsible for generating datasets using the GaussianCIT method.
    """

    def __init__(self, type, samples, data_seed):
        """
        Initialize the GaussianCITGen object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate.
        - data_seed (int): Seed for random number generation.
        """
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed = type, samples, data_seed
        self.d = 20 # for now is hardcoded
        # Generate random vectors for u and v
        self.v = np.random.normal(0, 1, (self.d - 1, 1))
        self.u = np.random.normal(0, 1, (self.d - 1, 1))

    def generate(self, seed, tau1, tau2) -> Dataset:
        """
        Generate data using the GaussianCIT method.

        Args:
        - seed (int): Seed for random number generation.
        - tau1 (float): Tau parameter 1.
        - tau2 (float): Tau parameter 2.

        Returns:
        - Dataset: A dataset generated using GaussianCIT.
        """
        # Use a modified seed value based on the provided seed and class's data_seed
        modified_seed = (self.data_seed + 1) * 100 + seed
        return GaussianCIT(self.type, self.samples, modified_seed, tau1, tau2, self.u, self.v)
