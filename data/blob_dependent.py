
import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
from .datagen import DataGenerator, DatasetOperator
def sample_D_blobs_dependent(N, sigma_mx_2, r=3, d=2, rho=0.03, nu=0.001,rs=None):
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
    X[:,0] = (1-nu)*X[:, 0] + nu * Y[:, 0].copy()
    Y[:, 0] = (1-nu)*Y[:, 0] + nu * X[:, 0].copy()


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


class BlobData(DatasetOperator): # TODO: the code works only for two dimensions (d=2)
    def __init__(self, type, samples, r=3, d=2, rho=0.03,nu=0, with_labels = False, seed = 0, tau1 = None, tau2 = None):
        super().__init__(tau1, tau2)

        sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
        sigma_mx_2 = np.zeros([r**d, 2, 2])

        for i in range(r**d):
            sigma_mx_2[i] = sigma_mx_2_standard
            if i < 4:
                sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
                sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
            if i == 4:
                sigma_mx_2[i][0, 1] = 0.00
                sigma_mx_2[i][1, 0] = 0.00
            if i > 4:
                sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
                sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)
        if type == "type2":
            X, Y = sample_D_blobs_dependent(samples, sigma_mx_2, r, d, rho,nu=nu, rs=seed)
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
        elif type == "type11":
            Z, _ = sample_D_blobs_dependent(samples, sigma_mx_2, nu=nu,rs=seed)
            X = torch.from_numpy(Z)
            Z, _ = sample_D_blobs_dependent(samples, sigma_mx_2,nu=nu, rs=10000*(seed + 1))
            Y = torch.from_numpy(Z)
        elif type == "type12":
            _, Z = sample_D_blobs_dependent(samples, sigma_mx_2,nu=nu, rs=seed)
            X = torch.from_numpy(Z)
            _, Z = sample_D_blobs_dependent(samples, sigma_mx_2,nu=nu, rs=10000*(seed + 1))
            Y = torch.from_numpy(Z)
        self.x = X.float()
        self.y = Y.float()
        self.z = torch.stack([self.x, self.y], dim=2)
        if with_labels:
            self.x = torch.concat((X.float(), torch.ones((X.shape[0], 1))), dim=1)
            self.y = torch.concat((Y.float(), torch.zeros((Y.shape[0], 1))), dim=1)
            self.z = torch.concat((self.x, self.y))
            idx = torch.randperm(self.z.shape[0])
            self.z  = self.z[idx]
            self.z = torch.stack([self.z[:samples,:], self.z[samples:,:]], dim=2)


class BlobDataGen(DataGenerator):
    def __init__(self, type, samples, data_seed, r, d, rho, with_labels,nu):
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed, self.r, self.d, self.rho, self.with_labels, self.nu = type, samples, data_seed, r, d, rho, with_labels,nu
    def generate(self, seed, tau1, tau2) ->Dataset:
        return BlobData(self.type, self.samples,self.r, self.d,
                        self.rho, self.nu, self.with_labels, (self.data_seed+1)*100+seed, tau1, tau2)