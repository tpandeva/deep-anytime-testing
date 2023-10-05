import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
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
    #scaler_Y = StandardScaler().fit(Y)
    #Y = scaler_Y.transform(Y)
    X = np.column_stack((X, Z))
    return X, Y, X_mu

def sample_X_given_Z(Z, X_mu):
    n, d = Z.shape
    X = np.random.normal(X_mu, 1, (n, 1))
    X = np.column_stack((X, Z))
    return X


def create_conditional_gauss(X, j, mu, sigma):
    """"
    This function learns the conditional distribution of X_j|X_-j

    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param mu, sigma: The mean and covariance of X.
    :return: The mean and covariance of the conditional distribution.
    To learn more about the implementation see: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    """
    a = np.delete(X, j, 1)
    mu_1 = np.array([mu[j]])
    mu_2 = np.delete(mu, j, 0)
    sigma_11 = sigma[j, j]
    sigma_12 = np.delete(sigma, j, 1)[j, :]
    sigma_21 = np.delete(sigma, j, 0)[:, j]
    sigma_22 = np.delete(np.delete(sigma, j, 0), j, 1)
    mu_bar_vec = []
    sigma12_22 = sigma_12 @ np.linalg.inv(sigma_22)
    sigma_bar = sigma_11 - sigma12_22 @ sigma_21
    for a_i in a:
        mu_bar = mu_1 + sigma12_22 @ (a_i - mu_2)
        mu_bar_vec.append(mu_bar)

    return mu_bar_vec, np.sqrt(sigma_bar)


def sample_from_gaussian(X, X_mu, X_sigma, j=0):
    """
    This function samples the dummy features for gaussian distribution.
    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param X_mu, X_sigma: The mean and covariance of X.
    :return: A copy of the batch X, with the dummy features in the j-th column.
    """
    mu_tilde, sigma_tilde = create_conditional_gauss(X, j, X_mu, X_sigma)
    n = X.shape[0]
    X_tilde = X.copy()
    Xj_tilde = np.random.normal(mu_tilde, sigma_tilde, (n, 1))
    X_tilde[:, j] = Xj_tilde.ravel()
    return X_tilde

class GaussianCIT(DatasetOperator):
    def __init__(self, type, samples, seed, tau1, tau2, u, v):
        super().__init__(tau1, tau2)
        X, Y, mu = get_cit_data(u=u, v=v, n=samples, test=type, seed=seed)
        X_tilde = sample_X_given_Z(X[:, 1:].copy(), mu)
        X = torch.from_numpy(X).to(torch.float32)
        Y = torch.from_numpy(Y).to(torch.float32)
        X_tilde = torch.from_numpy(X_tilde).to(torch.float32)
        Z = torch.cat((X, Y), dim=1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1)
        self.z = torch.stack([Z, Z_tilde], dim=2)

class GaussianCITGen(DataGenerator):
    def __init__(self, type, samples, data_seed):
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed= type, samples, data_seed
        self.d=20
        self.v = np.random.normal(0, 1, (self.d-1, 1))
        self.u = np.random.normal(0, 1, (self.d-1, 1))
    def generate(self, seed, tau1, tau2) ->Dataset:
        return GaussianCIT(self.type, self.samples, (self.data_seed+1)*100+seed, tau1, tau2, self.u, self.v)