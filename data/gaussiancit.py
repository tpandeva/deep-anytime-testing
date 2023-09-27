import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .datagen import DatasetOperator, DataGenerator


def get_cit_data(d=20, a=3, n=5000, test='type1', seed=0):
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
    v = np.random.normal(0, 1, (d-1, 1))
    X_mu = Z @ v
    X = np.random.normal(X_mu, 1, (n, 1))
    u = np.random.normal(0, 1, (d-1, 1))
    beta = np.ones((d, 1))
    if test == 'type2':
        Y_mu = (Z @ u) ** 2 + a * X
    elif test == 'type1':
        Y_mu = (Z @ u) ** 2
        beta[0] = 0
    Y = np.random.normal(Y_mu, 1, (n, 1))
    scaler_Y = StandardScaler().fit(Y)
    Y = scaler_Y.transform(Y)
    X = np.column_stack((X, Z))
    return X, Y, X_mu

def sample_X_given_Z(Z, X_mu):
    n, d = Z.shape
    X = np.random.normal(X_mu, 1, (n, 1))
    X = np.column_stack((X, Z))
    return X




class GaussianCIT(DatasetOperator):
    def __init__(self, type, samples, seed, tau1, tau2):
        super().__init__(tau1, tau2)
        X, Y, mu = get_cit_data(n=samples, test=type, seed=seed)
        X_tilde = sample_X_given_Z(X[:, 1:].copy(), mu)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        X_tilde = torch.from_numpy(X_tilde)
        Z = torch.cat((X, Y), dim=1)
        Z_tilde = torch.cat((X_tilde, Y), dim=1)
        self.z = torch.stack([Z, Z_tilde], dim=2)

class GaussianCITGen(DataGenerator):
    def __init__(self, type, samples, data_seed):
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed= type, samples, data_seed
    def generate(self, seed, tau1, tau2) ->Dataset:
        return GaussianCIT(self.type, self.samples, (self.data_seed+1)*100+seed, tau1, tau2)