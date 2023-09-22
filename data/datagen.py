from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import numpy as np

class DataGenerator(ABC):
    def __init__(self, type, samples,data_seed):
        assert type in ["type2", "type11", "type12"]
        assert samples > 0
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
    @abstractmethod
    def generate(self)->Dataset:
        pass

class DatasetOperator(Dataset):
    def __init__(self, tau1, tau2):
        self.tau1 = tau1
        self.tau2 = tau2

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        tau1_z, tau2_z = self.z[idx], self.z[idx].clone()
        if self.tau1 is not None:
            tau1_z = self.tau1(tau1_z)
        if self.tau2 is not None:
            tau2_z = self.tau2(tau2_z)
        return tau1_z, tau2_z