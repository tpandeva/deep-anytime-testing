from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DataGenerator(ABC):
    def __init__(self,data_config):
        super(DataGenerator, self).__init__()
        self.data_config = data_config
    @abstractmethod
    def generate(self)->Dataset:
        pass