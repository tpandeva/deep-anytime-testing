from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DataGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def generate(self)->Dataset:
        pass