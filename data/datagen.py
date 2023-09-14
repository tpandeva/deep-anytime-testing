from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class DataGenerator(ABC):
    def __init__(self, type, samples):
        assert type in ["type2", "type11", "type12"]
        assert samples > 0

    @abstractmethod
    def generate(self)->Dataset:
        pass