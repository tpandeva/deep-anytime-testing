from abc import ABC, abstractmethod
import torch.nn as nn
import torch
class Operator(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def compute(self, x: torch.Tensor) -> torch.Tensor:
        pass

