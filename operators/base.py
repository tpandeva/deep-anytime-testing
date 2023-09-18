from abc import ABC, abstractmethod
import torch

class Operator(ABC):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def to(self, device: torch.device) -> None:
        pass

