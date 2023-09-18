import torch
import torch.nn as nn
from .base import Operator


class ReshapeOperator(Operator):

    def __init__(self, shape):
        super().__init__()

        self.shape = shape if isinstance(shape, tuple) else tuple(shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return x.reshape(self.shape)

    def to(self, device: torch.device) -> None:
        pass
