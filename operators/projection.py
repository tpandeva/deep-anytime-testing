import torch
from .base import Operator

class ProjOperator(Operator):

    def __init__(self, input_dim=None, p=None):
        if input_dim is None and p is None:
            raise ValueError("Must specify either input_dim or p")
        if input_dim is not None and p is not None:
            raise ValueError("Specify either input_dim or p, not both")
        super().__init__()
        self.input_dim = input_dim
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim is not None and x.shape[-1]>1:
            proj_x = x[:, self.input_dim]
        elif self.p is not None:
            proj_x = x[:, :self.p]
        return proj_x

    def to(self, device: torch.device) -> None:
        pass