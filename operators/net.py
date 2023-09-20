import torch
import torch.nn as nn
from .base import Operator


class NetOperator(Operator):

    def __init__(self, net: nn.Module, file_to_model: str):
        super().__init__()
        self.Tau = net
        checkpoint = torch.load(file_to_model, map_location=torch.device('cpu'))
        self.Tau.load_state_dict(checkpoint['model_state_dict'])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        return self.Tau(x).detach()

    def to(self, device: torch.device) -> None:
        self.Tau = self.Tau.to(device)
