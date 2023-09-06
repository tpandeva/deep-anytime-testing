import torch
import torch.nn as nn
from .base import Operator


class SymOperator(Operator):
    """
    A PyTorch module that applies a symmetric operation
    on the input tensor by multiplying it with -1.
    """

    def __init__(self):
        super().__init__()

        # Scalar symmetric operation value
        self.Tau = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the symmetric operation on the tensor.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Tensor after applying symmetric operation.
        """
        return self.Tau * x
