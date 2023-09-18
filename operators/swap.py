import torch
import torch.nn as nn
from .base import Operator

class SwapOperator(Operator):
    """
    A PyTorch module that applies a specific swap operation
    on the last dimension of a 2D tensor.
    """

    def __init__(self, p, d=None):
        """
        Initializes the swap operator.

        Parameters:
        - p (int): size of the second dimension of the input tensor.
        - d (int): starting column for swap operation. Defaults to 0.
        """
        super().__init__()
        self.p = p
        self.d = 0 if d is None else d

        # Swap matrix
        self.Tau = nn.Parameter(torch.tensor([[0., 1.], [1., 0.]]), requires_grad=False)

        # Construct the mask
        self.mask = nn.Parameter(torch.zeros(p, 2), requires_grad=False)
        self.mask[self.d:, :] = 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the swap operation on the tensor.

        Parameters:
        - x (torch.Tensor): Input tensor of shape [batch_size, p, 2].

        Returns:
        - torch.Tensor: Tensor after applying swap operation.
        """
        # if x.dim() != 3:
        #     raise ValueError(f"Expected input tensor of 3 dimensions, but got {x.dim()}.")
        #
        # if x.size(1) != self.p:
        #     raise ValueError(f"Expected input tensor second dimension to be {self.p}, but got {x.size(1)}.")

        # Apply swap using masking
        return (1 - self.mask) * x + self.mask * (x @ self.Tau)

    def to(self, device: torch.device) -> None:
        self.mask = self.mask.to(device)
        self.Tau = self.Tau.to(device)
