import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import Operator
import numpy as np

class RotateImgOperator(Operator):
    def __init__(self, theta=None, num_rotations=None, dtype=torch.float64):
        super().__init__()

        if theta is None and num_rotations is None:
            raise ValueError("Must specify either theta or num_rotations")

        if theta is None:
            theta = 2 * torch.pi / num_rotations
        elif num_rotations is not None:
            theta = min(2 * torch.pi / num_rotations, theta)
        elif num_rotations is None:
            num_rotations = 1

        self.rot_mat = nn.Parameter(torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0]
        ], dtype=dtype), requires_grad=False)
        self.num_rotations = num_rotations
        self.dtype = dtype

    def rot_img(self, x):
        # x: (batch_size, channels, height, width)
        grid = F.affine_grid(self.rot_mat[None, ...].type(self.dtype).repeat(x.shape[0], 1, 1), x.size()).type(self.dtype)
        x = F.grid_sample(x, grid)
        return x

    def __call__(self, x):
        x_transformed = None
        for _ in range(self.num_rotations):
            x = self.rot_img(x.clone())
            if x_transformed is None:
                x_transformed = x[None, :, :, :, :]
            else:
                x_transformed = torch.cat((x_transformed, x[None, :, :, :, :]), dim=0)
        return x_transformed

    def to(self, device):
        self.dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
        self.rot_mat = self.rot_mat.type(self.dtype)