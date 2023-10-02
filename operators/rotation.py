import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import Operator
import numpy as np
import random
from typing import List


class RotateImgOperator(Operator):
    def __init__(self, theta=None, num_rotations=None):
        super().__init__()

        if theta is None and num_rotations is None:
            raise ValueError("Must specify either theta or num_rotations")

        if theta is None:
            theta = 2 * torch.pi / num_rotations
        elif num_rotations is not None:
            theta = min(2 * torch.pi / num_rotations, theta)
        elif num_rotations is None:
            num_rotations = 1
        self.dtype = torch.FloatTensor
        self.rot_mat = nn.Parameter(torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0]
        ]), requires_grad=False).type(self.dtype)
        self.num_rotations = num_rotations

    def rot_img(self, x):
        # x: ( channels, height, width)
        grid = F.affine_grid(self.rot_mat[None, ...].type(self.dtype).repeat(x.shape[0], 1, 1), x.size()).type(
            self.dtype)
        x = F.grid_sample(x.type(self.dtype), grid)
        return x

    def __call__(self, x):
        x_transformed = None
        x = x.unsqueeze(0)
        for _ in range(self.num_rotations):
            x = self.rot_img(x.clone()).type(self.dtype)
            if x_transformed is None:
                x_transformed = x
            else:
                x_transformed = torch.cat((x_transformed[..., None], x[..., None]), dim=-1)
        return x_transformed

    def to(self, device):
        self.dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
        self.rot_mat = self.rot_mat.type(self.dtype)


class RandomRotateImgOperator(Operator):
    def __init__(self, thetas: List):
        super().__init__()
        self.dtype = torch.FloatTensor
        self.theta = thetas

    def rot_img(self, x, theta):
        rot_mat = nn.Parameter(torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0]
        ]), requires_grad=False).type(self.dtype)
        # x: ( channels, height, width)
        grid = F.affine_grid(rot_mat[None, ...].type(self.dtype).repeat(x.shape[0], 1, 1), x.size()).type(self.dtype)
        x = F.grid_sample(x.type(self.dtype), grid)
        return x

    def __call__(self, x):
        x = x.unsqueeze(0)
        theta = random.choice(self.theta)
        x_transformed = self.rot_img(x.clone(), theta).type(self.dtype)
        return x_transformed

    def to(self, device):
        self.dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
        self.rot_mat = self.rot_mat.type(self.dtype)