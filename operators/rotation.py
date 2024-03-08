import torch
from .base import Operator
import random
from typing import List
import torchvision


class RotateImgOperator(Operator):
    def __init__(self, theta=None, num_rotations=None):
        super().__init__()

        if theta is None and num_rotations is None:
            raise ValueError("Must specify either theta or num_rotations")

        if theta is None:
            theta = int(360 / num_rotations)  # 2 * torch.pi / num_rotations
        elif num_rotations is not None:
            theta = min(360 / num_rotations, theta)  # min(2 * torch.pi / num_rotations, theta)
        elif num_rotations is None:
            num_rotations = 1
        self.theta = theta
        self.num_rotations = num_rotations

    def rot_img(self, x):
        # x: ( channels, height, width)
        x = torchvision.transforms.functional.rotate(x, self.theta)
        return x

    def __call__(self, x):
        x_transformed = None

        for _ in range(self.num_rotations - 1):
            x = self.rot_img(x.clone()).type(self.dtype)
            if x_transformed is None:
                x_transformed = x[..., None]
            else:
                x_transformed = torch.cat((x_transformed, x[..., None]), dim=-1)
        return x_transformed

    def to(self, device):
        self.dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor

class RandomRotateImgOperator(Operator):
    def __init__(self, thetas: List):
        super().__init__()
        self.dtype = torch.FloatTensor
        self.theta = thetas

    def rot_img(self, x, theta):
        x = torchvision.transforms.functional.rotate(x, theta)
        return x

    def __call__(self, x):
        # x = x.unsqueeze(0)
        theta = random.choice(self.theta)
        x_transformed = self.rot_img(x.clone(), theta).type(self.dtype)
        return x_transformed

    def to(self, device):
        self.dtype = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor