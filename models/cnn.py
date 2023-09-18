import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_size, 32, 3, 1),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2),
                                  nn.Dropout(0.25))
        self.fc = nn.Sequential(nn.Linear(9216, 128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, output_size))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class RotCNN(CNN):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.sigma = torch.nn.Tanh()

    def forward(self, x,y):

        num_rotations = x.shape[1] if len(x.shape) == 5 else 1
        out_x, out_y = 0, 0
        for i in range(num_rotations):
            x_conv = self.conv(x[:,i,...])
            out_x += torch.flatten(x_conv, 1)
            y_conv = self.conv(y[:,i,...])
            out_y += torch.flatten(y_conv, 1)

        g_x = self.fc(out_x)
        g_y = self.fc(out_y)
        output = 2 * torch.log(1 + self.sigma(g_x - g_y))
        return output