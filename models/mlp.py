
import torch
import torch.nn as nn
from omegaconf import ListConfig

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with ReLU activation function and optional batch normalization and dropout layers.
    """

    def __init__(self, input_size, hidden_layer_size, output_size, batch_norm=True, drop_out = True, p=0.3):
        super(MLP, self).__init__()

        layers = []
        in_features = input_size

        # Add hidden layers with optional batch normalization
        if isinstance(hidden_layer_size, (list, ListConfig)):
            for out_features in hidden_layer_size:
                layers.append(nn.Linear(in_features, out_features))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if drop_out:
                    layers.append(nn.Dropout(p))
                in_features = out_features

        # Add the output layer
        layers.append(nn.Linear(in_features, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Processed tensor.
        """
        return self.model(x)

class MMDEMLP(MLP):
    def __init__(self,input_size, hidden_layer_size, output_size, batch_norm, drop_out, p):
        super(MMDEMLP, self).__init__(input_size, hidden_layer_size, output_size, batch_norm, drop_out, p)
        self.f = torch.nn.Tanh()
    def forward(self, x,y):

        x_l = self.model(x)
        y_l = self.model(y)
        g = self.f(x_l-y_l)
        output = 2*torch.log(1+g)
        return output