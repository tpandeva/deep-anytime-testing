
import torch
import torch.nn as nn
from omegaconf import ListConfig

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with ReLU activation function and optional batch normalization and dropout layers.
    """

    def __init__(self, input_size, hidden_layer_size, output_size, layer_norm=True, drop_out=True, drop_out_p=0.3, bias=False):
        super(MLP, self).__init__()

        layers = []
        in_features = input_size

        # Add hidden layers with optional batch normalization and drop out
        if isinstance(hidden_layer_size, (list, ListConfig)):
            for out_features in hidden_layer_size:
                layers.append(nn.Linear(in_features, out_features, bias=bias))
                if layer_norm:
                    layers.append(nn.LayerNorm(out_features))
                layers.append(nn.ReLU())
                if drop_out:
                    layers.append(nn.Dropout(drop_out_p))
                in_features = out_features

        # Add the output layer
        layers.append(nn.Linear(in_features, output_size, bias=bias))

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
    """
    MMDEMLP (Modified Mean Discrepancy MLP) is an extension of the base MLP (Multi-Layer Perceptron).

    This class takes in a configuration object `cfg` that specifies the parameters for the neural network.
    The forward method implements a custom operation over the outputs of the base MLP.
    """

    def __init__(self, input_size, hidden_layer_size, output_size, layer_norm, drop_out, drop_out_p, bias):
        """
        Initializes the MMDEMLP object.

        Parameters:
        - input_size (int): Size of input layer.
        - hidden_layer_size (int): Size of hidden layer.
        - output_size (int): Size of output layer.
        - batch_norm (bool): Indicates if batch normalization should be applied.
        - drop_out(bool): Indicates if dropout should be applied.
        - drop_out_p (float): The probability of an element to be zeroed.
        - bias (bool): Indicates if bias term would be added to the linear layer
        - full_dim (bool): Indicates if the full dimension of the input should be used.
        """

        # Initialize base MLP
        super(MMDEMLP, self).__init__(input_size, hidden_layer_size, output_size, layer_norm, drop_out, drop_out_p, bias)

        # Activation function for the custom operation in the forward method
        self.sigma = torch.nn.Tanh()


    def forward(self, x, y) -> torch.Tensor:
        """
        Forward pass for the MMDEMLP model. Computes the output based on inputs x and y.

        Parameters:
        - x (torch.Tensor): First input tensor.
        - y (torch.Tensor): Second input tensor.

        Returns:
        - output (torch.Tensor): The output of the forward pass.
        """

        # Compute the outputs from the base MLP model for x and y
        if len(x.shape)>2: x = torch.flatten(x, start_dim=1)
        if len(y.shape)>2: y = torch.flatten(y, start_dim=1)
        g_x = self.model(x)
        g_y = self.model(y)

        # Compute final output
        output = torch.log(1 + self.sigma(g_x - g_y))

        return output
