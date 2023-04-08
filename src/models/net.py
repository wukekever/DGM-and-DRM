import os

import torch
import torch.nn as nn

# 1. ResNet
# Residual block


class ResidualBlock(nn.Module):
    def __init__(self, units, activation, name="residual_block", **kwargs):
        super(ResidualBlock, self).__init__()
        self._units = units
        self._activation = activation

        self._layers = nn.ModuleList(
            [nn.Linear(units[i], units[i]) for i in range(len(units))]
        )

    def forward(self, inputs):
        residual = inputs
        for i, h_i in enumerate(self._layers):
            inputs = self._activation(h_i(inputs))
        residual = residual + inputs
        return residual


# # inputs shape: [None, dims = 2]


class Model_ResNet(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Model_ResNet, self).__init__()

        self._layers = layers
        self._layer_in = nn.Linear(input_size, layers[0])
        self._residual_blocks = nn.ModuleList()
        self._residual_blocks.append(self._layer_in)

        for i in range(1, len(self._layers) - 1, 2):
            self._residual_blocks.append(
                ResidualBlock(
                    units=self._layers[i : i + 2], activation=self.activation,
                )
            )

        self._output_layer = nn.Linear(layers[-1], output_size)
        self._residual_blocks.append(self._output_layer)

    def forward(self, inputs):
        output = inputs

        for i in range(len(self._residual_blocks)):
            output = self._residual_blocks[i](output)

        return output

    def activation(self, o):
        return o * torch.sigmoid(o)
        # return torch.tanh(o)


# 2. Fully-connected net or Feedforward neural network


# # inputs shape: [None, dims = 2]


class Model_FCNet(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(Model_FCNet, self).__init__()

        self._layers = layers
        self._layer_in = nn.Linear(input_size, layers[0])
        self._hidden_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self._output_layer = nn.Linear(layers[-1], output_size)

    def forward(self, inputs):
        output = inputs
        output = self._layer_in(inputs)
        for i, h_i in enumerate(self._hidden_layers):
            output = self.activation(h_i(output))
        output = self._output_layer(output)

        return output

    def activation(self, o):
        return o * torch.sigmoid(o)
        # return torch.tanh(o)


# other setting


def Xavier_initi(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def save_param(net, path):
    torch.save(net.state_dict(), path)


def load_param(net, path):
    if os.path.exists(path):
        net.load_state_dict(torch.load(path))
    else:
        print("File does not exist.")
