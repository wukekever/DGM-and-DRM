"""Module for FeedForward model"""
import torch
import torch.nn as nn
import sys 
sys.path.append("..") 

from problem.label_tensor import LabelTensor


class FeedForward(torch.nn.Module):
    """
    The implementation of feedforward network, also refered as multilayer
    perceptron.

    :param list(str) input_variables: the list containing the labels
        corresponding to the input components of the model.
    :param list(str) output_variables: the list containing the labels
        corresponding to the components of the output computed by the model.
    :param int inner_size: number of neurons in the hidden layer(s). Default is
        20.
    :param int n_layers: number of hidden layers. Default is 2.
    :param func: the activation function to use. If a single
        :class:`torch.nn.Module` is passed, this is used as activation function
        after any layers, except the last one. If a list of Modules is passed,
        they are used as activation functions at any layers, in order.
    :param iterable(int) layers: a list containing the number of neurons for
        any hidden layers. If specified, the parameters `n_layers` e
        `inner_size` are not considered.
    :param iterable(torch.nn.Module) extra_features: the additional input
        features to use ad augmented input.
    :param bool bias: If `True` the MLP will consider some bias.
    """

    def __init__(self, input_variables, output_variables, inner_size=20,
                 n_layers=2, func=nn.Tanh, layers=None, extra_features=None,
                 bias=True):
        """
        """
        super().__init__()

        if extra_features is None:
            extra_features = []
        self.extra_features = nn.Sequential(*extra_features)

        if isinstance(input_variables, int):
            self.input_variables = None
            self.input_dimension = input_variables
        elif isinstance(input_variables, (tuple, list)):
            self.input_variables = input_variables
            self.input_dimension = len(input_variables)

        if isinstance(output_variables, int):
            self.output_variables = None
            self.output_dimension = output_variables
        elif isinstance(output_variables, (tuple, list)):
            self.output_variables = output_variables
            self.output_dimension = len(output_variables)

        n_features = len(extra_features)

        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension+n_features)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers)-1):
            self.layers.append(
                nn.Linear(tmp_layers[i], tmp_layers[i + 1], bias=bias)
            )

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers)-1)]

        if len(self.layers) != len(self.functions) + 1:
            raise RuntimeError('uncosistent number of layers and functions')

        unique_list = []
        for layer, func in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func is not None:
                unique_list.append(func())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param x: .
        :type x: :class:`LabelTensor`
        :return: the output computed by the model.
        :rtype: LabelTensor
        """

        if self.input_variables:
            x = x.extract(self.input_variables)

        for feature in self.extra_features:
            x = x.append(feature(x))

        output = self.model(x).as_subclass(LabelTensor)

        if self.output_variables:
            output.labels = self.output_variables

        return output
