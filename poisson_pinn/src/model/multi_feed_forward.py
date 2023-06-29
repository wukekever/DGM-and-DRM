"""Module for Multi FeedForward model"""
import torch

from model.feed_forward import FeedForward


class MultiFeedForward(torch.nn.Module):
    """
    This model allows to create a network with multiple FeedForward combined
    together. The user has to define the `forward` method choosing how to
    combine the different FeedForward networks.

    :param dict dff_dict: dictionary of FeedForward networks.
    """

    def __init__(self, ffn_dict):
        super().__init__()

        if not isinstance(ffn_dict, dict):
            raise TypeError

        for name, constructor_args in ffn_dict.items():
            setattr(self, name, FeedForward(**constructor_args))
