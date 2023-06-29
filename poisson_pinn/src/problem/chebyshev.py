import torch


def chebyshev_roots(n):
    """ Return the roots of *n* Chebyshev polynomials (between [-1, 1]) """
    pi = torch.acos(torch.zeros(1)).item() * 2
    k = torch.arange(n)
    nodes = torch.sort(torch.cos(pi * (k + 0.5) / n))[0]
    return nodes
