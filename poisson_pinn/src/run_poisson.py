import argparse
import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus


from problem.pinn import PINN
from problem.plotter import Plotter
from problem.label_tensor import LabelTensor
from model.feed_forward import FeedForward
from adaptive_functions.adaptive_sin import AdaptiveSin
from adaptive_functions.adaptive_tanh import AdaptiveTanh
from poisson import Poisson


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x'])*torch.pi) *
             torch.sin(x.extract(['y'])*torch.pi))
        return LabelTensor(t, ['sin(x)sin(y)'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINN")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature()] if args.features else []

    poisson_problem = Poisson()
    model = FeedForward(
        layers=[64, 64, 64, 64],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pinn = PINN(
        poisson_problem,
        model,
        lr=0.001,
        error_norm='mse',
        regularizer=1e-8,
        device=device,)

    if args.s:

        pinn.span_pts(20, 'grid', locations=[
                      'gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.span_pts(20, 'grid', locations=['D'])
        pinn.train(5000, 100)
        pinn.save_state('pinn.poisson')

    else:
        pinn.load_state('pinn.poisson')
        plotter = Plotter()
        plotter.plot(pinn)
