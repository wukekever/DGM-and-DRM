import numpy as np
import torch

from problem.problem_cls import SpatialProblem
from problem.condition import Condition
from problem.span import Span
from problem.operators import grad, div, nabla


# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Poisson problem. The Poisson class is defined        #
#  inheriting from SpatialProblem. We  denote:          #
#           u --> field variable                        #
#           x,y --> spatial variables                   #
#                                                       #
# ===================================================== #


class Poisson(SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})

    # define the laplace equation
    def laplace_equation(input_, output_):
        force_term = - (torch.sin(input_.extract(['x'])*torch.pi) *
                        torch.sin(input_.extract(['y'])*torch.pi))
        nabla_u = nabla(output_.extract(['u']), input_)
        return - nabla_u - force_term

    # define nill dirichlet boundary conditions
    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=Span({'x': [0, 1], 'y':  1}), function=nil_dirichlet),
        'gamma2': Condition(location=Span({'x': [0, 1], 'y': 0}), function=nil_dirichlet),
        'gamma3': Condition(location=Span({'x':  1, 'y': [0, 1]}), function=nil_dirichlet),
        'gamma4': Condition(location=Span({'x': 0, 'y': [0, 1]}), function=nil_dirichlet),
        'D': Condition(location=Span({'x': [0, 1], 'y': [0, 1]}), function=laplace_equation),
    }

    # real poisson solution
    def poisson_sol(self, pts):
        return -(
            torch.sin(pts.extract(['x'])*torch.pi) *
            torch.sin(pts.extract(['y'])*torch.pi)
        )/(2*torch.pi**2)

    truth_solution = poisson_sol
