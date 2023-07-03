import numpy as np
import torch

from problem.problem_cls import SpatialProblem
from problem.condition import Condition
from problem.span import Span
from problem.operators import grad, div, nabla
from problem.label_tensor import LabelTensor


class Posisson1D_pinn(SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['u']
    spatial_domain = Span({'x': [-1, 1]})

    # define the laplace equation
    def pinn_equation(input_, output_):
        x, u = input_.extract(['x']), output_.extract(['u'])
        force_term = - 2.0 * (x >= 0)
        a = fun_a(x)
        # nabla_u = nabla(u, input_)
        # return - a * nabla_u - force_term
        grad_u = grad(u, input_)
        a = LabelTensor(a, ['u'])
        ag = LabelTensor(a * grad_u, ['u'])
        nabla_u = grad(ag, input_)
        return - nabla_u - force_term

    # define nill dirichlet boundary conditions
    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=Span({'x': -1}), function=nil_dirichlet),
        'gamma2': Condition(location=Span({'x': 1}), function=nil_dirichlet),
        'D': Condition(location=Span({'x': [-1, 1]}), function=pinn_equation),
    }

    # real poisson solution
    def poisson_sol(self, pts):
        x = pts.extract(['x'])
        u = - 2 / 3 * (x + 1) * (x < 0) + (x**2 - x / 3 - 2 / 3) * (x >= 0)
        return u

    truth_solution = poisson_sol


class Posisson1D_ritz(SpatialProblem):

    # assign output/ spatial variables
    output_variables = ['u']
    spatial_domain = Span({'x': [-1, 1]})

    # define the laplace equation
    def ritz_equation(input_, output_):
        x, u = input_.extract(['x']), output_.extract(['u'])
        force_term = - 2.0 * (x >= 0)
        a = fun_a(x)
        grad_u = grad(u, input_)

        return (0.5*a*grad_u**2 - force_term * u) * 2
    
    # define nill dirichlet boundary conditions
    def nil_dirichlet(input_, output_):
        value = 0.0
        penalty = 1e2
        return (output_.extract(['u']) - value)**2  * penalty

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=Span({'x': -1}), function=nil_dirichlet),
        'gamma2': Condition(location=Span({'x': 1}), function=nil_dirichlet),
        'D': Condition(location=Span({'x': [-1, 1]}), function=ritz_equation),
    }

    # real poisson solution
    def poisson_sol(self, pts):
        x = pts.extract(['x'])
        u = - 2 / 3 * (x + 1) * (x < 0) + (x**2 - x / 3 - 2 / 3) * (x >= 0)
        return u

    truth_solution = poisson_sol


def fun_a(x):
    return 1 / 2 * (x < 0) + 1 * (x >= 0)

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



