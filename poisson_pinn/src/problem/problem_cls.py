"""Module for the Problem class"""

from abc import abstractmethod
from problem.abstract_problem import AbstractProblem


"""Module for the SpatialProblem class"""


class SpatialProblem(AbstractProblem):
    """
    The class for the definition of spatial problems, i.e., for problems
    with spatial input variables.

    Here's an example of a spatial 1-dimensional ODE problem.

    :Example:
        >>> from .operators import grad
        >>> from .condition import Condition
        >>> from .span import Span
        >>> import torch
        >>> class SimpleODE(SpatialProblem):
        >>>     output_variables = ['u']
        >>>     spatial_domain = Span({'x': [0, 1]})
        >>>     def ode_equation(input_, output_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         return u_x - u
        >>> 
        >>>     def initial_condition(input_, output_):
        >>>         value = 1.0
        >>>         u = output_.extract(['u'])
        >>>         return u - value
        >>>
        >>>     conditions = {
        >>>         'x0': Condition(Span({'x': 0.}), initial_condition),
        >>>         'D': Condition(Span({'x': [0, 1]}), ode_equation)}

    """

    @abstractmethod
    def spatial_domain(self):
        """
        The spatial domain of the problem.
        """
        pass

    @property
    def spatial_variables(self):
        """
        The spatial input variables of the problem.
        """
        return self.spatial_domain.variables


"""Module for the TimeDependentSpatialProblem class"""


class TimeDependentProblem(AbstractProblem):
    """
    The class for the definition of time-dependent problems, i.e., for problems
    depending on time.

    Here's an example of a 1D wave problem.

    :Example:
        >>> from .operators import grad, nabla
        >>> from .condition import Condition
        >>> from .span import Span
        >>> import torch
        >>>
        >>> class Wave(TimeDependentSpatialProblem):
        >>>
        >>>     output_variables = ['u']
        >>>     spatial_domain = Span({'x': [0, 3]})
        >>>     temporal_domain = Span({'t': [0, 1]})
        >>>
        >>>     def wave_equation(input_, output_):
        >>>         u_t = grad(output_, input_, components=['u'], d=['t'])
        >>>         u_tt = grad(u_t, input_, components=['dudt'], d=['t'])
        >>>         nabla_u = nabla(output_, input_, components=['u'], d=['x'])
        >>>         return nabla_u - u_tt
        >>>
        >>>     def nil_dirichlet(input_, output_):
        >>>         value = 0.0
        >>>         return output_.extract(['u']) - value
        >>>
        >>>     def initial_condition(input_, output_):
        >>>         u_expected = (-3*torch.sin(2*torch.pi*input_.extract(['x']))
        >>>             + 5*torch.sin(8/3*torch.pi*input_.extract(['x'])))
        >>>         u = output_.extract(['u'])
        >>>         return u - u_expected
        >>>
        >>>     conditions = {
        >>>         't0': Condition(Span({'x': [0, 3], 't':0}), initial_condition),
        >>>         'gamma1': Condition(Span({'x':0, 't':[0, 1]}), nil_dirichlet),
        >>>         'gamma2': Condition(Span({'x':3, 't':[0, 1]}), nil_dirichlet),
        >>>         'D': Condition(Span({'x': [0, 3], 't':[0, 1]}), wave_equation)}

    """

    @abstractmethod
    def temporal_domain(self):
        """
        The temporal domain of the problem.
        """
        pass

    @property
    def temporal_variable(self):
        """
        The time variable of the problem.
        """
        return self.temporal_domain.variables


"""Module for the ParametricProblem class"""


class ParametricProblem(AbstractProblem):
    """
    The class for the definition of parametric problems, i.e., problems
    with parameters among the input variables.

    Here's an example of a spatial parametric ODE problem, i.e., a spatial
    ODE problem with an additional parameter `alpha` as coefficient of the
    derivative term.

    :Example:
        >>> from .operators import grad
        >>> from .condition import Condition
        >>> from .span import Span
        >>> import torch
        >>>
        >>> class ParametricODE(SpatialProblem, ParametricProblem):
        >>>
        >>>     output_variables = ['u']
        >>>     spatial_domain = Span({'x': [0, 1]})
        >>>     parameter_domain = Span({'alpha': [1, 10]})
        >>>
        >>>     def ode_equation(input_, output_):
        >>>         u_x = grad(output_, input_, components=['u'], d=['x'])
        >>>         u = output_.extract(['u'])
        >>>         alpha = input_.extract(['alpha'])
        >>>         return alpha * u_x - u
        >>>
        >>>     def initial_condition(input_, output_):
        >>>         value = 1.0
        >>>         u = output_.extract(['u'])
        >>>         return u - value
        >>>
        >>>     conditions = {
        >>>         'x0': Condition(Span({'x': 0, 'alpha':[1, 10]}), initial_condition),
        >>>         'D': Condition(Span({'x': [0, 1], 'alpha':[1, 10]}), ode_equation)}
    """

    @abstractmethod
    def parameter_domain(self):
        """
        The parameters' domain of the problem.
        """
        pass

    @property
    def parameters(self):
        """
        The parameters' variables of the problem.
        """
        return self.parameter_domain.variables
