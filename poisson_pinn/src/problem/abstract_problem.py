""" Module for AbstractProblem class """
from abc import ABCMeta, abstractmethod


class AbstractProblem(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINN Problem
    should be inheritied from this class.

    In the definition of a PINN problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """
    @property
    def input_variables(self):
        """
        The input variables of the AbstractProblem, whose type depends on the
        type of domain (spatial, temporal, and parameter).

        :return: the input variables of self
        :rtype: list

        """
        variables = []

        if hasattr(self, 'spatial_variables'):
            variables += self.spatial_variables
        if hasattr(self, 'temporal_variable'):
            variables += self.temporal_variable
        if hasattr(self, 'parameters'):
            variables += self.parameters
        if hasattr(self, 'custom_variables'):
            variables += self.custom_variables

        return variables

    @property
    def domain(self):
        """
        The domain(s) where the conditions of the AbstractProblem are valid.

        :return: the domain(s) of self
        :rtype: list (if more than one domain are defined),
            `Span` domain (of only one domain is defined)
        """
        domains = [
            getattr(self, f'{t}_domain')
            for t in ['spatial', 'temporal', 'parameter']
            if hasattr(self, f'{t}_domain')
        ]

        if len(domains) == 1:
            return domains[0]
        elif len(domains) == 0:
            raise RuntimeError

        if len(set(map(type, domains))) == 1:
            domain = domains[0].__class__({})
            [domain.update(d) for d in domains]
            return domain
        else:
            raise RuntimeError('different domains')

    @input_variables.setter
    def input_variables(self, variables):
        raise RuntimeError

    @property
    @abstractmethod
    def output_variables(self):
        """
        The output variables of the problem.
        """
        pass

    @property
    @abstractmethod
    def conditions(self):
        """
        The conditions of the problem.
        """
        pass
