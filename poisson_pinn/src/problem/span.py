from problem.chebyshev import chebyshev_roots
import torch

from problem.location import Location
from problem.label_tensor import LabelTensor
from problem.utils import torch_lhs


class Span(Location):
    """PINN implementation of Hypercube domain."""

    def __init__(self, span_dict):
        """
        :param span_dict: A dictionary with dict-key a string representing
            the input variables for the pinn, and dict-value a list with
            the domain extrema.
        :type span_dict: dict

        :Example:
            >>> spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
        """
        self.fixed_ = {}
        self.range_ = {}

        for k, v in span_dict.items():
            if isinstance(v, (int, float)):
                self.fixed_[k] = v
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                self.range_[k] = v
            else:
                raise TypeError

    @property
    def variables(self):
        """Spatial variables.

        :return: Spatial variables defined in '__init__()'
        :rtype: list[str]
        """
        return list(self.fixed_.keys()) + list(self.range_.keys())

    def update(self, new_span):
        """Adding new dimensions on the span

        :param new_span: A new span object to merge
        :type new_span: Span

        :Example:
            >>> spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
            >>> spatial_domain.variables
            ['x', 'y']
            >>> spatial_domain_2 = Span({'z': [3, 4], 'w': [0, 1]})
            >>> spatial_domain.update(spatial_domain_2)
            >>> spatial_domain.variables
            ['x', 'y', 'z', 'w']
        """
        self.fixed_.update(new_span.fixed_)
        self.range_.update(new_span.range_)

    def _sample_range(self, n, mode, bounds):
        """Rescale the samples to the correct bounds

        :param n: Number of points to sample, see Note below
            for reference.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: random sampling, 'random';
            latin hypercube sampling, 'latin' or 'lh';
            chebyshev sampling, 'chebyshev'; grid sampling 'grid'.
        :type mode: str, optional
        :param bounds: Bounds to rescale the samples.
        :type bounds: torch.Tensor
        :return: Rescaled sample points.
        :rtype: torch.Tensor
        """
        dim = bounds.shape[0]
        if mode in ['chebyshev', 'grid'] and dim != 1:
            raise RuntimeError('Something wrong in Span...')

        if mode == 'random':
            pts = torch.rand(size=(n, dim))
        elif mode == 'chebyshev':
            pts = chebyshev_roots(n).mul(.5).add(.5).reshape(-1, 1)
        elif mode == 'grid':
            pts = torch.linspace(0, 1, n).reshape(-1, 1)
        elif mode == 'lh' or mode == 'latin':
            pts = torch_lhs(n, dim)

        pts *= bounds[:, 1] - bounds[:, 0]
        pts += bounds[:, 0]

        return pts

    def sample(self, n, mode='random', variables='all'):
        """Sample routine.

        :param n: Number of points to sample, see Note below
            for reference.
        :type n: int
        :param mode: Mode for sampling, defaults to 'random'.
            Available modes include: random sampling, 'random';
            latin hypercube sampling, 'latin' or 'lh';
            chebyshev sampling, 'chebyshev'; grid sampling 'grid'.
        :type mode: str, optional
        :param variables: pinn variable to be sampled, defaults to 'all'.
        :type variables: str or list[str], optional

        .. note::
            The total number of points sampled in case of multiple variables
            is not 'n', and it depends on the chosen 'mode'. If 'mode' is
            'grid' or 'chebyshev', the points are sampled independentely
            across the variables and the results crossed together, i.e. the
            final number of points is 'n' to the power of the number of
            variables. If 'mode' is 'random', 'lh' or 'latin', the variables
            are sampled all together, and the final number of points

        .. warning::
            The extrema values of Span are always sampled only for 'grid' mode.

        :Example:
            >>> spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
            >>> spatial_domain.sample(n=4, mode='random')
                tensor([[0.0108, 0.7643],
                        [0.4477, 0.8015],
                        [0.2063, 0.8087],
                        [0.8735, 0.6349]])
            >>> spatial_domain.sample(n=4, mode='grid')
                tensor([[0.0000, 0.0000],
                        [0.3333, 0.0000],
                        [0.6667, 0.0000],
                        [1.0000, 0.0000],
                        [0.0000, 0.3333],
                        [0.3333, 0.3333],
                        [0.6667, 0.3333],
                        [1.0000, 0.3333],
                        [0.0000, 0.6667],
                        [0.3333, 0.6667],
                        [0.6667, 0.6667],
                        [1.0000, 0.6667],
                        [0.0000, 1.0000],
                        [0.3333, 1.0000],
                        [0.6667, 1.0000],
                        [1.0000, 1.0000]])
        """
        def _1d_sampler(n, mode, variables):
            """ Sample independentely the variables and cross the results"""
            tmp = []
            for variable in variables:
                if variable in self.range_.keys():
                    bound = torch.tensor([self.range_[variable]])
                    pts_variable = self._sample_range(n, mode, bound)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode='cross')

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode='std')

            return result

        def _Nd_sampler(n, mode, variables):
            """Sample all the variables together

            :param n: Number of points to sample.
            :type n: int
            :param mode: Mode for sampling, defaults to 'random'.
                Available modes include: random sampling, 'random';
                latin hypercube sampling, 'latin' or 'lh';
                chebyshev sampling, 'chebyshev'; grid sampling 'grid'.
            :type mode: str, optional.
            :param variables: pinn variable to be sampled, defaults to 'all'.
            :type variables: str or list[str], optional.
            :return: Sample points.
            :rtype: list[torch.Tensor]
            """
            pairs = [(k, v) for k, v in self.range_.items() if k in variables]
            keys, values = map(list, zip(*pairs))
            bounds = torch.tensor(values)
            result = self._sample_range(n, mode, bounds)
            result = result.as_subclass(LabelTensor)
            result.labels = keys

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode='std')
            return result

        def _single_points_sample(n, variables):
            """Sample a single point in one dimension.

            :param n: Number of points to sample.
            :type n: int
            :param variables: Variables to sample from.
            :type variables: list[str]
            :return: Sample points.
            :rtype: list[torch.Tensor]
            """
            tmp = []
            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(n, 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]
                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode='std')

            return result

        if self.fixed_ and (not self.range_):
            return _single_points_sample(n, variables)

        if variables == 'all':
            variables = list(self.range_.keys()) + list(self.fixed_.keys())

        if mode in ['grid', 'chebyshev']:
            return _1d_sampler(n, mode, variables)
        elif mode in ['random', 'lh', 'latin']:
            return _Nd_sampler(n, mode, variables)
        else:
            raise ValueError(f'mode={mode} is not valid.')
