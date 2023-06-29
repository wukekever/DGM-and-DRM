"""Module for Location class."""

from abc import ABCMeta, abstractmethod


class Location(metaclass=ABCMeta):
    """
    Abstract Location class.
    Any geometry entity should inherit from this class.
    """
    @property
    @abstractmethod
    def sample(self):
        """
        Abstract method for sampling a point from the location. To be
        implemented in the child class.
        """
        pass