"""ML Wheel: A from-scratch machine learning library for educational purposes."""

__version__ = "0.1.0"
__author__ = "Azam"

from .math import Matrix, Vector
from .autograd import ComputationGraph
from .random import Sampler

__all__ = ["Matrix", "Vector", "ComputationGraph", "Sampler"]