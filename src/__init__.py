"""ML Wheel: A from-scratch machine learning library for educational purposes."""

__version__ = "0.1.0"
__author__ = "Azam"

from .math import Matrix, Vector, ActivationFunctions
from .autograd import ComputationGraph

__all__ = ["Matrix", "Vector", "ActivationFunctions", "ComputationGraph"]