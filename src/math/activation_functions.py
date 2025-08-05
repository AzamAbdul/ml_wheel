"""Activation functions for neural networks."""

import math
from typing import Union


class ActivationFunctions:
    """Collection of common activation functions used in neural networks."""
    
    @staticmethod
    def relu(x: float) -> float:
        """Rectified Linear Unit activation function.
        
        Args:
            x: Input value
            
        Returns:
            max(0, x)
        """
        return max(0.0, x)
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid activation function.
        
        Args:
            x: Input value
            
        Returns:
            1 / (1 + exp(-x))
        """
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def tanh(x: float) -> float:
        """Hyperbolic tangent activation function.
        
        Args:
            x: Input value
            
        Returns:
            tanh(x)
        """
        return math.tanh(x)
