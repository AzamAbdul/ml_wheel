import math
# Type annotations only using built-in types


class Vector:
    """A basic vector implementation for educational ML purposes."""
    
    def __init__(self, n: int) -> None:
        """Initialize vector with n dimensions, filled with zeros."""
        self.vector = [0.0 for _ in range(n)]

    def size(self) -> int:
        """Get the size/dimension of the vector."""
        return len(self.vector)

    def get(self, i: int) -> float:
        """Get value at index i. Raises IndexError if out of bounds."""
        if i >= len(self.vector) or i < 0:
            raise IndexError(f"Index {i} out of bounds for vector of size {len(self.vector)}")
        return self.vector[i]

    def set(self, i: int, v: float) -> None:
        """Set value at index i. Raises IndexError if out of bounds."""
        if i >= len(self.vector) or i < 0:
            raise IndexError(f"Index {i} out of bounds for vector of size {len(self.vector)}")
        self.vector[i] = v

    @staticmethod
    def dot_product(v1: 'Vector', v2: 'Vector') -> float:
        """Compute dot product of two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Dot product
            
        Raises:
            ValueError: If vector dimensions don't match
        """
        if v1.size() != v2.size():
            raise ValueError(f"Cannot compute dot product of vectors with sizes {v1.size()} and {v2.size()}")
        
        result = 0.0
        for i in range(v1.size()):
            result += v1.get(i) * v2.get(i)
        return result

    @staticmethod
    def apply_scalar(v: 'Vector', scalar: float) -> 'Vector':
        """Multiply vector by scalar value.
        
        Args:
            v: Input vector
            scalar: Scalar value to multiply by
            
        Returns:
            New vector with scaled values
        """
        result = Vector(v.size())
        for i in range(v.size()):
            result.set(i, v.get(i) * scalar)
        return result

    @staticmethod
    def project(onto: 'Vector', orig: 'Vector') -> 'Vector':
        """Project vector 'orig' onto vector 'onto'.
        
        Args:
            onto: Vector to project onto
            orig: Vector being projected
            
        Returns:
            Projected vector
        """
        onto_magnitude = onto.l2_norm()
        dot_prod = Vector.dot_product(onto, orig)

        # Projection formula: proj_onto(orig) = (orig·onto / |onto|²) * onto
        scale = dot_prod / onto_magnitude ** 2
        return Vector.apply_scalar(onto, scale)


    def l2_norm(self) -> float:
        """Compute L2 norm (Euclidean length) of the vector.
        
        Returns:
            L2 norm of the vector
        """
        sum_squares = 0.0
        for i in range(self.size()):
            sum_squares += self.get(i) ** 2
        return math.sqrt(sum_squares)
    
    def print(self) -> None:
        """Print the vector contents."""
        print(self.vector)
