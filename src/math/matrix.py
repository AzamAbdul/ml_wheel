from typing import Optional


class Matrix:
    """A basic matrix implementation for educational ML purposes."""
    
    def __init__(self, n: int, m: int) -> None:
        """Initialize matrix with n rows and m columns, filled with zeros."""
        self.rows = n
        self.cols = m
        self.matrix = [[0.0 for _ in range(m)] for _ in range(n)]
    
    def print(self) -> None:
        """Print the matrix dimensions and contents."""
        print(f"Num rows: {self.rows} Num Cols: {self.cols}")
        for row in self.matrix:
            print(row)

    def set_val_at(self, i: int, j: int, val: float) -> bool:
        """Set value at position (i, j). Returns True if successful, False if out of bounds."""
        if i >= self.rows or i < 0:
            return False
        if j >= self.cols or j < 0:
            return False
        self.matrix[i][j] = val
        return True

    def get_val_at(self, i: int, j: int) -> Optional[float]:
        """Get value at position (i, j). Returns None if out of bounds."""
        if i >= self.rows or i < 0:
            return None
        if j >= self.cols or j < 0:
            return None
        return self.matrix[i][j]

    def get_rows(self) -> int:
        """Get number of rows."""
        return self.rows
    
    def get_cols(self) -> int:
        """Get number of columns."""
        return self.cols

    @staticmethod
    def add(m1: 'Matrix', m2: 'Matrix', subtract: bool = False) -> Optional['Matrix']:
        """Add or subtract two matrices element-wise.
        
        Args:
            m1: First matrix
            m2: Second matrix
            subtract: If True, performs subtraction (m1 - m2)
            
        Returns:
            Result matrix or None if dimensions don't match
        """
        sign = 1
        if subtract:
            sign = -1
        
        # Check dimensions match
        if m1.get_rows() != m2.get_rows():
            return None
        if m1.get_cols() != m2.get_cols():
            return None
        
        result = Matrix(m1.get_rows(), m1.get_cols())
        for i in range(0, m1.get_rows()):
            for j in range(0, m1.get_cols()):
                v1 = m1.get_val_at(i, j)
                v2 = sign * m2.get_val_at(i, j)
                result.set_val_at(i, j, v1 + v2)
        return result
    
    @staticmethod
    def multiply(m1: 'Matrix', m2: 'Matrix') -> Optional['Matrix']:
        """Matrix multiplication using dot product.
        
        Args:
            m1: Left matrix
            m2: Right matrix
            
        Returns:
            Result matrix or None if dimensions incompatible
        """
        # Check if multiplication is valid (m1 cols must equal m2 rows)
        if m1.get_cols() != m2.get_rows():
            return None
            
        num_rows = m1.get_rows()
        num_cols = m2.get_cols()
        result = Matrix(num_rows, num_cols)
        
        for i in range(num_rows):
            for j in range(num_cols):
                res = 0
                # Dot product of row i from m1 and column j from m2
                for k in range(m2.get_rows()):
                    res += m1.get_val_at(i, k) * m2.get_val_at(k, j)
                result.set_val_at(i, j, res)
        return result

    @staticmethod
    def transpose(m1: 'Matrix') -> 'Matrix':
        """Transpose a matrix (flip rows and columns).
        
        Args:
            m1: Matrix to transpose
            
        Returns:
            Transposed matrix
        """
        prior_num_rows = m1.get_rows()
        prior_num_cols = m1.get_cols()
        
        # Transposed matrix has flipped dimensions
        new_num_rows = prior_num_cols
        new_num_cols = prior_num_rows

        result = Matrix(new_num_rows, new_num_cols)

        # Swap indices: result[j][i] = original[i][j]
        for i in range(prior_num_rows):
            for j in range(prior_num_cols):
                v = m1.get_val_at(i, j)
                result.set_val_at(j, i, v)
        return result
