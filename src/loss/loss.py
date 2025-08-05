"""Loss functions for training machine learning models."""

from ..math.matrix import Matrix


class Loss:
    """Collection of loss functions for training and evaluation."""

    @staticmethod
    def mse_loss(predictions: Matrix, actual: Matrix) -> float:
        """Compute Mean Squared Error between predictions and actual values.
        
        Args:
            predictions: Matrix of predicted values
            actual: Matrix of actual/target values
            
        Returns:
            Mean squared error as float
            
        Raises:
            ValueError: If matrix dimensions don't match
        """
        num_cols = predictions.get_cols()
        num_rows = predictions.get_rows()
        if num_cols != actual.get_cols():
            raise ValueError("Matrices don't have the same col size")
        if num_rows != actual.get_rows():
            raise ValueError("Matrices don't have the same row size")
        squared_sum_diff = 0
        for i in range(0, num_rows):
            for j in range(0, num_cols):
                squared_sum_diff += (predictions.get_val_at(i, j) - actual.get_val_at(i, j))**2
        return squared_sum_diff / (num_rows * num_cols)


