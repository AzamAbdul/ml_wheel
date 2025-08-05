"""Tests for math module - extracted from original implementation with assertions."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.math.matrix import Matrix
from src.math.vector import Vector
from src.math.activation_functions import ActivationFunctions


def test_matrix_operations():
    """Test matrix operations as originally implemented."""
    print("Testing Matrix Operations")
    print("=" * 40)
    
    # Create first matrix: [[1, 0], [0, 3]]
    m1 = Matrix(2, 2)
    m1.set_val_at(0, 0, 1)
    m1.set_val_at(1, 1, 3)
    assert m1.get_val_at(0, 0) == 1
    assert m1.get_val_at(1, 1) == 3
    assert m1.get_val_at(0, 1) == 0  # Should be 0 (default)
    print("Matrix m1:")
    m1.print()
    print()

    # Create second matrix: [[1, 10], [3, 4]]
    m2 = Matrix(2, 2)
    m2.set_val_at(0, 0, 1)
    m2.set_val_at(1, 1, 4)
    m2.set_val_at(0, 1, 10)
    m2.set_val_at(1, 0, 3)
    print("Matrix m2:")
    m2.print()
    print()

    # Test addition: [[2, 10], [3, 7]]
    print("m1 + m2:")
    result_add = Matrix.add(m1, m2)
    assert result_add.get_val_at(0, 0) == 2  # 1 + 1
    assert result_add.get_val_at(0, 1) == 10  # 0 + 10
    assert result_add.get_val_at(1, 0) == 3   # 0 + 3
    assert result_add.get_val_at(1, 1) == 7   # 3 + 4
    result_add.print()
    print()

    # Test subtraction: [[0, -10], [-3, -1]]
    print("m1 - m2:")
    result_sub = Matrix.add(m1, m2, subtract=True)
    assert result_sub.get_val_at(0, 0) == 0   # 1 - 1
    assert result_sub.get_val_at(0, 1) == -10 # 0 - 10
    assert result_sub.get_val_at(1, 0) == -3  # 0 - 3
    assert result_sub.get_val_at(1, 1) == -1  # 3 - 4
    result_sub.print()
    print()

    # Test multiplication: [[1, 10], [9, 12]]
    print("m1 * m2:")
    m3 = Matrix.multiply(m1, m2)
    assert m3.get_val_at(0, 0) == 1   # 1*1 + 0*3 = 1
    assert m3.get_val_at(0, 1) == 10  # 1*10 + 0*4 = 10
    assert m3.get_val_at(1, 0) == 9   # 0*1 + 3*3 = 9
    assert m3.get_val_at(1, 1) == 12  # 0*10 + 3*4 = 12
    m3.print()
    print()

    # Test transpose: [[1, 9], [10, 12]]
    print("transpose m3:")
    transposed = Matrix.transpose(m3)
    assert transposed.get_val_at(0, 0) == 1   # m3[0,0]
    assert transposed.get_val_at(0, 1) == 9   # m3[1,0]
    assert transposed.get_val_at(1, 0) == 10  # m3[0,1]
    assert transposed.get_val_at(1, 1) == 12  # m3[1,1]
    transposed.print()
    print()


def test_vector_operations():
    """Test vector operations as originally implemented."""
    print("Testing Vector Operations")
    print("=" * 40)
    
    # Create first vector: [0, 1, 2, 0, 0]
    v = Vector(5)
    v.set(1, 1)
    v.set(2, 2)
    assert v.get(0) == 0
    assert v.get(1) == 1
    assert v.get(2) == 2
    assert v.size() == 5
    print("Vector v:")
    v.print()
    
    # Test dot product with itself: 0² + 1² + 2² + 0² + 0² = 5
    dot_self = Vector.dot_product(v, v)
    assert dot_self == 5
    print(f"v · v = {dot_self}")
    
    # Test L2 norm: sqrt(5) ≈ 2.236
    norm_v = v.l2_norm()
    assert abs(norm_v - 2.23606797749979) < 1e-10  # sqrt(5)
    print(f"||v||₂ = {norm_v}")
    print()

    # Create second vector: [3, 4, 3, 0, 0]
    v2 = Vector(5)
    v2.set(0, 3)
    v2.set(1, 4)
    v2.set(2, 3)
    print("Vector v2:")
    v2.print()
    
    # Test L2 norm of v2: sqrt(3² + 4² + 3²) = sqrt(34) ≈ 5.831
    norm_v2 = v2.l2_norm()
    expected_norm_v2 = (3**2 + 4**2 + 3**2)**0.5  # sqrt(34)
    assert abs(norm_v2 - expected_norm_v2) < 1e-10
    print(f"||v2||₂ = {norm_v2}")
    
    # Test dot product: 0*3 + 1*4 + 2*3 + 0*0 + 0*0 = 10
    dot_product = Vector.dot_product(v, v2)
    assert dot_product == 10
    print(f"v · v2 = {dot_product}")
    
    # Test projection of v onto v2
    print("Projection of v onto v2:")
    projection = Vector.project(v2, v)
    # proj_v2(v) = (v·v2 / ||v2||²) * v2
    # = (10 / 34) * [3, 4, 3, 0, 0]
    scale_factor = 10 / 34
    expected_proj = [3 * scale_factor, 4 * scale_factor, 3 * scale_factor, 0, 0]
    
    for i in range(5):
        assert abs(projection.get(i) - expected_proj[i]) < 1e-10
    
    projection.print()
    print()


def test_error_cases():
    """Test error handling."""
    print("Testing Error Cases")
    print("=" * 40)
    
    # Test mismatched matrix dimensions for addition
    m1 = Matrix(2, 3)
    m2 = Matrix(3, 2)
    
    # This should raise ValueError (dimension mismatch for addition)
    try:
        result = Matrix.add(m1, m2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Adding 2x3 and 3x2 matrices correctly raised: {type(e).__name__}")
    
    # This should work for multiplication (2x3 * 3x2 = 2x2)
    result = Matrix.multiply(m1, m2)
    assert result.get_rows() == 2
    assert result.get_cols() == 2
    print(f"Multiplying 2x3 by 3x2 matrices: Success")
    
    # Test invalid multiplication dimensions
    m3 = Matrix(2, 3)
    m4 = Matrix(2, 3)  # Can't multiply 2x3 by 2x3
    try:
        result = Matrix.multiply(m3, m4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Invalid multiplication (2x3 * 2x3) correctly raised: {type(e).__name__}")
    
    # Test out of bounds access
    m = Matrix(2, 2)
    try:
        val = m.get_val_at(5, 5)  # Out of bounds
        assert False, "Should have raised IndexError"
    except IndexError as e:
        print(f"Out of bounds access correctly raised: {type(e).__name__}")
    
    # Test out of bounds setting
    try:
        m.set_val_at(-1, 0, 5)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        m.set_val_at(0, 10, 5)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    print("Out of bounds setting: Correctly rejected")
    
    # Test vector dimension mismatch
    v1 = Vector(3)
    v2 = Vector(5)
    try:
        dot = Vector.dot_product(v1, v2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Dot product of different sized vectors correctly raised: {type(e).__name__}")
    
    # Test vector out of bounds
    v = Vector(3)
    try:
        val = v.get(10)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        v.set(-1, 5)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    print("Vector out of bounds access: Correctly handled")
    print()


def test_activation_functions():
    """Test activation functions."""
    print("Testing Activation Functions")
    print("=" * 40)
    
    # Test ReLU
    assert ActivationFunctions.relu(5.0) == 5.0
    assert ActivationFunctions.relu(-3.0) == 0.0
    assert ActivationFunctions.relu(0.0) == 0.0
    print("✓ ReLU tests passed")
    
    # Test Sigmoid
    sig_0 = ActivationFunctions.sigmoid(0.0)
    assert abs(sig_0 - 0.5) < 1e-10  # sigmoid(0) = 0.5
    
    sig_large = ActivationFunctions.sigmoid(100.0)
    assert abs(sig_large - 1.0) < 1e-10  # sigmoid(large) ≈ 1
    
    sig_negative = ActivationFunctions.sigmoid(-100.0)
    assert abs(sig_negative - 0.0) < 1e-10  # sigmoid(-large) ≈ 0
    print("✓ Sigmoid tests passed")
    
    # Test Tanh
    tanh_0 = ActivationFunctions.tanh(0.0)
    assert abs(tanh_0 - 0.0) < 1e-10  # tanh(0) = 0
    
    tanh_large = ActivationFunctions.tanh(100.0)
    assert abs(tanh_large - 1.0) < 1e-10  # tanh(large) ≈ 1
    
    tanh_negative = ActivationFunctions.tanh(-100.0)
    assert abs(tanh_negative - (-1.0)) < 1e-10  # tanh(-large) ≈ -1
    print("✓ Tanh tests passed")
    print()


def run_all_tests():
    """Run all tests and report results."""
    try:
        test_matrix_operations()
        print("✓ Matrix operations tests passed")
        
        test_vector_operations()
        print("✓ Vector operations tests passed")
        
        test_error_cases()
        print("✓ Error handling tests passed")
        
        test_activation_functions()
        print("✓ Activation functions tests passed")
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! 🎉")
        
    except AssertionError as e:
        print(f"\n❌ Test failed with assertion error: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()