# ML Wheel

A from-scratch machine learning library built purely for educational purposes. This project implements core ML components without relying on external libraries like NumPy or PyTorch.

## Purpose

This is primarily a learning exercise to understand the mathematical foundations of machine learning by implementing everything from scratch. Perfect for:

- Understanding how automatic differentiation works
- Learning the math behind matrix operations
- Exploring ML algorithms at a fundamental level
- Building intuition for optimization and gradients

## Current Components

### Math Module (`src/math/`)
- **Matrix**: Basic matrix operations (addition, multiplication, transpose)
- **Vector**: Vector operations (dot product, L2 norm, projection)
- **ActivationFunctions**: Neural network activation functions (ReLU, Sigmoid, Tanh)

### Autograd Module (`src/autograd/`)
- **ComputationGraph**: Foundation for automatic differentiation (placeholder)

### Random Module (`src/random/`)
- **Sampler**: Random sampling utilities (placeholder)

## Usage

```python
from src.math import Matrix, Vector, ActivationFunctions

# Create matrices
m1 = Matrix(2, 2)
m1.set_val_at(0, 0, 1)
m1.set_val_at(1, 1, 3)

m2 = Matrix(2, 2) 
m2.set_val_at(0, 0, 2)
m2.set_val_at(1, 1, 4)

# Matrix operations
result = Matrix.add(m1, m2)      # Addition
product = Matrix.multiply(m1, m2) # Matrix multiplication
transposed = Matrix.transpose(result) # Transpose

# Vector operations
v1 = Vector(3)
v1.set(0, 1)
v1.set(1, 2)

v2 = Vector(3)
v2.set(0, 3)
v2.set(1, 4)

dot_product = Vector.dot_product(v1, v2) # Dot product
norm = v1.l2_norm()                      # Euclidean norm
projection = Vector.project(v2, v1)      # Vector projection

# Activation functions
output = ActivationFunctions.relu(-2.5)    # → 0.0
output = ActivationFunctions.sigmoid(0.0)  # → 0.5
output = ActivationFunctions.tanh(1.0)     # → 0.7616...
```

## 🧪 Testing

Run the test suite to verify all operations:

```bash
python tests/test_math.py
```

The tests include:
- Matrix operations with known expected results
- Vector calculations with mathematical verification
- Activation function boundary conditions and mathematical properties
- Error handling for invalid operations
- Comprehensive edge case testing

## 📚 Learning Goals

- [x] Basic matrix and vector operations
- [x] Activation functions (ReLU, Sigmoid, Tanh)
- [ ] Automatic differentiation (backpropagation)
- [ ] Neural network primitives (layers, forward pass)
- [ ] Loss functions (MSE, Cross-entropy)
- [ ] Optimization algorithms (SGD, Adam)
- [ ] Gradient computation and backpropagation

## 🔧 Development

This project uses no external ML libraries by design. The only dependencies are for development tools:

```bash
# Install in development mode
pip install -e .[dev]

# Run tests
python tests/test_math.py

# Type checking
mypy src/

# Code formatting
black src/ tests/
```

## 📝 Notes

- Performance is intentionally not optimized - this is for learning, not production
- Implements basic operations to understand the underlying mathematics
- Focus is on clarity and educational value over efficiency
- Each component is built to be understandable and hackable

---

*"The best way to understand something is to build it from scratch."*
