# XOR Neural Network Implementations
# Based on: https://github.com/ceasedfonts/NNBasics/blob/main/very_simple_neural_network.py
#
# This file contains THREE implementations:
# 1. NumPy-based implementation (original)
# 2. Pure Python implementation (no external dependencies)
# 3. Propagator-based implementation using the propagator network model with TMS

import math
import random

# =============================================================================
# NUMPY VERSION
# =============================================================================

try:
    import numpy as np
    import matplotlib.pyplot as plt
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

def sigmoid_np(x):
    """Sigmoid activation function using NumPy."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative_np(x):
    """Derivative of sigmoid function (assumes x is already sigmoid output)."""
    return x * (1 - x)

class NeuralNetworkNumPy:
    """
    A simple neural network with one hidden layer for solving the XOR problem.
    Uses NumPy for matrix operations.
    
    Architecture:
    - Input layer: 2 neurons
    - Hidden layer: configurable size (default 2)
    - Output layer: 1 neuron
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    def forward(self, inputs):
        """Forward pass: compute output from inputs."""
        # Input to hidden layer
        hidden_input = np.dot(inputs, self.weights_input_hidden)
        hidden_output = sigmoid_np(hidden_input)
        
        # Hidden to output layer
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        output = sigmoid_np(output_input)
        
        return output
    
    def train(self, inputs, targets, epochs, learning_rate=1.0):
        """
        Train the network using backpropagation.
        
        The training combines:
        - Forward propagation
        - Error calculation
        - Backpropagation
        - Weight updates via gradient descent
        """
        loss_history = []
        
        for _ in range(epochs):
            # Forward pass
            hidden_input = np.dot(inputs, self.weights_input_hidden)
            hidden_output = sigmoid_np(hidden_input)
            output_input = np.dot(hidden_output, self.weights_hidden_output)
            output = sigmoid_np(output_input)
            
            # Calculate error
            error = targets - output
            
            # Backpropagation
            d_output = error * sigmoid_derivative_np(output)
            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * sigmoid_derivative_np(hidden_output)
            
            # Update weights
            self.weights_hidden_output += learning_rate * hidden_output.T.dot(d_output)
            self.weights_input_hidden += learning_rate * inputs.T.dot(d_hidden)
            
            # Track loss (Mean Squared Error)
            loss = np.mean(np.square(error))
            loss_history.append(loss)
        
        return loss_history


# =============================================================================
# PURE PYTHON VERSION (No NumPy)
# =============================================================================

def sigmoid(x):
    """Sigmoid activation function using pure Python."""
    # Clamp to avoid overflow
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid (assumes x is already sigmoid output)."""
    return x * (1 - x)

def dot_product(vec1, vec2):
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))

def matrix_vector_mult(matrix, vector):
    """Multiply a matrix by a vector. Matrix is list of rows."""
    return [dot_product(row, vector) for row in matrix]

def outer_product(vec1, vec2):
    """Compute outer product of two vectors, returning a matrix."""
    return [[a * b for b in vec2] for a in vec1]

def transpose(matrix):
    """Transpose a matrix (list of lists)."""
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_add(m1, m2):
    """Add two matrices element-wise."""
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

def scalar_mult_matrix(scalar, matrix):
    """Multiply a matrix by a scalar."""
    return [[scalar * val for val in row] for row in matrix]


class NeuralNetworkPython:
    """
    A simple neural network with one hidden layer for solving the XOR problem.
    Uses only pure Python data structures (lists) - no NumPy required.
    
    Architecture:
    - Input layer: 2 neurons
    - Hidden layer: configurable size (default 2)
    - Output layer: 1 neuron
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        # weights_input_hidden: input_size x hidden_size matrix
        self.weights_input_hidden = [
            [random.random() for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        # weights_hidden_output: hidden_size x output_size matrix
        self.weights_hidden_output = [
            [random.random() for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, inputs):
        """
        Forward pass: compute output from inputs.
        
        Args:
            inputs: list of input values (length = input_size)
            
        Returns:
            list of output values (length = output_size)
        """
        # Input to hidden layer: multiply inputs by weights, apply activation
        hidden_input = matrix_vector_mult(transpose(self.weights_input_hidden), inputs)
        hidden_output = [sigmoid(x) for x in hidden_input]
        
        # Hidden to output layer
        output_input = matrix_vector_mult(transpose(self.weights_hidden_output), hidden_output)
        output = [sigmoid(x) for x in output_input]
        
        return output
    
    def train(self, inputs_batch, targets_batch, epochs, learning_rate=1.0):
        """
        Train the network using backpropagation.
        
        Args:
            inputs_batch: list of input samples, each sample is a list
            targets_batch: list of target outputs, each target is a list
            epochs: number of training iterations
            learning_rate: step size for weight updates
            
        Returns:
            loss_history: list of MSE loss values for each epoch
        """
        loss_history = []
        
        for _ in range(epochs):
            total_error_squared = 0
            
            # Accumulate gradients for batch
            grad_hidden_output = [[0.0] * self.output_size for _ in range(self.hidden_size)]
            grad_input_hidden = [[0.0] * self.hidden_size for _ in range(self.input_size)]
            
            for inputs, targets in zip(inputs_batch, targets_batch):
                # Forward pass
                hidden_input = matrix_vector_mult(transpose(self.weights_input_hidden), inputs)
                hidden_output = [sigmoid(x) for x in hidden_input]
                
                output_input = matrix_vector_mult(transpose(self.weights_hidden_output), hidden_output)
                output = [sigmoid(x) for x in output_input]
                
                # Calculate error
                error = [t - o for t, o in zip(targets, output)]
                total_error_squared += sum(e * e for e in error)
                
                # Backpropagation - output layer
                d_output = [e * sigmoid_derivative(o) for e, o in zip(error, output)]
                
                # Backpropagation - hidden layer
                # error_hidden = d_output * weights_hidden_output^T
                error_hidden = []
                for h in range(self.hidden_size):
                    err = sum(d_output[o] * self.weights_hidden_output[h][o] 
                              for o in range(self.output_size))
                    error_hidden.append(err)
                
                d_hidden = [eh * sigmoid_derivative(ho) 
                            for eh, ho in zip(error_hidden, hidden_output)]
                
                # Accumulate gradients
                # grad_hidden_output += outer_product(hidden_output, d_output)
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        grad_hidden_output[h][o] += hidden_output[h] * d_output[o]
                
                # grad_input_hidden += outer_product(inputs, d_hidden)
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        grad_input_hidden[i][h] += inputs[i] * d_hidden[h]
            
            # Update weights (using accumulated gradients)
            for h in range(self.hidden_size):
                for o in range(self.output_size):
                    self.weights_hidden_output[h][o] += learning_rate * grad_hidden_output[h][o]
            
            for i in range(self.input_size):
                for h in range(self.hidden_size):
                    self.weights_input_hidden[i][h] += learning_rate * grad_input_hidden[i][h]
            
            # Track loss (Mean Squared Error)
            loss = total_error_squared / len(inputs_batch)
            loss_history.append(loss)
        
        return loss_history


# =============================================================================
# PROPAGATOR-BASED NEURAL NETWORK (Using Propagator Network Model with TMS)
# =============================================================================
"""
This implementation uses the propagator network model to build a neural network.

Key concepts:
- Cells hold values and support dependencies via TMS (Truth Maintenance System)
- Propagators are computational units that increase information in cells
- Training data is tagged with 'training-data' premise for provenance tracking
- Weights are stored in cells and can be updated through propagation

The propagator model is particularly interesting for neural networks because:
1. It provides automatic dependency tracking via TMS
2. It supports bidirectional computation (though we primarily use forward here)
3. Information from multiple sources merges automatically
"""

from propagator import (
    Cell,
    compound_propagator,
    constant,
    function_to_propagator_constructor,
    adder,
    subtractor,
    multiplier,
    divider,
    run,
    initialize_scheduler,
)
from propagator.merge import make_generic_operator, assign_merge_operation
from propagator.supported_values import supported, Supported, supported_p, flat_p
from propagator.tms import bring_in, kick_out, make_tms, tms_query, Tms


# -----------------------------------------------------------------------------
# Generic operators for neural network computations
# -----------------------------------------------------------------------------

def _sigmoid_func(x):
    """Sigmoid activation function."""
    if x is None:
        return None
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def _sigmoid_derivative_func(x):
    """Derivative of sigmoid (assumes x is already sigmoid output)."""
    if x is None:
        return None
    return x * (1 - x)

# Create generic operators for sigmoid and its derivative
generic_sigmoid = make_generic_operator(1, 'sigmoid', _sigmoid_func)
generic_sigmoid_derivative = make_generic_operator(1, 'sigmoid_derivative', _sigmoid_derivative_func)

# Create propagator constructors from these operators
sigmoid_propagator = function_to_propagator_constructor(generic_sigmoid)
sigmoid_deriv_propagator = function_to_propagator_constructor(generic_sigmoid_derivative)


# Register operators for Supported values
def supported_unary(op):
    """Lift a unary operator to work with Supported values."""
    def wrapper(vs):
        if isinstance(vs, Supported):
            return supported(op(vs.value), vs.support)
        return op(vs)
    return wrapper

generic_sigmoid.assign_operation(supported_unary(generic_sigmoid), supported_p)
generic_sigmoid_derivative.assign_operation(supported_unary(generic_sigmoid_derivative), supported_p)

# Register operators for TMS values
# Import TMS-related functions for handling TMS types
from propagator.tms import tms_p, to_tms, full_tms_unpacking, coercing

# Register TMS handlers for sigmoid operations
generic_sigmoid.assign_operation(full_tms_unpacking(generic_sigmoid), tms_p)
generic_sigmoid_derivative.assign_operation(full_tms_unpacking(generic_sigmoid_derivative), tms_p)


class NeuralNetworkPropagator:
    """
    A neural network implemented using propagators.
    
    This implementation demonstrates how propagator networks can be used
    for neural computation. Training data is tagged with TMS premises
    to track data provenance.
    
    Architecture:
    - Input layer: 2 neurons
    - Hidden layer: configurable size
    - Output layer: 1 neuron
    """
    
    def __init__(self, input_size, hidden_size, output_size, seed=None):
        if seed is not None:
            random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Store weights as regular Python values (updated during training)
        self.weights_input_hidden = [
            [random.random() for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.weights_hidden_output = [
            [random.random() for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
    
    def forward_propagator(self, input_values, premise_tag=None):
        """
        Build a propagator network for forward pass and return output cell.
        
        Args:
            input_values: List of input values [x1, x2]
            premise_tag: Optional tag for TMS support
            
        Returns:
            output_cell: Cell containing the network output
        """
        initialize_scheduler()
        
        # Create input cells with optional TMS tagging
        input_cells = []
        for i, val in enumerate(input_values):
            cell = Cell(name=f"input_{i}")
            if premise_tag:
                cell.add_content(make_tms(supported(val, [premise_tag])))
            else:
                cell.add_content(val)
            input_cells.append(cell)
        
        # Hidden layer computation
        hidden_cells = []
        for h in range(self.hidden_size):
            # Weighted sum: sum(input_i * weight_i_h)
            weighted_sum_cell = Cell(name=f"hidden_sum_{h}")
            
            # Accumulate weighted inputs
            current_sum = Cell(name=f"hidden_acc_{h}_init")
            constant(0.0, current_sum)
            
            for i in range(self.input_size):
                weight_cell = Cell(name=f"w_ih_{i}_{h}")
                constant(self.weights_input_hidden[i][h], weight_cell)
                
                product_cell = Cell(name=f"prod_ih_{i}_{h}")
                multiplier(input_cells[i], weight_cell, product_cell)
                
                new_sum = Cell(name=f"hidden_acc_{h}_{i}")
                adder(current_sum, product_cell, new_sum)
                current_sum = new_sum
            
            # Apply sigmoid activation
            hidden_output = Cell(name=f"hidden_{h}")
            sigmoid_propagator(current_sum, hidden_output)
            hidden_cells.append(hidden_output)
        
        # Output layer computation
        output_cells = []
        for o in range(self.output_size):
            # Weighted sum of hidden outputs
            current_sum = Cell(name=f"output_acc_{o}_init")
            constant(0.0, current_sum)
            
            for h in range(self.hidden_size):
                weight_cell = Cell(name=f"w_ho_{h}_{o}")
                constant(self.weights_hidden_output[h][o], weight_cell)
                
                product_cell = Cell(name=f"prod_ho_{h}_{o}")
                multiplier(hidden_cells[h], weight_cell, product_cell)
                
                new_sum = Cell(name=f"output_acc_{o}_{h}")
                adder(current_sum, product_cell, new_sum)
                current_sum = new_sum
            
            # Apply sigmoid activation
            output = Cell(name=f"output_{o}")
            sigmoid_propagator(current_sum, output)
            output_cells.append(output)
        
        # Run the propagator network
        run()
        
        return output_cells, hidden_cells, input_cells
    
    def forward(self, inputs):
        """
        Perform forward pass and return output values.
        
        Args:
            inputs: List of input values
            
        Returns:
            List of output values
        """
        output_cells, _, _ = self.forward_propagator(inputs)
        
        # Extract values from cells (handle TMS if present)
        outputs = []
        for cell in output_cells:
            content = cell.content
            if isinstance(content, Tms):
                result = tms_query(content)
                outputs.append(result.value if isinstance(result, Supported) else result)
            elif isinstance(content, Supported):
                outputs.append(content.value)
            else:
                outputs.append(content)
        
        return outputs
    
    def train(self, inputs_batch, targets_batch, epochs, learning_rate=1.0):
        """
        Train the network using backpropagation.
        
        Training data is tagged with 'training-data' premise for TMS tracking.
        
        Args:
            inputs_batch: List of input samples
            targets_batch: List of target outputs
            epochs: Number of training iterations
            learning_rate: Step size for weight updates
            
        Returns:
            loss_history: List of MSE loss values for each epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            total_error_squared = 0
            
            # Accumulate gradients for batch
            grad_hidden_output = [[0.0] * self.output_size for _ in range(self.hidden_size)]
            grad_input_hidden = [[0.0] * self.hidden_size for _ in range(self.input_size)]
            
            for sample_idx, (inputs, targets) in enumerate(zip(inputs_batch, targets_batch)):
                # Tag training data with premise
                premise_tag = f'training-sample-{sample_idx}'
                
                # Forward pass using propagator network
                output_cells, hidden_cells, input_cells = self.forward_propagator(
                    inputs, premise_tag=premise_tag
                )
                
                # Extract output values
                output = []
                for cell in output_cells:
                    content = cell.content
                    if isinstance(content, Tms):
                        result = tms_query(content)
                        output.append(result.value if isinstance(result, Supported) else result)
                    elif isinstance(content, Supported):
                        output.append(content.value)
                    else:
                        output.append(content)
                
                # Extract hidden values
                hidden_output = []
                for cell in hidden_cells:
                    content = cell.content
                    if isinstance(content, Tms):
                        result = tms_query(content)
                        hidden_output.append(result.value if isinstance(result, Supported) else result)
                    elif isinstance(content, Supported):
                        hidden_output.append(content.value)
                    else:
                        hidden_output.append(content)
                
                # Calculate error
                error = [t - o for t, o in zip(targets, output)]
                total_error_squared += sum(e * e for e in error)
                
                # Backpropagation - output layer
                d_output = [e * sigmoid_derivative(o) for e, o in zip(error, output)]
                
                # Backpropagation - hidden layer
                error_hidden = []
                for h in range(self.hidden_size):
                    err = sum(d_output[o] * self.weights_hidden_output[h][o] 
                              for o in range(self.output_size))
                    error_hidden.append(err)
                
                d_hidden = [eh * sigmoid_derivative(ho) 
                            for eh, ho in zip(error_hidden, hidden_output)]
                
                # Accumulate gradients
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        grad_hidden_output[h][o] += hidden_output[h] * d_output[o]
                
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        grad_input_hidden[i][h] += inputs[i] * d_hidden[h]
            
            # Update weights
            for h in range(self.hidden_size):
                for o in range(self.output_size):
                    self.weights_hidden_output[h][o] += learning_rate * grad_hidden_output[h][o]
            
            for i in range(self.input_size):
                for h in range(self.hidden_size):
                    self.weights_input_hidden[i][h] += learning_rate * grad_input_hidden[i][h]
            
            # Track loss
            loss = total_error_squared / len(inputs_batch)
            loss_history.append(loss)
        
        return loss_history
    
    def forward_with_tms(self, inputs, premise_tag='query'):
        """
        Perform forward pass with TMS tracking for dependency analysis.
        
        This demonstrates how propagators can track which training data
        influenced the output through the TMS system.
        
        Args:
            inputs: List of input values
            premise_tag: Tag for the input premise
            
        Returns:
            Tuple of (output_values, tms_content) for analysis
        """
        output_cells, _, _ = self.forward_propagator(inputs, premise_tag=premise_tag)
        
        outputs = []
        tms_info = []
        
        for cell in output_cells:
            content = cell.content
            if isinstance(content, Tms):
                result = tms_query(content)
                if isinstance(result, Supported):
                    outputs.append(result.value)
                    tms_info.append({
                        'value': result.value,
                        'support': list(result.support)
                    })
                else:
                    outputs.append(result)
                    tms_info.append({'value': result, 'support': []})
            elif isinstance(content, Supported):
                outputs.append(content.value)
                tms_info.append({
                    'value': content.value,
                    'support': list(content.support)
                })
            else:
                outputs.append(content)
                tms_info.append({'value': content, 'support': []})
        
        return outputs, tms_info


# =============================================================================
# DEMO / MAIN
# =============================================================================

def demo_numpy():
    """Demonstrate the NumPy-based neural network on XOR problem."""
    if not NUMPY_AVAILABLE:
        print("NumPy not available. Install with: pip install numpy matplotlib")
        return
    
    print("=" * 60)
    print("XOR Neural Network - NumPy Version")
    print("=" * 60)
    
    # Create neural network
    nn = NeuralNetworkNumPy(input_size=2, hidden_size=2, output_size=1)
    
    # XOR training data
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[0], [1], [1], [0]])
    
    # Train
    epochs = 10000
    loss_history = nn.train(training_inputs, training_targets, epochs)
    
    print(f"\nTraining complete! Final loss: {loss_history[-1]:.6f}")
    print("\nResults:")
    
    # Test on all inputs
    for i, test_input in enumerate(training_inputs):
        predicted = nn.forward(test_input.reshape(1, -1))
        expected = training_targets[i][0]
        print(f"  Input: {test_input} -> Predicted: {predicted[0][0]:.4f}, Expected: {expected}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss Over Epochs (NumPy Version)")
    plt.grid(True)
    plt.show()


def demo_python():
    """Demonstrate the pure Python neural network on XOR problem."""
    print("=" * 60)
    print("XOR Neural Network - Pure Python Version")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create neural network
    nn = NeuralNetworkPython(input_size=2, hidden_size=2, output_size=1)
    
    # XOR training data (using Python lists)
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    training_targets = [[0], [1], [1], [0]]
    
    # Train
    epochs = 10000
    loss_history = nn.train(training_inputs, training_targets, epochs)
    
    print(f"\nTraining complete! Final loss: {loss_history[-1]:.6f}")
    print("\nResults:")
    
    # Test on all inputs
    for inputs, expected in zip(training_inputs, training_targets):
        predicted = nn.forward(inputs)
        print(f"  Input: {inputs} -> Predicted: {predicted[0]:.4f}, Expected: {expected[0]}")
    
    # Print loss at key epochs
    print("\nLoss progression:")
    for i in [0, 100, 1000, 5000, epochs-1]:
        print(f"  Epoch {i+1}: {loss_history[i]:.6f}")
    
    # Simple ASCII plot of loss
    print("\nTraining loss (ASCII plot):")
    plot_ascii(loss_history, width=60, height=15)


def demo_propagator():
    """Demonstrate the propagator-based neural network on XOR problem."""
    print("=" * 60)
    print("XOR Neural Network - Propagator Version (with TMS)")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create neural network using propagators
    nn = NeuralNetworkPropagator(input_size=2, hidden_size=2, output_size=1, seed=42)
    
    # XOR training data - tagged with TMS premises for provenance tracking
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    training_targets = [[0], [1], [1], [0]]
    
    print("\nXOR Training Data (with TMS tagging):")
    print("  Each sample is tagged with a premise for provenance tracking:")
    for i, (inp, tgt) in enumerate(zip(training_inputs, training_targets)):
        print(f"    Sample {i}: input={inp}, target={tgt[0]}, premise='training-sample-{i}'")
    
    # Train (fewer epochs since propagator version is slower)
    epochs = 5000
    print(f"\nTraining for {epochs} epochs...")
    loss_history = nn.train(training_inputs, training_targets, epochs)
    
    print(f"Training complete! Final loss: {loss_history[-1]:.6f}")
    print("\nResults:")
    
    # Test on all inputs
    for inputs, expected in zip(training_inputs, training_targets):
        predicted = nn.forward(inputs)
        print(f"  Input: {inputs} -> Predicted: {predicted[0]:.4f}, Expected: {expected[0]}")
    
    # Demonstrate TMS tracking with detailed output
    print("\n" + "=" * 60)
    print("TMS Dependency Tracking Demo")
    print("=" * 60)
    print("\nThe propagator network uses Truth Maintenance System (TMS)")
    print("to track data provenance through the network.")
    print("\nForward pass with premise tagging:")
    
    test_input = [1, 0]
    outputs, tms_info = nn.forward_with_tms(test_input, premise_tag='xor-query')
    print(f"  Query: input={test_input}, premise='xor-query'")
    print(f"  Output: {outputs[0]:.4f} (expected: 1)")

    test_input = [1, 1]
    outputs, tms_info = nn.forward_with_tms(test_input, premise_tag='xor-query')
    print(f"  Query: input={test_input}, premise='xor-query'")
    print(f"  Output: {outputs[0]:.4f} (expected: 0)")

    # Explanation of why kick_out doesn't work with current architecture
    print("\n" + "=" * 60)
    print("Why kick_out() Has No Effect on This Network")
    print("=" * 60)
    print("""
LIMITATION: The current implementation stores weights as plain Python floats,
not as TMS-tracked values. When we train the network:

  1. Each training sample contributes to gradient updates
  2. Weights are updated: w += learning_rate * gradient
  3. The weights "forget" which samples contributed to them

For kick_out() to work, we would need:
  - Weights stored as Cells with TMS-tracked Supported values
  - Each weight update to merge dependencies from training samples
  - Weight values to track their full provenance history

This would require a fundamentally different architecture where the
entire training history is encoded in the TMS support of each weight.
""")
    
    # Demonstrate what DOES work with TMS - single-pass provenance tracking
    print("=" * 60)
    print("What DOES Work: Single-Pass Provenance Tracking")
    print("=" * 60)
    demo_tms_single_pass()
    
    # Print loss at key epochs
    print("\nLoss progression:")
    for i in [0, 100, 1000, epochs-1]:
        print(f"  Epoch {i+1}: {loss_history[i]:.6f}")


def demo_tms_single_pass():
    """
    Demonstrate TMS tracking in a single propagator computation.
    
    This shows how kick_out DOES work when all values are TMS-tracked
    within a single propagator network.
    """
    from propagator.tms import bring_in, kick_out, tms_query, make_tms
    from propagator.supported_values import supported
    
    initialize_scheduler()
    
    # Create cells for a simple weighted sum: output = w1*x1 + w2*x2
    x1 = Cell(name="x1")
    x2 = Cell(name="x2")
    w1 = Cell(name="w1")
    w2 = Cell(name="w2")
    prod1 = Cell(name="prod1")
    prod2 = Cell(name="prod2")
    output = Cell(name="output")
    
    # Set up the propagator network
    multiplier(x1, w1, prod1)  # prod1 = x1 * w1
    multiplier(x2, w2, prod2)  # prod2 = x2 * w2
    adder(prod1, prod2, output)  # output = prod1 + prod2
    
    # Add values WITH TMS tracking - each from a different "source"
    x1.add_content(make_tms(supported(1.0, ['input-x1'])))
    x2.add_content(make_tms(supported(2.0, ['input-x2'])))
    w1.add_content(make_tms(supported(0.5, ['weight-source-A'])))
    w2.add_content(make_tms(supported(0.3, ['weight-source-B'])))
    
    run()
    
    print("\nSimple weighted sum: output = w1*x1 + w2*x2")
    print(f"  x1=1.0 (from 'input-x1'), x2=2.0 (from 'input-x2')")
    print(f"  w1=0.5 (from 'weight-source-A'), w2=0.3 (from 'weight-source-B')")
    
    result = tms_query(output.content)
    print(f"\n  Output: {result.value} = 0.5*1.0 + 0.3*2.0 = 1.1")
    print(f"  Dependencies: {[w.obj for w in result.support]}")
    
    # Now kick out one of the weight sources
    print("\n  Kicking out 'weight-source-B'...")
    kick_out('weight-source-B')
    
    result_after = tms_query(output.content)
    print(f"  Output after kick_out: {result_after}")
    print("  (The w2 contribution is no longer believed, so output may be partial or None)")
    
    # Bring it back
    bring_in('weight-source-B')
    result_restored = tms_query(output.content)
    print(f"\n  After bring_in('weight-source-B'): {result_restored.value}")


def plot_ascii(data, width=60, height=15):
    """Create a simple ASCII plot of the data."""
    if not data:
        return
    
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
    
    # Sample data to fit width
    step = max(1, len(data) // width)
    sampled = [data[i] for i in range(0, len(data), step)][:width]
    
    # Create plot
    for row in range(height):
        threshold = max_val - (row / (height - 1)) * range_val
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "*"
            else:
                line += " "
        
        # Add y-axis labels
        if row == 0:
            print(f"  {max_val:.3f} |{line}|")
        elif row == height - 1:
            print(f"  {min_val:.3f} |{line}|")
        else:
            print(f"        |{line}|")
    
    # X-axis
    print(f"        +{'-' * width}+")
    print(f"        0{' ' * (width//2 - 2)}Epochs{' ' * (width//2 - 4)}{len(data)}")


# =============================================================================
# TRUE TMS-NATIVE NEURAL NETWORK - Propagator-Based Forward Pass
# =============================================================================
"""
This implementation uses the propagator model for FORWARD COMPUTATION:
- Each forward pass creates a fresh propagator network
- Weights carry TMS-supported values with training sample premises
- All arithmetic (multiply, add, sigmoid) happens through propagators
- kick_out() affects output because premises flow through computation

Why fresh networks for each forward pass?
- Propagator cells are MONOTONIC (values accumulate, never replace)
- Neural network training REPLACES weight values
- If we use persistent cells, adding different weight values causes contradictions
- Fresh networks avoid this: each forward pass uses current weight values

The TMS worldview (which premises are believed) persists across forward passes,
so kick_out/bring_in correctly affects outputs.
"""


class NeuralNetworkPropagatorTMS:
    """
    A neural network using propagators for all forward computation.
    
    Weights are stored externally as (value, premise_set) tuples.
    Each forward pass builds a fresh propagator network that uses
    these weights as TMS-supported values. kick_out/bring_in work
    because the TMS worldview persists across forward passes.
    """
    
    def __init__(self, input_size, hidden_size, output_size, seed=None):
        if seed is not None:
            random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize scheduler ONCE at network creation
        # This clears TMS worldview for a fresh start
        initialize_scheduler()
        
        # Premise management - same objects throughout
        self.training_premises = {}
        self.init_premise = 'weight-initialization'
        
        # Weights stored as (value, premise_set) tuples
        # This allows us to track provenance without cell contradictions
        self.weights_ih = []
        self.weights_ho = []
        
        for i in range(input_size):
            row = []
            for h in range(hidden_size):
                initial_val = random.random()
                row.append((initial_val, {self.init_premise}))
            self.weights_ih.append(row)
        
        for h in range(hidden_size):
            row = []
            for o in range(output_size):
                initial_val = random.random()
                row.append((initial_val, {self.init_premise}))
            self.weights_ho.append(row)
    
    def _get_training_premise(self, sample_idx):
        """Get or create the premise object for a training sample."""
        if sample_idx not in self.training_premises:
            self.training_premises[sample_idx] = f'training-sample-{sample_idx}'
        return self.training_premises[sample_idx]
    
    def forward(self, inputs, premise_tag=None):
        """
        Forward pass building a FRESH propagator network each time.
        
        Creates cells and propagators, runs propagation, extracts outputs.
        
        IMPORTANT: We do NOT call initialize_scheduler() here because that
        would reset the TMS worldview (clearing kicked-out premises). Instead,
        we just create fresh cells and propagators. The TMS worldview persists.
        
        Args:
            inputs: List of input values
            premise_tag: Optional premise to tag input data
            
        Returns:
            outputs: List of output values (None if dependencies kicked out)
            hidden_outputs: List of hidden layer activations
            all_supports: Support dependencies for each output
        """
        # DO NOT call initialize_scheduler() - it would reset TMS worldview!
        # Just create fresh cells and propagators.
        
        # Input cells
        input_cells = []
        for i, val in enumerate(inputs):
            cell = Cell(name=f"input_{i}")
            if premise_tag:
                cell.add_content(make_tms(supported(val, [premise_tag])))
            else:
                cell.add_content(make_tms(supported(val, [])))
            input_cells.append(cell)
        
        # Hidden layer computation
        hidden_output_cells = []
        for h in range(self.hidden_size):
            # Weighted sum: sum_i(input_i * weight_ih[i][h])
            prev_sum = Cell(name=f"h_sum_init_{h}")
            prev_sum.add_content(make_tms(supported(0.0, [])))
            
            for i in range(self.input_size):
                # Weight cell with TMS support from training
                weight_val, weight_support = self.weights_ih[i][h]
                weight_cell = Cell(name=f"W_ih_{i}_{h}")
                weight_cell.add_content(make_tms(supported(weight_val, list(weight_support))))
                
                prod_cell = Cell(name=f"h_prod_{h}_{i}")
                multiplier(input_cells[i], weight_cell, prod_cell)
                
                new_sum = Cell(name=f"h_sum_{h}_{i}")
                adder(prev_sum, prod_cell, new_sum)
                prev_sum = new_sum
            
            h_out = Cell(name=f"hidden_{h}")
            sigmoid_propagator(prev_sum, h_out)
            hidden_output_cells.append(h_out)
        
        # Output layer computation
        output_cells = []
        for o in range(self.output_size):
            prev_sum = Cell(name=f"o_sum_init_{o}")
            prev_sum.add_content(make_tms(supported(0.0, [])))
            
            for h in range(self.hidden_size):
                # Weight cell with TMS support from training
                weight_val, weight_support = self.weights_ho[h][o]
                weight_cell = Cell(name=f"W_ho_{h}_{o}")
                weight_cell.add_content(make_tms(supported(weight_val, list(weight_support))))
                
                prod_cell = Cell(name=f"o_prod_{o}_{h}")
                multiplier(hidden_output_cells[h], weight_cell, prod_cell)
                
                new_sum = Cell(name=f"o_sum_{o}_{h}")
                adder(prev_sum, prod_cell, new_sum)
                prev_sum = new_sum
            
            out = Cell(name=f"output_{o}")
            sigmoid_propagator(prev_sum, out)
            output_cells.append(out)
        
        # Run the propagator network
        run()
        
        # Extract outputs using tms_query (respects current worldview)
        outputs = []
        hidden_outputs = []
        all_supports = []
        
        for cell in hidden_output_cells:
            content = cell.content
            if isinstance(content, Tms):
                result = tms_query(content)
                if result is None:
                    hidden_outputs.append(None)
                elif isinstance(result, Supported):
                    hidden_outputs.append(result.value)
                else:
                    hidden_outputs.append(result)
            elif isinstance(content, Supported):
                hidden_outputs.append(content.value)
            else:
                hidden_outputs.append(content)
        
        for cell in output_cells:
            content = cell.content
            if isinstance(content, Tms):
                result = tms_query(content)
                if result is None:
                    outputs.append(None)
                    all_supports.append([])
                elif isinstance(result, Supported):
                    outputs.append(result.value)
                    all_supports.append([w.obj for w in result.support])
                else:
                    outputs.append(result)
                    all_supports.append([])
            elif isinstance(content, Supported):
                outputs.append(content.value)
                all_supports.append([w.obj for w in content.support])
            else:
                outputs.append(content)
                all_supports.append([])
        
        return outputs, hidden_outputs, all_supports
    
    def train_step(self, inputs, targets, learning_rate, sample_idx):
        """
        Perform one training step using propagator-based forward pass.
        
        1. Forward pass through fresh propagator network
        2. Compute gradients from outputs
        3. Update weight tuples with new values and combined premise sets
        
        Args:
            inputs: Input values for this sample
            targets: Target values for this sample
            learning_rate: Step size
            sample_idx: Index of this training sample
            
        Returns:
            error_squared: The squared error for this sample
        """
        # Get premise for this training sample
        sample_premise = self._get_training_premise(sample_idx)
        
        # Forward pass through propagator network
        outputs, hidden_outputs, _ = self.forward(inputs, premise_tag=sample_premise)
        
        # Handle None outputs (shouldn't happen during training)
        output_vals = [o if o is not None else 0.5 for o in outputs]
        hidden_vals = [h if h is not None else 0.5 for h in hidden_outputs]
        
        # Get current weight values
        weights_ih_vals = [[self.weights_ih[i][h][0] for h in range(self.hidden_size)] 
                          for i in range(self.input_size)]
        weights_ho_vals = [[self.weights_ho[h][o][0] for o in range(self.output_size)] 
                          for h in range(self.hidden_size)]
        
        # Calculate error
        error = [t - o for t, o in zip(targets, output_vals)]
        error_squared = sum(e * e for e in error)
        
        # Backpropagation - output layer
        d_output = [e * sigmoid_derivative(o) for e, o in zip(error, output_vals)]
        
        # Backpropagation - hidden layer
        error_hidden = []
        for h in range(self.hidden_size):
            err = sum(d_output[o] * weights_ho_vals[h][o] for o in range(self.output_size))
            error_hidden.append(err)
        
        d_hidden = [eh * sigmoid_derivative(ho) 
                    for eh, ho in zip(error_hidden, hidden_vals)]
        
        # Update weight tuples with new values and combined premise sets
        # Hidden-to-output weights
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                current_val, current_support = self.weights_ho[h][o]
                delta = learning_rate * hidden_vals[h] * d_output[o]
                new_val = current_val + delta
                # Accumulate premises - this weight now depends on this sample
                new_support = current_support | {sample_premise}
                self.weights_ho[h][o] = (new_val, new_support)
        
        # Input-to-hidden weights
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                current_val, current_support = self.weights_ih[i][h]
                delta = learning_rate * inputs[i] * d_hidden[h]
                new_val = current_val + delta
                # Accumulate premises
                new_support = current_support | {sample_premise}
                self.weights_ih[i][h] = (new_val, new_support)
        
        return error_squared
    
    def train(self, inputs_batch, targets_batch, epochs, learning_rate=1.0):
        """
        Train the network over multiple epochs.
        
        Each weight update accumulates the training sample premise.
        The TMS tracks which samples influenced each weight.
        
        Returns:
            loss_history: List of MSE loss values per epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            total_error_squared = 0
            
            for sample_idx, (inputs, targets) in enumerate(zip(inputs_batch, targets_batch)):
                error_sq = self.train_step(inputs, targets, learning_rate, sample_idx)
                total_error_squared += error_sq
            
            loss = total_error_squared / len(inputs_batch)
            loss_history.append(loss)
        
        return loss_history
    
    def get_weight_provenance(self):
        """
        Get the provenance (dependencies) for all weights.
        
        Returns a dict mapping weight names to their support premises.
        """
        provenance = {}
        
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                name = f"W_ih_{i}_{h}"
                _, support = self.weights_ih[i][h]
                provenance[name] = list(support)
        
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                name = f"W_ho_{h}_{o}"
                _, support = self.weights_ho[h][o]
                provenance[name] = list(support)
        
        return provenance


def demo_propagator_tms():
    """
    Demonstrate the TMS-native neural network where kick_out() works!
    
    This version:
    - Uses fresh propagator networks for each forward pass (no initialize_scheduler in forward)
    - Stores weights as (value, premise_set) tuples externally
    - Each training sample's premise is accumulated in weight supports
    - kick_out() works because TMS worldview persists across forward passes
    """
    print("=" * 70)
    print("XOR Neural Network - TMS-Native (Propagator-based forward pass)")
    print("=" * 70)
    
    # Create network
    nn = NeuralNetworkPropagatorTMS(input_size=2, hidden_size=2, output_size=1, seed=42)
    
    # XOR training data
    training_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    training_targets = [[0], [1], [1], [0]]
    
    print("\nXOR Training Data:")
    for i, (inp, tgt) in enumerate(zip(training_inputs, training_targets)):
        print(f"  Sample {i}: input={inp}, target={tgt[0]}")
    
    # Train for enough epochs to learn XOR
    # Note: Propagator-based training is slower than pure Python
    epochs = 1000
    print(f"\nTraining for {epochs} epochs...")
    print("(Each weight accumulates training sample premises)")
    loss_history = nn.train(training_inputs, training_targets, epochs, learning_rate=1.0)
    
    print(f"Training complete! Final loss: {loss_history[-1]:.6f}")
    
    # Test on all inputs
    print("\nResults after training:")
    for inputs, expected in zip(training_inputs, training_targets):
        outputs, _, supports = nn.forward(inputs, premise_tag='query')
        out_val = outputs[0]
        if isinstance(out_val, float):
            print(f"  Input: {inputs} -> Output: {out_val:.4f}, Expected: {expected[0]}")
        else:
            print(f"  Input: {inputs} -> Output: {out_val}, Expected: {expected[0]}")
    
    # Show weight provenance
    print("\n" + "-" * 70)
    print("Weight Provenance (which samples influenced each weight):")
    print("-" * 70)
    provenance = nn.get_weight_provenance()
    for name, support in provenance.items():
        print(f"  {name}: {support}")
    
    # THE DEMO: kick_out a training sample and see the effect!
    print("\n" + "=" * 70)
    print("DEMONSTRATION: TMS Provenance Tracking")
    print("=" * 70)
    
    # Test input [1, 0] which should output ~1
    test_input = [1, 0]
    outputs_before, _, supports_before = nn.forward(test_input, premise_tag='test-query')
    
    print(f"\nBefore kick_out:")
    print(f"  Input: {test_input}")
    out_val = outputs_before[0]
    if out_val is not None:
        print(f"  Output: {out_val:.4f} (expected: 1)")
    else:
        print(f"  Output: None (expected: 1)")
    print(f"  Output depends on: {supports_before[0]}")
    
    # Explanation of provenance model
    print("\n" + "-" * 70)
    print("PROVENANCE MODEL:")
    print("-" * 70)
    print("""
Each weight's support accumulates ALL training samples that
contributed to its value. After training, ALL sample premises
are in every weight's support (training-sample-0 through -3).

kick_out('training-sample-3') should make output None
because that premise is part of every weight's support!
""")
    
    # Get the ACTUAL premise object that the network uses
    premise_3 = nn._get_training_premise(3)
    
    # Show that kick_out actually works
    print("=" * 70)
    print("DEMONSTRATION: kick_out() WORKS when premise is actually used")
    print("=" * 70)
    
    print(f"\nKicking out '{premise_3}' (a premise all weights depend on)...")
    kick_out(premise_3)
    
    outputs_no_sample_3, _, supports_no = nn.forward(test_input, premise_tag='test-query')
    out_val = outputs_no_sample_3[0]
    
    print(f"\nAfter kicking out training-sample-3:")
    if out_val is None:
        print(f"  Output: None (NO VALID VALUE!)")
        print(f"  ✓ kick_out() WORKS! The weights depend on sample 3.")
    else:
        print(f"  Output: {out_val}")
        print(f"  Supports: {supports_no[0]}")
    
    # Bring it back
    print(f"\nBringing back '{premise_3}'...")
    bring_in(premise_3)
    
    outputs_restored, _, _ = nn.forward(test_input, premise_tag='test-query')
    print(f"  Output after bring_in: {outputs_restored[0]:.4f}")
    
    if outputs_restored[0] is not None and abs(outputs_before[0] - outputs_restored[0]) < 0.01:
        print("  ✓ bring_in() restored the output!")
    
    # Show alternative: train with only 3 samples and demonstrate difference
    print("\n" + "=" * 70)
    print("COMPARISON: Training WITHOUT sample 2 (the [1,0]->1 sample)")
    print("=" * 70)
    
    # Create new network, train without sample 2
    nn2 = NeuralNetworkPropagatorTMS(input_size=2, hidden_size=2, output_size=1, seed=42)
    inputs_without_2 = [[0, 0], [0, 1], [1, 1]]  # Skip [1, 0]
    targets_without_2 = [[0], [1], [0]]
    
    print("\nTraining WITHOUT sample [1,0]->1...")
    nn2.train(inputs_without_2, targets_without_2, epochs, learning_rate=1.0)
    
    outputs_without_2, _, _ = nn2.forward([1, 0], premise_tag='query')
    print(f"  Output for [1,0] when trained WITHOUT that sample: {outputs_without_2[0]:.4f}")
    print(f"  Output for [1,0] when trained WITH all samples:    {outputs_before[0]:.4f}")
    print(f"  Difference: {abs(outputs_without_2[0] - outputs_before[0]):.4f}")
    
    if abs(outputs_without_2[0] - outputs_before[0]) > 0.1:
        print("  ✓ Significant difference shows sample 2 WAS important!")
    
    # Loss progression
    print("\nLoss progression (full training):")
    for i in [0, 100, 500, 1000, epochs-1]:
        if i < len(loss_history):
            print(f"  Epoch {i+1}: {loss_history[i]:.6f}")


if __name__ == "__main__":
    # Run the TRUE TMS-native propagator demo (kick_out works!)
    demo_propagator_tms()
    
    print("\n" + "=" * 70)
    print("\n")
    
    # Run pure Python demo for comparison
    demo_python()
    
    print("\n")
    
    # Run original propagator demo (shows limitation)
    # demo_propagator()
    
    # Run NumPy demo if available
    if NUMPY_AVAILABLE:
        print("\n")
        demo_numpy()
    else:
        print("\nNote: Install NumPy and matplotlib for the NumPy version demo:")
        print("  pip install numpy matplotlib")