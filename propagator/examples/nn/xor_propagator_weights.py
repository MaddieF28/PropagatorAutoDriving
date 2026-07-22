"""
Propagator Neural Network - Deep Integration

This implements neural networks WHERE THE WEIGHTS ARE PROPAGATOR CELLS.

Key Insight:
-----------
Instead of "propagators trigger retraining of an external network",
we make the network ITSELF a propagator network:

1. Each WEIGHT is a Cell
2. Each training SAMPLE contributes GRADIENT INFORMATION to weight cells
3. MERGE combines gradient contributions from believed samples
4. Forward pass is PROPAGATORS connecting activation cells
5. kick_out/bring_in changes which gradient contributions are believed

The Lattice of Gradient Contributions:
-------------------------------------
For a weight w, we track:
  - Initial value w₀
  - Gradient contribution from sample 0: Δw₀
  - Gradient contribution from sample 1: Δw₁
  - etc.

The "current" weight is: w₀ + Σ(believed Δwᵢ)

IMPORTANT CAVEAT:
----------------
In true SGD, gradients aren't additive - training on A then B gives different
results than training on B then A, and different from training on A+B together.

This implementation uses a LINEARIZED APPROXIMATION:
- Compute each sample's gradient contribution independently
- Combine by summing (assumes small learning rate / linear regime)

This is similar to "influence functions" in interpretable ML - a first-order
approximation of how each sample affects the model.

For exact behavior, use xor_retrain_on_query.py which actually retrains.
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Optional, FrozenSet, Tuple
from dataclasses import dataclass

from propagator import Cell, run, initialize_scheduler
from propagator.nothing import nothing_p
from propagator.supported_values import supported, Supported
from propagator.tms import make_tms, tms_query, kick_out, bring_in, premise_in
from propagator.merge import assign_merge_operation


# =============================================================================
# Gradient Contribution - The atomic unit of training information
# =============================================================================

@dataclass(frozen=True)
class GradientContribution:
    """
    A gradient update contributed by a specific training sample.
    
    This represents: "Sample X says this weight should change by delta"
    
    Multiple contributions merge by:
    1. Union of sources (idempotent - same source doesn't double-count)
    2. Sum of deltas from distinct sources
    """
    delta: float              # The gradient update
    source: FrozenSet[str]    # Which samples contributed this
    
    @staticmethod
    def from_sample(delta: float, sample_id: str) -> 'GradientContribution':
        """Create a contribution from a single sample."""
        return GradientContribution(delta, frozenset([sample_id]))
    
    @property
    def total_delta(self) -> float:
        """The total delta (for single-source, same as delta)."""
        return self.delta
    
    def __repr__(self):
        sources = sorted(self.source)
        if len(sources) == 1:
            return f"Δ({self.delta:.6f}, {sources[0]})"
        return f"Δ({self.delta:.6f}, {len(sources)} sources)"


def is_gradient_contribution(x) -> bool:
    return isinstance(x, GradientContribution)


def merge_gradient_contributions(g1: GradientContribution, 
                                  g2: GradientContribution) -> GradientContribution:
    """
    Merge gradient contributions by summing deltas from DISTINCT sources.
    
    IDEMPOTENT: merge(g, g) = g
    - Same sources → same delta (no double-counting)
    - New sources → add their delta
    """
    combined_sources = g1.source | g2.source
    
    if g1.source == g2.source:
        # Same sources: idempotent
        return g1
    elif g1.source.issuperset(g2.source):
        # g1 already includes g2's contribution
        return g1
    elif g2.source.issuperset(g1.source):
        # g2 already includes g1's contribution
        return g2
    else:
        # Distinct or partially overlapping sources
        # For simplicity, assume each source contributes delta/|sources|
        # and sum only the NEW contributions
        
        only_in_g1 = g1.source - g2.source
        only_in_g2 = g2.source - g1.source
        in_both = g1.source & g2.source
        
        # Per-source contribution
        per_source_g1 = g1.delta / len(g1.source) if g1.source else 0
        per_source_g2 = g2.delta / len(g2.source) if g2.source else 0
        
        combined_delta = (
            per_source_g1 * len(only_in_g1) +
            per_source_g2 * len(only_in_g2) +
            (per_source_g1 + per_source_g2) / 2 * len(in_both)  # Average for overlap
        )
        
        # Actually simpler: just sum if no overlap
        if not in_both:
            combined_delta = g1.delta + g2.delta
        
        return GradientContribution(combined_delta, combined_sources)


# Register merge
assign_merge_operation(merge_gradient_contributions,
                       is_gradient_contribution, is_gradient_contribution)


# =============================================================================
# Weight Cell - A weight that accumulates gradient contributions
# =============================================================================

class WeightCell:
    """
    A neural network weight implemented as a propagator cell.
    
    Stores:
    - initial_value: The starting weight (before any training)
    - cell: A propagator Cell containing gradient contributions
    
    The effective weight is: initial_value + sum of believed gradient contributions
    """
    
    def __init__(self, initial_value: float, name: str = None):
        self.initial_value = initial_value
        self.cell = Cell(name=name)
        self._name = name
    
    def add_gradient(self, delta: float, sample_id: str):
        """Add a gradient contribution from a training sample."""
        contribution = GradientContribution.from_sample(delta, sample_id)
        self.cell.add_content(
            make_tms(supported(contribution, [sample_id]))
        )
    
    @property
    def effective_value(self) -> float:
        """Get the current weight value based on believed contributions."""
        content = self.cell.content
        if nothing_p(content):
            return self.initial_value

        result = tms_query(content)
        if nothing_p(result):
            return self.initial_value
        
        if isinstance(result, Supported):
            contribution = result.value
        else:
            contribution = result
        
        if isinstance(contribution, GradientContribution):
            return self.initial_value + contribution.total_delta
        
        return self.initial_value
    
    def __repr__(self):
        return f"WeightCell({self._name}, val={self.effective_value:.4f})"


# =============================================================================
# Propagator Neural Network Layer
# =============================================================================

def sigmoid(x: float) -> float:
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output: float) -> float:
    """Derivative of sigmoid given its output."""
    return output * (1 - output)


class PropagatorNeuralNetwork:
    """
    A neural network where weights are propagator cells.
    
    Architecture:
    - Input layer: plain values (not cells - these are fixed per forward pass)
    - Hidden layer: WeightCells for input→hidden connections
    - Output layer: WeightCells for hidden→output connections
    
    Training:
    - For each sample, compute gradients via backprop
    - Add gradient contributions to weight cells with sample as premise
    
    Querying:
    - kick_out(sample) removes that sample's gradient contributions
    - bring_in(sample) restores them
    - Forward pass uses effective_value of each weight
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 seed: int = None):
        if seed is not None:
            random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weight cells with random values
        self.weights_ih: List[List[WeightCell]] = [
            [WeightCell(random.random() * 2 - 1, f"w_ih[{i}][{h}]")
             for h in range(hidden_size)]
            for i in range(input_size)
        ]
        
        self.weights_ho: List[List[WeightCell]] = [
            [WeightCell(random.random() * 2 - 1, f"w_ho[{h}][{o}]")
             for o in range(output_size)]
            for h in range(hidden_size)
        ]
        
        self.sample_premises: List[str] = []
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass using current effective weights."""
        # Hidden layer
        hidden_in = []
        for h in range(self.hidden_size):
            total = sum(inputs[i] * self.weights_ih[i][h].effective_value
                       for i in range(self.input_size))
            hidden_in.append(total)
        
        hidden_out = [sigmoid(x) for x in hidden_in]
        
        # Output layer
        output_in = []
        for o in range(self.output_size):
            total = sum(hidden_out[h] * self.weights_ho[h][o].effective_value
                       for h in range(self.hidden_size))
            output_in.append(total)
        
        output_out = [sigmoid(x) for x in output_in]
        
        return output_out
    
    def train_sample(self, inputs: List[float], targets: List[float],
                     sample_id: str, learning_rate: float = 1.0,
                     epochs: int = 1000):
        """
        Train on a single sample and record gradient contributions.
        
        This computes what gradient updates this sample would contribute
        if trained in isolation, then records them in the weight cells.
        """
        self.sample_premises.append(sample_id)
        
        # We need to train temporarily to find the gradient direction
        # Store original weights
        original_ih = [[self.weights_ih[i][h].initial_value 
                       for h in range(self.hidden_size)]
                      for i in range(self.input_size)]
        original_ho = [[self.weights_ho[h][o].initial_value
                       for o in range(self.output_size)]
                      for h in range(self.hidden_size)]
        
        # Train copies
        w_ih = [row[:] for row in original_ih]
        w_ho = [row[:] for row in original_ho]
        
        for _ in range(epochs):
            # Forward
            h_in = [sum(inputs[i] * w_ih[i][h] for i in range(self.input_size))
                    for h in range(self.hidden_size)]
            h_out = [sigmoid(x) for x in h_in]
            
            o_in = [sum(h_out[h] * w_ho[h][o] for h in range(self.hidden_size))
                    for o in range(self.output_size)]
            o_out = [sigmoid(x) for x in o_in]
            
            # Backward
            o_err = [targets[o] - o_out[o] for o in range(self.output_size)]
            o_delta = [o_err[o] * sigmoid_derivative(o_out[o])
                      for o in range(self.output_size)]
            
            h_err = [sum(o_delta[o] * w_ho[h][o] for o in range(self.output_size))
                    for h in range(self.hidden_size)]
            h_delta = [h_err[h] * sigmoid_derivative(h_out[h])
                      for h in range(self.hidden_size)]
            
            # Update temporary weights
            for h in range(self.hidden_size):
                for o in range(self.output_size):
                    w_ho[h][o] += learning_rate * h_out[h] * o_delta[o]
            for i in range(self.input_size):
                for h in range(self.hidden_size):
                    w_ih[i][h] += learning_rate * inputs[i] * h_delta[h]
        
        # Compute total delta (final - initial) and add to cells
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                delta = w_ih[i][h] - original_ih[i][h]
                self.weights_ih[i][h].add_gradient(delta, sample_id)
        
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                delta = w_ho[h][o] - original_ho[h][o]
                self.weights_ho[h][o].add_gradient(delta, sample_id)
    
    def train_all(self, training_data: List[Tuple[List[float], List[float]]],
                  learning_rate: float = 1.0, epochs: int = 1000,
                  verbose: bool = True):
        """Train on all samples, recording each sample's contribution."""
        for idx, (inputs, targets) in enumerate(training_data):
            sample_id = f"sample_{idx}"
            self.train_sample(inputs, targets, sample_id, learning_rate, epochs)
            
            if verbose:
                pred = self.forward(inputs)
                print(f"  {sample_id}: {inputs} → {targets}, "
                      f"predicts {pred[0]:.4f}")
    
    def get_weight_summary(self) -> Dict[str, float]:
        """Get current effective weights."""
        summary = {}
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                w = self.weights_ih[i][h]
                summary[f"ih[{i}][{h}]"] = w.effective_value
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                w = self.weights_ho[h][o]
                summary[f"ho[{h}][{o}]"] = w.effective_value
        return summary


# =============================================================================
# Demo
# =============================================================================

def demo():
    print("=" * 70)
    print("Propagator Neural Network - Deep Integration")
    print("=" * 70)
    print("""
In this implementation, the WEIGHTS THEMSELVES are propagator cells.

Each weight cell stores:
- An initial random value
- Gradient contributions from each training sample (as supported values)

The effective weight = initial + sum of believed gradient contributions

This means:
- kick_out(sample) removes that sample's gradient contribution from ALL weights
- bring_in(sample) restores it
- The network's behavior changes based on the TMS worldview

CAVEAT: This uses a LINEAR APPROXIMATION (gradients are summed).
For exact behavior, actual retraining is needed (see xor_retrain_on_query.py).
""")
    
    initialize_scheduler()
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    print("\nCreating network with weight cells...")
    nn = PropagatorNeuralNetwork(
        input_size=2, hidden_size=4, output_size=1, seed=42
    )
    
    print("\nTraining each sample (recording gradient contributions):")
    print("-" * 60)
    nn.train_all(training_data, learning_rate=1.0, epochs=5000)
    print("-" * 60)
    
    run()  # Run propagator network to merge contributions
    
    print("\n" + "=" * 70)
    print("Testing with different believed samples")
    print("=" * 70)
    
    test_cases = [
        ("All samples believed", [0, 1, 2, 3]),
        ("Only output=1 samples (1,2)", [1, 2]),
        ("Only output=0 samples (0,3)", [0, 3]),
        ("Without sample 0", [1, 2, 3]),
        ("Only sample 2 ([1,0]→1)", [2]),
    ]
    
    test_input = [1, 0]
    print(f"\nTest input: {test_input} (expected: 1)")
    print("-" * 60)
    
    for description, believed_indices in test_cases:
        # Set worldview
        for p in nn.sample_premises:
            kick_out(p)
        run()
        
        for idx in believed_indices:
            bring_in(f"sample_{idx}")
        run()
        
        # Forward pass with current effective weights
        pred = nn.forward(test_input)
        
        print(f"\n{description}:")
        print(f"  Believed samples: {believed_indices}")
        print(f"  Prediction: {pred[0]:.4f}")
        
        # Show a couple weight values
        w_ih_00 = nn.weights_ih[0][0].effective_value
        w_ho_00 = nn.weights_ho[0][0].effective_value
        print(f"  Sample weights: w_ih[0][0]={w_ih_00:.4f}, w_ho[0][0]={w_ho_00:.4f}")
    
    # Restore all
    for p in nn.sample_premises:
        bring_in(p)
    run()
    
    print("\n" + "=" * 70)
    print("Full XOR truth table (all samples believed)")
    print("=" * 70)
    
    print(f"\n{'Input':<10} | {'Prediction':<12} | {'Expected':<10}")
    print("-" * 40)
    for inputs, targets in training_data:
        pred = nn.forward(inputs)
        expected = targets[0]
        print(f"{inputs!s:<10} | {pred[0]:.4f}       | {expected}")
    
    print("\n" + "=" * 70)
    print("Examining weight cells")
    print("=" * 70)
    
    # Show one weight cell in detail
    w = nn.weights_ih[0][0]
    print(f"\nWeight cell: {w._name}")
    print(f"  Initial value: {w.initial_value:.4f}")
    print(f"  Current effective value: {w.effective_value:.4f}")
    print(f"  Cell content:")
    
    content = w.cell.content
    if content and hasattr(content, 'values'):
        for v in content.values:
            print(f"    {v}")


def demo_comparison():
    """Compare linear approximation with exact retraining."""
    print("\n" + "=" * 70)
    print("Comparison: Linear Approximation vs Exact Retraining")
    print("=" * 70)
    print("""
The linear approximation assumes gradient contributions are additive.
This works well when:
- Learning rate is small
- Samples don't conflict too much
- We're near the linear regime

Let's compare predictions from:
1. This implementation (sum of gradient contributions)
2. Actual retraining (xor_retrain_on_query.py approach)
""")
    
    # This would require importing the other module
    # For now, just show the linear approximation results
    
    initialize_scheduler()
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    nn = PropagatorNeuralNetwork(
        input_size=2, hidden_size=4, output_size=1, seed=42
    )
    nn.train_all(training_data, learning_rate=0.5, epochs=2000, verbose=False)
    run()
    
    configs = [
        ([0, 1, 2, 3], "All 4"),
        ([1, 2], "1,2 only"),
        ([0, 3], "0,3 only"),
    ]
    
    print(f"\n{'Config':<12} | {'[0,0]':<8} | {'[0,1]':<8} | {'[1,0]':<8} | {'[1,1]':<8}")
    print("-" * 60)
    
    for believed, name in configs:
        for p in nn.sample_premises:
            kick_out(p)
        for idx in believed:
            bring_in(f"sample_{idx}")
        run()
        
        preds = [nn.forward(inputs)[0] for inputs, _ in training_data]
        print(f"{name:<12} | {preds[0]:.4f}   | {preds[1]:.4f}   | {preds[2]:.4f}   | {preds[3]:.4f}")
    
    print("-" * 60)
    print("Expected:    | 0        | 1        | 1        | 0")


if __name__ == "__main__":
    demo()
    demo_comparison()
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
This implementation puts propagators at the WEIGHT level:

┌─────────────────────────────────────────────────────────────────┐
│                    PROPAGATOR LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  WeightCell (w_ih[0][0])                                 │   │
│  │    initial_value: 0.5                                    │   │
│  │    cell.content: TMS containing                          │   │
│  │      - Δ(+0.3, sample_0) supported by [sample_0]         │   │
│  │      - Δ(-0.1, sample_1) supported by [sample_1]         │   │
│  │      - Δ(+0.2, sample_2) supported by [sample_2]         │   │
│  │      - Δ(-0.2, sample_3) supported by [sample_3]         │   │
│  │    effective_value = 0.5 + sum(believed deltas)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Forward pass: uses effective_value of each WeightCell          │
│  kick_out/bring_in: changes which deltas are believed           │
│  Merge: combines gradient contributions (idempotent)            │
└─────────────────────────────────────────────────────────────────┘

This is "propagators AS the neural network" rather than 
"propagators watching the neural network".

The tradeoff:
- Pro: True integration of TMS into network structure
- Pro: Each weight naturally tracks sample contributions
- Con: Linear approximation (gradients aren't truly additive)
- Con: For exact results, need actual retraining

For exact results, see xor_retrain_on_query.py which retrains on demand.
""")
