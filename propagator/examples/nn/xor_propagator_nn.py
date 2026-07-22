"""
Propagator Neural Network - Weights ARE Cells, Merge IS Training

This is a TRUE propagator neural network where:
1. Each WEIGHT is a propagator Cell
2. Training samples contribute TRAINABLE information to weight cells
3. MERGE performs actual gradient descent (not delegation to external trainer)
4. The propagator network IS the neural network - no metadata layer

TODO: Information Theory Analysis of Merge Semantics, How much does training data affect a particular weight value? How does this compare to standard training?

Key Design Decision:
-------------------
To avoid recursion (merge calling get_weights which triggers more merges),
each TrainableWeight carries ALL the weight values for its configuration.

This means:
- TrainableWeight stores not just "its" value, but ALL weight values
- Merge can perform training without querying other cells
- The forward pass reads from ANY weight cell (they all have the same info)

Architecture:
------------
┌────────────────────────────────────────────────────────────────┐
│  Weight Cell (one per weight, e.g., ih_0_0)                   │
│                                                                │
│  Each cell stores TrainableWeight:                             │
│    all_weights: Dict[str, float] - ALL weights for this config│
│    samples: FrozenSet[str] - which samples contributed        │
│                                                                │
│  merge(w1, w2):                                                │
│    - If w1.samples == w2.samples: idempotent                  │
│    - If superset: absorption                                   │
│    - Else: TRAIN w1 on new samples, return combined           │
│                                                                │
│  Training happens IN the merge, not via external call.         │
└────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Tuple, FrozenSet, Optional
from dataclasses import dataclass

from propagator import Cell, run, initialize_scheduler
from propagator.nothing import nothing_p
from propagator.supported_values import supported, Supported
from propagator.tms import make_tms, tms_query, kick_out, bring_in
from propagator.merge import assign_merge_operation


# =============================================================================
# Activation Functions
# =============================================================================

def sigmoid(x: float) -> float:
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(y: float) -> float:
    return y * (1 - y)


# =============================================================================
# TrainableWeight - Self-contained trainable value
# =============================================================================

@dataclass(frozen=False)
class TrainableWeight:
    """
    A neural network weight configuration that knows how to train itself.
    
    CRITICAL: Contains ALL weight values, not just "its" value.
    This allows merge to perform training without querying other cells,
    avoiding infinite recursion.
    
    Attributes:
        all_weights: Dict of ALL weight values for this trained configuration
        samples: Set of sample IDs that contributed to these weights
        training_data: The actual training data (inputs, targets)
        hidden_size: Network architecture
        input_size: Network architecture
        output_size: Network architecture
        epochs_per_merge: Training epochs when merging
        learning_rate: Learning rate for training
    """
    all_weights: Dict[str, float]
    samples: FrozenSet[str]
    training_data: List[Tuple[List[float], List[float]]]
    hidden_size: int
    input_size: int
    output_size: int
    epochs_per_merge: int = 500
    learning_rate: float = 0.5
    
    def get_value(self, weight_id: str) -> float:
        """Get a specific weight's value."""
        return self.all_weights.get(weight_id, 0.0)
    
    def __repr__(self):
        n = len(self.samples)
        nw = len(self.all_weights)
        return f"TW(n={n}, {nw} weights)"
    
    def __eq__(self, other):
        if isinstance(other, TrainableWeight):
            return self.samples == other.samples
        return False
    
    def __hash__(self):
        return hash(self.samples)
    
    def get_samples_data(self, sample_ids: FrozenSet[str]) -> List[Tuple]:
        """Get training data for specific sample IDs."""
        indices = [int(s.split('_')[1]) for s in sample_ids]
        return [self.training_data[i] for i in indices]
    
    def train_incremental(self, new_samples: FrozenSet[str]) -> Dict[str, float]:
        """
        Train incrementally on NEW samples, starting from current weights.
        
        This is the actual gradient descent - happens IN merge.
        """
        training_samples = self.get_samples_data(new_samples)
        if not training_samples:
            return self.all_weights.copy()
        
        return self._do_training(self.all_weights.copy(), training_samples, 
                                  self.epochs_per_merge)
    
    def train_full(self, all_samples: FrozenSet[str], 
                   initial_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Train from scratch on ALL samples in the combined set.
        
        This gives better results than incremental but is slower.
        """
        training_samples = self.get_samples_data(all_samples)
        if not training_samples:
            return initial_weights.copy()
        
        # Use more epochs for full training
        epochs = self.epochs_per_merge * len(all_samples)
        return self._do_training(initial_weights.copy(), training_samples, epochs)
    
    def _do_training(self, weights: Dict[str, float], 
                     training_samples: List[Tuple], epochs: int) -> Dict[str, float]:
        """Core training loop."""
        if not training_samples:
            return weights
        
        # Copy weights into arrays
        w_ih = [[weights[f"ih_{i}_{h}"] 
                for h in range(self.hidden_size)]
                for i in range(self.input_size)]
        w_ho = [[weights[f"ho_{h}_{o}"] 
                for o in range(self.output_size)]
                for h in range(self.hidden_size)]
        
        # Train
        for _ in range(epochs):
            for inputs, targets in training_samples:
                # Forward
                h_in = [sum(inputs[i] * w_ih[i][h] 
                           for i in range(self.input_size))
                        for h in range(self.hidden_size)]
                h_out = [sigmoid(x) for x in h_in]
                
                o_in = [sum(h_out[h] * w_ho[h][o] 
                           for h in range(self.hidden_size))
                        for o in range(self.output_size)]
                o_out = [sigmoid(x) for x in o_in]
                
                # Backward
                o_err = [targets[o] - o_out[o] for o in range(self.output_size)]
                o_delta = [o_err[o] * sigmoid_deriv(o_out[o]) 
                          for o in range(self.output_size)]
                
                h_err = [sum(o_delta[o] * w_ho[h][o] 
                            for o in range(self.output_size))
                        for h in range(self.hidden_size)]
                h_delta = [h_err[h] * sigmoid_deriv(h_out[h]) 
                          for h in range(self.hidden_size)]
                
                # Update
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        w_ho[h][o] += self.learning_rate * h_out[h] * o_delta[o]
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        w_ih[i][h] += self.learning_rate * inputs[i] * h_delta[h]
        
        # Convert back to dict
        result = {}
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                result[f"ih_{i}_{h}"] = w_ih[i][h]
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                result[f"ho_{h}_{o}"] = w_ho[h][o]
        
        return result
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass with these weights."""
        # Hidden layer
        h_out = []
        for h in range(self.hidden_size):
            total = sum(inputs[i] * self.all_weights[f"ih_{i}_{h}"]
                       for i in range(self.input_size))
            h_out.append(sigmoid(total))
        
        # Output layer
        o_out = []
        for o in range(self.output_size):
            total = sum(h_out[h] * self.all_weights[f"ho_{h}_{o}"]
                       for h in range(self.hidden_size))
            o_out.append(sigmoid(total))
        
        return o_out


def is_trainable_weight(x) -> bool:
    return isinstance(x, TrainableWeight)


# =============================================================================
# Merge Operation - THIS IS WHERE TRAINING HAPPENS
# =============================================================================

# Cache to avoid redundant training
_merge_cache: Dict[Tuple[FrozenSet[str], FrozenSet[str]], TrainableWeight] = {}

# Global setting for merge mode
# 'full' = retrain from scratch on combined samples (slower but more accurate)
# 'incremental' = start from one, train on new samples (faster but less accurate)
_merge_mode: str = 'full'

# Store initial weights for full retraining
_initial_weights: Dict[str, float] = {}


def set_merge_mode(mode: str):
    """Set the merge mode: 'full' or 'incremental'."""
    global _merge_mode
    if mode not in ('full', 'incremental'):
        raise ValueError(f"mode must be 'full' or 'incremental', got {mode}")
    _merge_mode = mode


def set_initial_weights(weights: Dict[str, float]):
    """Set initial weights for full retraining."""
    global _initial_weights
    _initial_weights = weights.copy()


def reset_merge_cache():
    global _merge_cache
    _merge_cache = {}


def merge_trainable_weights(w1: TrainableWeight, w2: TrainableWeight) -> TrainableWeight:
    """
    Merge two trainable weights by TRAINING on the union of samples.
    
    This IS the training operation - not a call to an external trainer.
    The TrainableWeight carries all the information needed to train.
    
    Properties:
    - Idempotent: merge(w, w) = w
    - Absorption: if w1.samples ⊇ w2.samples, return w1
    - Training: otherwise, retrain on combined samples
    """
    # Idempotence: same samples → same result
    if w1.samples == w2.samples:
        return w1
    
    # Absorption: if one is superset of other
    if w1.samples.issuperset(w2.samples):
        return w1
    if w2.samples.issuperset(w1.samples):
        return w2
    
    # Check cache
    cache_key = (w1.samples, w2.samples)
    if cache_key in _merge_cache:
        return _merge_cache[cache_key]
    
    combined_samples = w1.samples | w2.samples
    
    # TRAIN: this is where gradient descent happens IN the merge
    if _merge_mode == 'full':
        # Full retrain from initial weights on all combined samples
        new_weights = w1.train_full(combined_samples, _initial_weights)
    else:
        # Incremental: start from larger, train on new samples only
        if len(w1.samples) >= len(w2.samples):
            base = w1
            new_samples = w2.samples - w1.samples
        else:
            base = w2
            new_samples = w1.samples - w2.samples
        new_weights = base.train_incremental(new_samples)
    
    result = TrainableWeight(
        all_weights=new_weights,
        samples=combined_samples,
        training_data=w1.training_data,
        hidden_size=w1.hidden_size,
        input_size=w1.input_size,
        output_size=w1.output_size,
        epochs_per_merge=w1.epochs_per_merge,
        learning_rate=w1.learning_rate
    )
    
    _merge_cache[cache_key] = result
    return result


# Register the merge operation
assign_merge_operation(merge_trainable_weights, is_trainable_weight, is_trainable_weight)


# =============================================================================
# Propagator Neural Network
# =============================================================================

class PropagatorNeuralNet:
    """
    A neural network where weights ARE propagator cells.
    
    The propagator cells ARE the weights. The merge operation IS training.
    No external trainer. No metadata layer.
    
    Implementation note: We use a SINGLE weight cell that stores all weights
    for a configuration. This is semantically cleaner and avoids coordination
    issues between multiple cells.
    """
    
    def __init__(self, hidden_size: int = 4, epochs_per_merge: int = 500,
                 learning_rate: float = 0.5):
        self.hidden_size = hidden_size
        self.epochs_per_merge = epochs_per_merge
        self.learning_rate = learning_rate
        
        self.input_size = 0
        self.output_size = 0
        
        self.weight_cell: Optional[Cell] = None
        self.training_data: List[Tuple] = []
        self.sample_premises: List[str] = []
    
    def setup(self, training_data: List[Tuple], seed: int = None,
              epochs_per_sample: int = 2000, verbose: bool = True,
              merge_mode: str = 'full'):
        """
        Set up the network and train on each sample.
        
        Args:
            training_data: List of (inputs, targets) tuples
            seed: Random seed for weight initialization
            epochs_per_sample: Training epochs for individual samples
            verbose: Print setup progress
            merge_mode: 'full' (retrain from scratch) or 'incremental'
        """
        initialize_scheduler()
        reset_merge_cache()
        set_merge_mode(merge_mode)
        
        self.training_data = training_data
        self.input_size = len(training_data[0][0])
        self.output_size = len(training_data[0][1])
        
        if seed is not None:
            random.seed(seed)
        
        # Create single weight cell
        self.weight_cell = Cell(name="weights")
        
        # Generate initial weights
        initial_weights = {}
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                initial_weights[f"ih_{i}_{h}"] = random.random() * 2 - 1
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                initial_weights[f"ho_{h}_{o}"] = random.random() * 2 - 1
        
        # Store for full retraining
        set_initial_weights(initial_weights)
        
        if verbose:
            print(f"Setting up network: {self.input_size}→{self.hidden_size}→{self.output_size}")
            print(f"Merge mode: {merge_mode}")
            print(f"Training each sample with {epochs_per_sample} epochs")
            print("-" * 60)
        
        # Train on each sample, create TrainableWeight for each
        for idx, (inputs, targets) in enumerate(training_data):
            sample_id = f"sample_{idx}"
            self.sample_premises.append(sample_id)
            
            # Train on this sample (full training, not merge)
            trained_weights = self._train_full(
                initial_weights.copy(),
                [(inputs, targets)],
                epochs_per_sample
            )
            
            tw_trained = TrainableWeight(
                all_weights=trained_weights,
                samples=frozenset([sample_id]),
                training_data=training_data,
                hidden_size=self.hidden_size,
                input_size=self.input_size,
                output_size=self.output_size,
                epochs_per_merge=self.epochs_per_merge,
                learning_rate=self.learning_rate
            )
            
            # Add to cell with TMS support
            self.weight_cell.add_content(
                make_tms(supported(tw_trained, [sample_id]))
            )
            
            if verbose:
                pred = tw_trained.forward(inputs)
                print(f"  {sample_id}: {inputs} → {targets[0]}, predicts {pred[0]:.4f}")
        
        run()
        
        if verbose:
            print("-" * 60)
    
    def _train_full(self, weights: Dict[str, float], 
                    samples: List[Tuple], epochs: int) -> Dict[str, float]:
        """Full training (not incremental merge)."""
        w_ih = [[weights[f"ih_{i}_{h}"] for h in range(self.hidden_size)]
                for i in range(self.input_size)]
        w_ho = [[weights[f"ho_{h}_{o}"] for o in range(self.output_size)]
                for h in range(self.hidden_size)]
        
        for _ in range(epochs):
            for inputs, targets in samples:
                h_in = [sum(inputs[i] * w_ih[i][h] 
                           for i in range(self.input_size))
                        for h in range(self.hidden_size)]
                h_out = [sigmoid(x) for x in h_in]
                
                o_in = [sum(h_out[h] * w_ho[h][o] 
                           for h in range(self.hidden_size))
                        for o in range(self.output_size)]
                o_out = [sigmoid(x) for x in o_in]
                
                o_err = [targets[o] - o_out[o] for o in range(self.output_size)]
                o_delta = [o_err[o] * sigmoid_deriv(o_out[o]) 
                          for o in range(self.output_size)]
                
                h_err = [sum(o_delta[o] * w_ho[h][o] 
                            for o in range(self.output_size))
                        for h in range(self.hidden_size)]
                h_delta = [h_err[h] * sigmoid_deriv(h_out[h]) 
                          for h in range(self.hidden_size)]
                
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        w_ho[h][o] += self.learning_rate * h_out[h] * o_delta[o]
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        w_ih[i][h] += self.learning_rate * inputs[i] * h_delta[h]
        
        result = {}
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                result[f"ih_{i}_{h}"] = w_ih[i][h]
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                result[f"ho_{h}_{o}"] = w_ho[h][o]
        
        return result
    
    def get_weights(self) -> Optional[TrainableWeight]:
        """Get current believed TrainableWeight."""
        if self.weight_cell is None or nothing_p(self.weight_cell.content):
            return None
        
        result = tms_query(self.weight_cell.content)
        if isinstance(result, Supported):
            return result.value
        return result
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass using current believed weights."""
        tw = self.get_weights()
        if tw is None:
            return [0.5] * self.output_size
        return tw.forward(inputs)
    
    def get_believed_samples(self) -> Optional[FrozenSet[str]]:
        """Get which samples are believed."""
        tw = self.get_weights()
        if tw is None:
            return None
        return tw.samples


# =============================================================================
# Demo
# =============================================================================

def demo():
    print("=" * 70)
    print("Propagator Neural Network - Weights ARE Cells, Merge IS Training")
    print("=" * 70)
    print("""
In this implementation:
- The weight cell stores TrainableWeight values
- Each TrainableWeight knows how to train itself
- MERGE performs actual gradient descent
- No external trainer - merge IS training
""")
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    nn = PropagatorNeuralNet(
        hidden_size=4,
        epochs_per_merge=2000,  # More epochs for better accuracy
        learning_rate=0.5
    )
    nn.setup(training_data, seed=42, epochs_per_sample=2000, merge_mode='full')
    
    test_input = [0, 0]
    
    print("\n" + "=" * 70)
    print(f"Forward pass for {test_input} (expected: 0)")
    print("=" * 70)
    
    print("\nWith ALL samples believed:")
    output = nn.forward(test_input)
    samples = nn.get_believed_samples()
    print(f"  Believed samples: {sorted(samples)}")
    print(f"  Output: {output[0]:.4f}")
    
    print("\n--- Testing kick_out ---")
    
    print("\nKick out sample_0 ([0,0]→0):")
    kick_out('sample_0')
    run()  # This triggers merge → which IS training
    output = nn.forward(test_input)
    samples = nn.get_believed_samples()
    print(f"  Believed: {sorted(samples) if samples else 'None'}")
    print(f"  Output: {output[0]:.4f}")
    
    print("\nKick out sample_3 ([1,1]→0):")
    kick_out('sample_3')
    run()
    output = nn.forward(test_input)
    samples = nn.get_believed_samples()
    print(f"  Believed: {sorted(samples) if samples else 'None'}")
    print(f"  Output: {output[0]:.4f}")
    print(f"  (Only samples 1,2 - both output=1)")
    
    print("\nBring back both:")
    bring_in('sample_0')
    bring_in('sample_3')
    run()
    output = nn.forward(test_input)
    samples = nn.get_believed_samples()
    print(f"  Believed: {sorted(samples) if samples else 'None'}")
    print(f"  Output: {output[0]:.4f}")
    
    print("\n Kick out sample 1 ([0,1]→1) and sample 2 ([1,0]→1):")
    kick_out('sample_1')
    kick_out('sample_2')
    run()
    output = nn.forward(test_input)
    samples = nn.get_believed_samples()
    print(f"  Believed: {sorted(samples) if samples else 'None'}")
    print(f"  Output: {output[0]:.4f}")

    print("\n Bring back both, kick out [1,1] ->0 :")
    bring_in('sample_1')
    bring_in('sample_2')
    kick_out('sample_3')
    run()
    output = nn.forward(test_input)
    samples = nn.get_believed_samples()
    print(f"  Believed: {sorted(samples) if samples else 'None'}")
    print(f"  Output: {output[0]:.4f}")


    print("\n" + "=" * 70)
    print("Testing different configurations")
    print("=" * 70)
    
    configs = [
        ([0, 1, 2, 3], "All 4 samples"),
        ([1, 2], "Samples 1,2 (output=1)"),
        ([0, 3], "Samples 0,3 (output=0)"),
        ([2], "Only sample 2 ([1,0]→1)"),
    ]
    
    for believed, description in configs:
        # Set worldview
        for p in nn.sample_premises:
            kick_out(p)
        for idx in believed:
            bring_in(f"sample_{idx}")
        run()
        
        output = nn.forward(test_input)
        samples = nn.get_believed_samples()
        print(f"\n{description}:")
        print(f"  Believed: {sorted(samples) if samples else 'None'}")
        print(f"  Prediction for {test_input}: {output[0]:.4f}")
    
    # Restore
    for p in nn.sample_premises:
        bring_in(p)
    run()
    
    print("\n" + "=" * 70)
    print("Full XOR truth table (all samples)")
    print("=" * 70)
    
    print(f"\n{'Input':<10} | {'Output':<10} | {'Expected':<10}")
    print("-" * 35)
    for inputs, targets in training_data:
        output = nn.forward(inputs)
        print(f"{inputs!s:<10} | {output[0]:.4f}     | {targets[0]}")


def demo_incremental_merge():
    """Show that merge IS training, incrementally."""
    print("\n" + "=" * 70)
    print("Demonstration: Merge IS Training (Full Retrain Mode)")
    print("=" * 70)
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    nn = PropagatorNeuralNet(hidden_size=4, epochs_per_merge=2000)
    nn.setup(training_data, seed=42, epochs_per_sample=2000, verbose=False,
             merge_mode='full')
    
    # Start with just sample_0
    for p in nn.sample_premises:
        kick_out(p)
    bring_in('sample_0')
    run()
    
    test_input = [1, 0]
    
    print(f"\nTest input: {test_input}")
    print("-" * 50)
    
    output = nn.forward(test_input)
    print(f"1. sample_0 only: {output[0]:.4f}")
    
    print("\n   bring_in(sample_1)")
    print("   → merge(TW({sample_0}), TW({sample_1}))")
    print("   → TrainableWeight.train_full() called IN the merge")
    print("   → Full retrain on {sample_0, sample_1}")
    
    bring_in('sample_1')
    run()
    output = nn.forward(test_input)
    print(f"   Result: {output[0]:.4f}")
    
    print("\n   bring_in(sample_2)")
    print("   → Full retrain on {sample_0, sample_1, sample_2}")
    bring_in('sample_2')
    run()
    output = nn.forward(test_input)
    print(f"   Result: {output[0]:.4f}")
    
    print("\n   bring_in(sample_3)")
    print("   → Full retrain on all 4 samples")
    bring_in('sample_3')
    run()
    output = nn.forward(test_input)
    print(f"   Result (all 4): {output[0]:.4f}")
    
    print("\n" + "-" * 50)
    print("Each bring_in triggered a merge.")
    print("Each merge performed FULL retraining on all believed samples.")
    print("No external trainer was invoked - merge IS training.")


if __name__ == "__main__":
    demo()
    demo_incremental_merge()
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE SUMMARY")
    print("=" * 70)
    print("""
This implementation has NO METADATA LAYER:

┌────────────────────────────────────────────────────────────────┐
│  PROPAGATOR CELL (weights)                                     │
│                                                                │
│  Contains TMS with TrainableWeight values:                     │
│                                                                │
│    TrainableWeight:                                            │
│      all_weights: {'ih_0_0': 0.73, 'ih_0_1': -0.42, ...}      │
│      samples: frozenset({'sample_0'})                          │
│      training_data: [([0,0],[0]), ([0,1],[1]), ...]           │
│      train(new_samples) → performs gradient descent            │
│                                                                │
│  merge(TW1, TW2):                                              │
│    if TW1.samples == TW2.samples: return TW1  (idempotent)    │
│    if TW1.samples ⊃ TW2.samples: return TW1  (absorption)     │
│    else: new_weights = TW1.train(TW2.samples - TW1.samples)   │
│          return TrainableWeight(new_weights, union)            │
│                                                                │
│  THE MERGE IS THE TRAINING.                                    │
└────────────────────────────────────────────────────────────────┘

Key insight: TrainableWeight is SELF-CONTAINED.
- It carries ALL weight values (not just "its" weight)
- It carries the training data
- It knows how to train itself

When kick_out/bring_in changes the worldview:
  → TMS computes strongest consequence
  → This may invoke merge
  → merge() calls TrainableWeight.train()
  → Gradient descent happens RIGHT THERE
  → No external orchestration needed

This is a TRUE propagator neural network:
- The cell content IS the network weights
- The merge operation IS training
- No external trainer
- No metadata layer
""")
