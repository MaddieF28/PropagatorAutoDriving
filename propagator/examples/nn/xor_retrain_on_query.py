"""
XOR Neural Network with Retrain-on-Query

This demonstrates the CORRECT approach to propagator-based neural networks:

The lattice isn't over OUTPUTS (averaging predictions is meaningless).
The lattice is over TRAINING CONFIGURATIONS.

Each point in the lattice represents:
- A subset of training samples
- The weights that result from training on that subset
- The predictions those weights produce

Navigation:
- kick_out(sample) → move to a point trained WITHOUT that sample
- bring_in(sample) → move to a point trained WITH that sample

The "merge" operation is RETRAINING - computing the weights for a new
configuration that hasn't been visited yet.

This is computationally expensive but semantically correct.
Caching ensures we don't retrain for configurations we've seen before.
"""

import math
import random
from typing import FrozenSet, Dict, Tuple, Optional, List
from dataclasses import dataclass

from propagator import Cell, run, initialize_scheduler
from propagator.supported_values import supported, Supported
from propagator.tms import make_tms, tms_query, kick_out, bring_in, premise_in
from propagator.merge import assign_merge_operation


# =============================================================================
# Training Configuration as Lattice Element
# =============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """
    A point in the lattice of training configurations.
    
    This represents "the network trained on exactly these samples".
    The lattice ordering is by subset: config A ≤ config B iff A.samples ⊆ B.samples
    
    The VALUE at each lattice point is computed lazily (by training).
    """
    samples: FrozenSet[int]  # Which sample indices are included
    
    def __repr__(self):
        return f"TrainingConfig({sorted(self.samples)})"


@dataclass
class TrainedNetwork:
    """
    The result of training on a specific configuration.
    
    This is what we COMPUTE when we reach a lattice point.
    """
    config: TrainingConfig
    weights_ih: List[List[float]]  # Input -> Hidden weights
    weights_ho: List[List[float]]  # Hidden -> Output weights
    
    def forward(self, inputs: List[float]) -> float:
        """Run forward pass with these weights."""
        hidden_size = len(self.weights_ih[0])
        output_size = len(self.weights_ho[0])
        
        # Hidden layer
        h_in = [sum(inputs[i] * self.weights_ih[i][h] 
                   for i in range(len(inputs)))
                for h in range(hidden_size)]
        h_out = [sigmoid(x) for x in h_in]
        
        # Output layer
        o_in = [sum(h_out[h] * self.weights_ho[h][o] 
                   for h in range(hidden_size))
                for o in range(output_size)]
        o_out = [sigmoid(x) for x in o_in]
        
        return o_out[0]


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


# =============================================================================
# Training Cache - Memoization over the Lattice
# =============================================================================

class TrainingCache:
    """
    Cache of trained networks indexed by training configuration.
    
    This is the key to making retrain-on-query efficient:
    - Each unique configuration is trained only ONCE
    - Subsequent queries for the same configuration use cached result
    
    The cache represents explored regions of the lattice.
    """
    
    def __init__(self, training_data: List[Tuple], seed: int = None,
                 epochs: int = 5000, lr: float = 1.0, hidden_size: int = 4):
        self.training_data = training_data
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.hidden_size = hidden_size
        self.input_size = len(training_data[0][0])
        self.output_size = len(training_data[0][1])
        
        # Cache: config -> trained network
        self._cache: Dict[TrainingConfig, TrainedNetwork] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_or_train(self, config: TrainingConfig) -> Optional[TrainedNetwork]:
        """Get cached network or train a new one."""
        if not config.samples:
            return None  # Can't train on zero samples
            
        if config in self._cache:
            self.cache_hits += 1
            return self._cache[config]
        
        self.cache_misses += 1
        network = self._train(config)
        self._cache[config] = network
        return network
    
    def _train(self, config: TrainingConfig) -> TrainedNetwork:
        """Actually train a network on the given configuration."""
        # Use deterministic initialization based on seed
        if self.seed is not None:
            random.seed(self.seed)
        
        # Initialize weights
        w_ih = [[random.random() * 2 - 1 for _ in range(self.hidden_size)] 
                for _ in range(self.input_size)]
        w_ho = [[random.random() * 2 - 1 for _ in range(self.output_size)] 
                for _ in range(self.hidden_size)]
        
        # Get training samples for this config
        samples = [self.training_data[i] for i in sorted(config.samples)]
        
        # Train
        for _ in range(self.epochs):
            for inputs, targets in samples:
                # Forward
                h_in = [sum(inputs[i] * w_ih[i][h] for i in range(self.input_size))
                        for h in range(self.hidden_size)]
                h_out = [sigmoid(x) for x in h_in]
                
                o_in = [sum(h_out[h] * w_ho[h][o] for h in range(self.hidden_size))
                        for o in range(self.output_size)]
                o_out = [sigmoid(x) for x in o_in]
                
                # Backward
                o_err = [targets[o] - o_out[o] for o in range(self.output_size)]
                o_delta = [o_err[o] * o_out[o] * (1 - o_out[o]) 
                          for o in range(self.output_size)]
                
                h_err = [sum(o_delta[o] * w_ho[h][o] for o in range(self.output_size))
                         for h in range(self.hidden_size)]
                h_delta = [h_err[h] * h_out[h] * (1 - h_out[h]) 
                          for h in range(self.hidden_size)]
                
                # Update
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        w_ho[h][o] += self.lr * h_out[h] * o_delta[o]
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        w_ih[i][h] += self.lr * inputs[i] * h_delta[h]
        
        return TrainedNetwork(
            config=config,
            weights_ih=w_ih,
            weights_ho=w_ho
        )


# =============================================================================
# Prediction Cell - Queries trigger lattice navigation
# =============================================================================

class ConfiguredPrediction:
    """
    A prediction that knows its training configuration.
    
    This is what goes into cells. Merging predictions means:
    - Union the training configurations
    - Retrain on the combined configuration
    - Return the prediction from the retrained network
    """
    
    def __init__(self, value: float, config: TrainingConfig, 
                 cache: TrainingCache, test_input: List[float]):
        self.value = value
        self.config = config
        self._cache = cache
        self._test_input = test_input
    
    def __repr__(self):
        samples = sorted(self.config.samples)
        return f"Prediction({self.value:.4f}, trained_on={samples})"
    
    def __eq__(self, other):
        if isinstance(other, ConfiguredPrediction):
            return self.config == other.config
        return False
    
    def __hash__(self):
        return hash(self.config)


def is_configured_prediction(x) -> bool:
    return isinstance(x, ConfiguredPrediction)


def merge_configured_predictions(p1: ConfiguredPrediction, 
                                  p2: ConfiguredPrediction) -> ConfiguredPrediction:
    """
    Merge predictions by RETRAINING on the union of configurations.
    
    This is the KEY insight: merge doesn't average predictions.
    It computes what the prediction WOULD BE if trained on both sets.
    """
    # Union of training samples
    combined_samples = p1.config.samples | p2.config.samples
    combined_config = TrainingConfig(combined_samples)
    
    # If same config, no need to retrain
    if p1.config == p2.config:
        return p1
    
    # Check if one subsumes the other
    if p1.config.samples.issuperset(p2.config.samples):
        return p1
    if p2.config.samples.issuperset(p1.config.samples):
        return p2
    
    # Need to train on combined config
    # Use p1's cache (they should be the same)
    network = p1._cache.get_or_train(combined_config)
    
    if network is None:
        return p1  # Fallback
    
    new_value = network.forward(p1._test_input)
    return ConfiguredPrediction(new_value, combined_config, p1._cache, p1._test_input)


# Register merge operation
assign_merge_operation(merge_configured_predictions, 
                       is_configured_prediction, is_configured_prediction)


# =============================================================================
# XOR Network with Retrain-on-Query
# =============================================================================

class XORRetrainOnQuery:
    """
    XOR network that retrains when the training configuration changes.
    
    Usage:
        nn = XORRetrainOnQuery(seed=42)
        nn.setup(training_data, test_input=[1, 0])
        
        # Query with all samples
        print(nn.get_prediction())  # Trains on all 4, predicts
        
        # Remove a sample
        kick_out('sample_0')
        run()
        print(nn.get_prediction())  # Retrains on 1,2,3, predicts
    """
    
    def __init__(self, seed: int = None, epochs: int = 5000, 
                 lr: float = 1.0, hidden_size: int = 4):
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.hidden_size = hidden_size
        self.cache = None
        self.prediction_cell = None
        self.sample_premises = []
        self.test_input = None
    
    def setup(self, training_data: List[Tuple], test_input: List[float], 
              verbose: bool = True):
        """Set up the network with training data and test input."""
        initialize_scheduler()
        
        self.test_input = test_input
        self.cache = TrainingCache(
            training_data, 
            seed=self.seed,
            epochs=self.epochs,
            lr=self.lr,
            hidden_size=self.hidden_size
        )
        
        self.prediction_cell = Cell(name="prediction")
        
        if verbose:
            print(f"Setting up XOR network for test input: {test_input}")
            print(f"Training parameters: epochs={self.epochs}, lr={self.lr}, hidden={self.hidden_size}")
            print("-" * 60)
        
        # Each sample contributes its "single-sample prediction" as a lattice point
        for idx in range(len(training_data)):
            premise = f'sample_{idx}'
            self.sample_premises.append(premise)
            
            # Create single-sample config
            config = TrainingConfig(frozenset([idx]))
            
            # Train on just this sample (will be cached)
            network = self.cache.get_or_train(config)
            value = network.forward(test_input)
            
            if verbose:
                inputs, targets = training_data[idx]
                print(f"  Sample {idx}: train on {inputs}→{targets[0]}, "
                      f"predicts {test_input}→{value:.4f}")
            
            # Add to cell with premise
            prediction = ConfiguredPrediction(value, config, self.cache, test_input)
            self.prediction_cell.add_content(
                make_tms(supported(prediction, [premise]))
            )
        
        run()
        
        if verbose:
            print("-" * 60)
            print(f"Cache: {self.cache.cache_misses} trainings performed")
    
    def get_prediction(self) -> Optional[ConfiguredPrediction]:
        """Get the current prediction based on believed samples."""
        result = tms_query(self.prediction_cell.content)
        if result is None:
            return None
        if isinstance(result, Supported):
            return result.value
        return result
    
    def get_value(self) -> Optional[float]:
        """Get just the numeric prediction value."""
        pred = self.get_prediction()
        if pred is None:
            return None
        return pred.value
    
    def get_config(self) -> Optional[TrainingConfig]:
        """Get the current training configuration."""
        pred = self.get_prediction()
        if pred is None:
            return None
        return pred.config


def demo():
    print("=" * 70)
    print("XOR Neural Network with RETRAIN-ON-QUERY")
    print("=" * 70)
    print("""
KEY INSIGHT: The lattice is over TRAINING CONFIGURATIONS, not outputs.

- Each point = "network trained on samples {X, Y, Z}"
- Merge = train on the UNION of samples
- kick_out = navigate to a point WITHOUT that sample
- bring_in = navigate to a point WITH that sample

This is semantically correct: we're asking "what would the network
predict if trained on this subset of samples?"
""")
    
    training_data = [
        ([0, 0], [0]),  # Sample 0
        ([0, 1], [1]),  # Sample 1
        ([1, 0], [1]),  # Sample 2
        ([1, 1], [0]),  # Sample 3
    ]
    
    test_input = [1, 0]  # Expected: 1
    
    nn = XORRetrainOnQuery(seed=42, epochs=100000, lr=1.0, hidden_size=4)
    nn.setup(training_data, test_input)
    
    print("\n" + "=" * 70)
    print(f"Querying predictions for test input {test_input} (expected: 1)")
    print("=" * 70)
    
    # Query with all samples
    print("\nWith ALL samples believed:")
    pred = nn.get_prediction()
    print(f"  {pred}")
    print(f"  → This network was trained on ALL 4 XOR samples!")
    print(f"  → Cache stats: {nn.cache.cache_hits} hits, {nn.cache.cache_misses} misses")
    
    # Kick out samples trained on output=0
    print("\n--- Navigate the lattice by removing samples ---")
    
    print("\nKick out 'sample_0' (train on [0,0]→0):")
    kick_out('sample_0')
    run()
    pred = nn.get_prediction()
    print(f"  {pred}")
    print(f"  → Retrained on samples 1,2,3")
    print(f"  → Cache stats: {nn.cache.cache_hits} hits, {nn.cache.cache_misses} misses")
    
    print("\nKick out 'sample_3' (train on [1,1]→0):")
    kick_out('sample_3')
    run()
    pred = nn.get_prediction()
    print(f"  {pred}")
    print(f"  → Retrained on samples 1,2 only (both output=1)")
    print(f"  → This should predict HIGH for [1,0]!")
    
    print("\nBring back sample_0 and sample_3:")
    bring_in('sample_0')
    bring_in('sample_3')
    run()
    pred = nn.get_prediction()
    print(f"  {pred}")
    print(f"  → Back to full training (should use cache)")
    print(f"  → Cache stats: {nn.cache.cache_hits} hits, {nn.cache.cache_misses} misses")
    
    print("\n" + "=" * 70)
    print("Exploring training configurations")
    print("=" * 70)
    
    configs_to_test = [
        ([0, 1, 2, 3], "All 4 samples (full XOR)"),
        ([1, 2], "Samples 1,2 only (both output=1)"),
        ([0, 3], "Samples 0,3 only (both output=0)"),
        ([0, 1, 2], "Samples 0,1,2 (without [1,1])"),
        ([1, 2, 3], "Samples 1,2,3 (without [0,0])"),
        ([2], "Sample 2 only ([1,0]→1)"),
    ]
    
    for believed_indices, description in configs_to_test:
        # Reset worldview
        for p in nn.sample_premises:
            kick_out(p)
        for idx in believed_indices:
            bring_in(f'sample_{idx}')
        run()
        
        pred = nn.get_prediction()
        
        print(f"\n{description}:")
        if pred:
            print(f"  Config: trained on {sorted(pred.config.samples)}")
            print(f"  Prediction: {pred.value:.4f}")
        else:
            print(f"  No prediction (no samples believed)")
    
    # Restore all
    for p in nn.sample_premises:
        bring_in(p)
    
    print("\n" + "=" * 70)
    print(f"Final cache stats: {nn.cache.cache_hits} hits, {nn.cache.cache_misses} misses")
    print("=" * 70)


def demo_all_inputs():
    """Test the network on all XOR inputs with different configurations."""
    print("\n" + "=" * 70)
    print("Full XOR Truth Table with Different Training Configs")
    print("=" * 70)
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    test_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    
    configs = [
        [0, 1, 2, 3],  # Full XOR
        [1, 2],        # Only output=1 samples
        [0, 3],        # Only output=0 samples
        [0,1,2],        # No 1-1 output

    ]
    
    print("\n" + "-" * 60)
    print(f"{'Input':<10} | {'Full XOR':<12} | {'Only 1s':<12} | {'Only 0s':<12} {'Remove 1,1 -> 0'}")
    print("-" * 60)
    
    for test_input in test_inputs:
        expected = test_input[0] ^ test_input[1]
        row = f"{test_input!s:<10} |"
        
        for config_indices in configs:
            nn = XORRetrainOnQuery(seed=42, epochs=10000, lr=1.0, hidden_size=4)
            nn.setup(training_data, test_input, verbose=False)
            
            # Set config
            for p in nn.sample_premises:
                kick_out(p)
            for idx in config_indices:
                bring_in(f'sample_{idx}')
            run()
            
            pred = nn.get_value()
            row += f" {pred:.4f}      |"
        
        row += f"  (expected: {expected})"
        print(row)
    
    print("-" * 60)


if __name__ == "__main__":
    demo()
    demo_all_inputs()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This demonstrates the CORRECT propagator approach to neural networks:

1. The LATTICE is over training configurations, not outputs
   - Each point: "network trained on samples {A, B, C}"
   - Lattice order: subset inclusion

2. MERGE means RETRAIN
   - merge(config_A, config_B) = train on (samples_A ∪ samples_B)
   - This is semantically correct!

3. CACHING makes this practical
   - Each configuration is trained only once
   - Subsequent queries use cached results
   - The cache IS the explored portion of the lattice

4. kick_out/bring_in NAVIGATE the lattice
   - They don't average or interpolate
   - They move to a different training configuration
   - The system retrains (or uses cache) to get the correct answer

This gives us TRUE "what-if" analysis:
- "What would the network predict if we hadn't trained on sample X?"
- Answer: Actually train without X and see!
""")
