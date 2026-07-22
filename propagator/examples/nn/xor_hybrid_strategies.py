"""
Hybrid Propagator Neural Network with Pluggable Merge Strategies

This combines the structural elegance of "weights as cells" with the
semantic correctness of "retrain on query", making the merge behavior
extensible and configurable.

Key Insight:
-----------
The merge operation in a propagator network is PROGRAMMABLE. We can:

1. LinearApproximation: Sum gradient contributions (fast, approximate)
2. RetrainOnMerge: Actually retrain when configurations merge (exact, slow)
3. IncrementalCorrection: Do a few gradient steps to correct (middle ground)
4. Cached: Memoize any strategy for efficiency

The right strategy depends on the problem:
- Linear: Good for small learning rates, near-linear regimes, quick prototyping
- Retrain: Good for exact results, complex interactions, final analysis
- Incremental: Good for balance of speed and accuracy

Architecture:
------------
┌────────────────────────────────────────────────────────────────────┐
│  TrainedWeights - Represents a trained configuration              │
│    config: FrozenSet of sample IDs                                │
│    weights: Dict[str, float] - the actual weight values           │
│    strategy: MergeStrategy - how to merge with other configs      │
│    trainer: NetworkTrainer - reference for retraining             │
└────────────────────────────────────────────────────────────────────┘

Merge(w1, w2) delegates to w1.strategy.merge(w1, w2)
- LinearStrategy: interpolate/sum weight values
- RetrainStrategy: train on union of samples
- IncrementalStrategy: start from one, fine-tune on the other's samples
"""

from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, FrozenSet, Tuple, Callable
from dataclasses import dataclass, field

from propagator import Cell, run, initialize_scheduler
from propagator.supported_values import supported, Supported
from propagator.tms import make_tms, tms_query, kick_out, bring_in
from propagator.merge import assign_merge_operation


# =============================================================================
# Network Trainer - The actual training logic (shared across strategies)
# =============================================================================

def sigmoid(x: float) -> float:
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


class NetworkTrainer:
    """
    Encapsulates the training logic for a neural network.
    
    This is the "oracle" that strategies can call to get trained weights
    for any configuration of samples.
    """
    
    def __init__(self, training_data: List[Tuple[List[float], List[float]]],
                 input_size: int, hidden_size: int, output_size: int,
                 seed: int = None, default_epochs: int = 5000, 
                 default_lr: float = 1.0):
        self.training_data = training_data
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seed = seed
        self.default_epochs = default_epochs
        self.default_lr = default_lr
        
        # Cache for memoization
        self._cache: Dict[FrozenSet[str], Dict[str, float]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_trainings = 0
    
    def get_initial_weights(self) -> Dict[str, float]:
        """Get deterministic initial weights based on seed."""
        if self.seed is not None:
            random.seed(self.seed)
        
        weights = {}
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                weights[f"ih_{i}_{h}"] = random.random() * 2 - 1
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                weights[f"ho_{h}_{o}"] = random.random() * 2 - 1
        return weights
    
    def train(self, config: FrozenSet[str], 
              initial_weights: Dict[str, float] = None,
              epochs: int = None, lr: float = None,
              use_cache: bool = True) -> Dict[str, float]:
        """
        Train on the given configuration of samples.
        
        Args:
            config: Set of sample IDs to train on
            initial_weights: Starting weights (None = use seed-based init)
            epochs: Training epochs (None = use default)
            lr: Learning rate (None = use default)
            use_cache: Whether to use/update the cache
        
        Returns:
            Dict mapping weight names to values
        """
        if use_cache and config in self._cache:
            self.cache_hits += 1
            return self._cache[config].copy()
        
        self.cache_misses += 1
        self.total_trainings += 1
        
        epochs = epochs or self.default_epochs
        lr = lr or self.default_lr
        
        # Get sample indices from IDs
        sample_indices = [int(s.split('_')[1]) for s in config]
        samples = [self.training_data[i] for i in sample_indices]
        
        if not samples:
            return self.get_initial_weights()
        
        # Initialize weights
        if initial_weights is None:
            weights = self.get_initial_weights()
        else:
            weights = initial_weights.copy()
        
        # Extract into arrays for training
        w_ih = [[weights[f"ih_{i}_{h}"] for h in range(self.hidden_size)]
                for i in range(self.input_size)]
        w_ho = [[weights[f"ho_{h}_{o}"] for o in range(self.output_size)]
                for h in range(self.hidden_size)]
        
        # Train
        for _ in range(epochs):
            for inputs, targets in samples:
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
                o_delta = [o_err[o] * o_out[o] * (1 - o_out[o]) 
                          for o in range(self.output_size)]
                
                h_err = [sum(o_delta[o] * w_ho[h][o] 
                            for o in range(self.output_size))
                        for h in range(self.hidden_size)]
                h_delta = [h_err[h] * h_out[h] * (1 - h_out[h]) 
                          for h in range(self.hidden_size)]
                
                # Update
                for h in range(self.hidden_size):
                    for o in range(self.output_size):
                        w_ho[h][o] += lr * h_out[h] * o_delta[o]
                for i in range(self.input_size):
                    for h in range(self.hidden_size):
                        w_ih[i][h] += lr * inputs[i] * h_delta[h]
        
        # Convert back to dict
        result = {}
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                result[f"ih_{i}_{h}"] = w_ih[i][h]
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                result[f"ho_{h}_{o}"] = w_ho[h][o]
        
        if use_cache:
            self._cache[config] = result.copy()
        
        return result
    
    def forward(self, inputs: List[float], weights: Dict[str, float]) -> List[float]:
        """Run forward pass with given weights."""
        # Hidden layer
        h_out = []
        for h in range(self.hidden_size):
            total = sum(inputs[i] * weights[f"ih_{i}_{h}"]
                       for i in range(self.input_size))
            h_out.append(sigmoid(total))
        
        # Output layer
        o_out = []
        for o in range(self.output_size):
            total = sum(h_out[h] * weights[f"ho_{h}_{o}"]
                       for h in range(self.hidden_size))
            o_out.append(sigmoid(total))
        
        return o_out


# =============================================================================
# Merge Strategies - Pluggable behaviors for combining trained configurations
# =============================================================================

class MergeStrategy(ABC):
    """Abstract base class for merge strategies."""
    
    @abstractmethod
    def merge(self, w1: 'TrainedWeights', w2: 'TrainedWeights') -> 'TrainedWeights':
        """Merge two trained weight configurations."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        pass


class LinearStrategy(MergeStrategy):
    """
    Linear approximation: average weights proportionally to sample counts.
    
    Fast but approximate. Works well when:
    - Learning rates are small
    - Samples don't strongly conflict
    - You need quick exploration
    """
    
    @property
    def name(self) -> str:
        return "Linear"
    
    def merge(self, w1: 'TrainedWeights', w2: 'TrainedWeights') -> 'TrainedWeights':
        combined_config = w1.config | w2.config
        
        # Idempotence checks
        if w1.config == w2.config:
            return w1
        if w1.config.issuperset(w2.config):
            return w1
        if w2.config.issuperset(w1.config):
            return w2
        
        # Weighted average based on sample counts
        n1 = len(w1.config)
        n2 = len(w2.config)
        total = n1 + n2
        
        combined_weights = {}
        for key in w1.weights:
            v1 = w1.weights[key]
            v2 = w2.weights[key]
            combined_weights[key] = (v1 * n1 + v2 * n2) / total
        
        return TrainedWeights(
            config=combined_config,
            weights=combined_weights,
            strategy=self,
            trainer=w1.trainer
        )


class RetrainStrategy(MergeStrategy):
    """
    Exact retraining: train from scratch on the union of samples.
    
    Slow but exact. Use when:
    - You need precise results
    - Samples have complex interactions
    - Final analysis/reporting
    """
    
    @property
    def name(self) -> str:
        return "Retrain"
    
    def merge(self, w1: 'TrainedWeights', w2: 'TrainedWeights') -> 'TrainedWeights':
        combined_config = w1.config | w2.config
        
        # Idempotence checks
        if w1.config == w2.config:
            return w1
        if w1.config.issuperset(w2.config):
            return w1
        if w2.config.issuperset(w1.config):
            return w2
        
        # Actually retrain on the combined configuration
        combined_weights = w1.trainer.train(combined_config)
        
        return TrainedWeights(
            config=combined_config,
            weights=combined_weights,
            strategy=self,
            trainer=w1.trainer
        )


class IncrementalStrategy(MergeStrategy):
    """
    Incremental correction: start from one config, fine-tune on new samples.
    
    Middle ground between linear and full retrain. Use when:
    - You want better than linear accuracy
    - Full retraining is too slow
    - Samples are added incrementally
    
    Args:
        correction_epochs: How many epochs to fine-tune
        correction_lr: Learning rate for fine-tuning
    """
    
    def __init__(self, correction_epochs: int = 500, correction_lr: float = 0.5):
        self.correction_epochs = correction_epochs
        self.correction_lr = correction_lr
    
    @property
    def name(self) -> str:
        return f"Incremental({self.correction_epochs})"
    
    def merge(self, w1: 'TrainedWeights', w2: 'TrainedWeights') -> 'TrainedWeights':
        combined_config = w1.config | w2.config
        
        # Idempotence checks
        if w1.config == w2.config:
            return w1
        if w1.config.issuperset(w2.config):
            return w1
        if w2.config.issuperset(w1.config):
            return w2
        
        # Start from the larger config's weights
        if len(w1.config) >= len(w2.config):
            base_weights = w1.weights.copy()
            new_samples = w2.config - w1.config
        else:
            base_weights = w2.weights.copy()
            new_samples = w1.config - w2.config
        
        # Fine-tune on the new samples
        combined_weights = w1.trainer.train(
            combined_config,
            initial_weights=base_weights,
            epochs=self.correction_epochs,
            lr=self.correction_lr,
            use_cache=False  # Don't cache intermediate results
        )
        
        return TrainedWeights(
            config=combined_config,
            weights=combined_weights,
            strategy=self,
            trainer=w1.trainer
        )


class AdaptiveStrategy(MergeStrategy):
    """
    Adaptive strategy: choose between linear and retrain based on heuristics.
    
    Uses linear approximation for quick exploration, switches to retraining
    when accuracy matters (e.g., near the final configuration).
    
    Args:
        retrain_threshold: Retrain if combined config has this many samples
        accuracy_check: Optional function to verify approximation quality
    """
    
    def __init__(self, retrain_threshold: int = None,
                 linear_strategy: LinearStrategy = None,
                 retrain_strategy: RetrainStrategy = None):
        self.retrain_threshold = retrain_threshold
        self.linear = linear_strategy or LinearStrategy()
        self.retrain = retrain_strategy or RetrainStrategy()
        self.linear_count = 0
        self.retrain_count = 0
    
    @property
    def name(self) -> str:
        return f"Adaptive(linear={self.linear_count}, retrain={self.retrain_count})"
    
    def merge(self, w1: 'TrainedWeights', w2: 'TrainedWeights') -> 'TrainedWeights':
        combined_config = w1.config | w2.config
        
        # Idempotence checks
        if w1.config == w2.config:
            return w1
        if w1.config.issuperset(w2.config):
            return w1
        if w2.config.issuperset(w1.config):
            return w2
        
        # Decide which strategy to use
        use_retrain = False
        
        if self.retrain_threshold is not None:
            use_retrain = len(combined_config) >= self.retrain_threshold
        
        if use_retrain:
            self.retrain_count += 1
            return self.retrain.merge(w1, w2)
        else:
            self.linear_count += 1
            return self.linear.merge(w1, w2)


# =============================================================================
# TrainedWeights - The value type stored in propagator cells
# =============================================================================

@dataclass
class TrainedWeights:
    """
    Represents a set of neural network weights trained on a specific configuration.
    
    This is the value type that goes into propagator cells.
    The merge behavior is delegated to the strategy.
    """
    config: FrozenSet[str]           # Which samples this was trained on
    weights: Dict[str, float]        # The actual weight values
    strategy: MergeStrategy          # How to merge with other configs
    trainer: NetworkTrainer          # Reference for retraining
    
    def __repr__(self):
        samples = sorted(self.config)
        return f"Weights(trained_on={samples}, strategy={self.strategy.name})"
    
    def __eq__(self, other):
        if isinstance(other, TrainedWeights):
            return self.config == other.config
        return False
    
    def __hash__(self):
        return hash(self.config)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Run forward pass with these weights."""
        return self.trainer.forward(inputs, self.weights)


def is_trained_weights(x) -> bool:
    return isinstance(x, TrainedWeights)


def merge_trained_weights(w1: TrainedWeights, w2: TrainedWeights) -> TrainedWeights:
    """
    Merge trained weights using the strategy from w1.
    
    This is the key extension point - the merge behavior is determined
    by the strategy attached to the weights.
    """
    return w1.strategy.merge(w1, w2)


# Register the merge operation
assign_merge_operation(merge_trained_weights, is_trained_weights, is_trained_weights)


# =============================================================================
# Hybrid Propagator Neural Network
# =============================================================================

class HybridPropagatorNetwork:
    """
    Neural network with propagator-based configuration tracking and
    pluggable merge strategies.
    
    Usage:
        nn = HybridPropagatorNetwork(training_data, strategy=RetrainStrategy())
        nn.setup()
        
        # Full training
        print(nn.forward([1, 0]))
        
        # Remove a sample
        kick_out('sample_0')
        run()
        print(nn.forward([1, 0]))  # Uses chosen merge strategy
    """
    
    def __init__(self, training_data: List[Tuple[List[float], List[float]]],
                 strategy: MergeStrategy = None,
                 seed: int = None, epochs: int = 5000, lr: float = 1.0,
                 hidden_size: int = 4):
        self.training_data = training_data
        self.strategy = strategy or RetrainStrategy()
        
        input_size = len(training_data[0][0])
        output_size = len(training_data[0][1])
        
        self.trainer = NetworkTrainer(
            training_data=training_data,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=seed,
            default_epochs=epochs,
            default_lr=lr
        )
        
        self.weights_cell = None
        self.sample_premises: List[str] = []
    
    def setup(self, verbose: bool = True):
        """Initialize the network and train on each sample."""
        initialize_scheduler()
        
        self.weights_cell = Cell(name="network_weights")
        
        if verbose:
            print(f"Setting up network with {self.strategy.name} strategy")
            print("-" * 60)
        
        for idx in range(len(self.training_data)):
            sample_id = f"sample_{idx}"
            self.sample_premises.append(sample_id)
            
            # Train on just this sample
            config = frozenset([sample_id])
            weights = self.trainer.train(config)
            
            trained = TrainedWeights(
                config=config,
                weights=weights,
                strategy=self.strategy,
                trainer=self.trainer
            )
            
            if verbose:
                inputs, targets = self.training_data[idx]
                pred = trained.forward(inputs)
                print(f"  {sample_id}: {inputs} → {targets[0]}, "
                      f"predicts {pred[0]:.4f}")
            
            # Add to cell with this sample as premise
            self.weights_cell.add_content(
                make_tms(supported(trained, [sample_id]))
            )
        
        run()
        
        if verbose:
            print("-" * 60)
            print(f"Initial trainings: {self.trainer.total_trainings}")
    
    def get_current_weights(self) -> Optional[TrainedWeights]:
        """Get the current believed weights."""
        result = tms_query(self.weights_cell.content)
        if result is None:
            return None
        if isinstance(result, Supported):
            return result.value
        return result
    
    def forward(self, inputs: List[float]) -> Optional[List[float]]:
        """Forward pass with current believed weights."""
        weights = self.get_current_weights()
        if weights is None:
            return None
        return weights.forward(inputs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'strategy': self.strategy.name,
            'total_trainings': self.trainer.total_trainings,
            'cache_hits': self.trainer.cache_hits,
            'cache_misses': self.trainer.cache_misses,
        }


# =============================================================================
# Demo and Comparison
# =============================================================================

def demo_single_strategy(strategy: MergeStrategy, training_data, 
                         test_input, seed=42, verbose=True):
    """Run a demo with a specific strategy."""
    nn = HybridPropagatorNetwork(
        training_data, 
        strategy=strategy,
        seed=seed,
        epochs=5000,
        lr=1.0,
        hidden_size=4
    )
    nn.setup(verbose=verbose)
    
    results = {}
    
    # Test different configurations
    configs = [
        ([0, 1, 2, 3], "All 4"),
        ([1, 2], "1,2 (output=1)"),
        ([0, 3], "0,3 (output=0)"),
        ([1, 2, 3], "Without 0"),
    ]
    
    for believed_indices, name in configs:
        for p in nn.sample_premises:
            kick_out(p)
        for idx in believed_indices:
            bring_in(f"sample_{idx}")
        run()
        
        pred = nn.forward(test_input)
        results[name] = pred[0] if pred else None
        
        if verbose:
            print(f"  {name}: {pred[0]:.4f}" if pred else f"  {name}: None")
    
    # Restore all
    for p in nn.sample_premises:
        bring_in(p)
    run()
    
    return results, nn.get_stats()


def demo():
    print("=" * 70)
    print("Hybrid Propagator Neural Network - Pluggable Merge Strategies")
    print("=" * 70)
    print("""
This demonstrates different merge strategies for propagator neural networks.

Each strategy offers different tradeoffs:
- Linear: Fast, approximate (sums gradient-like contributions)
- Retrain: Slow, exact (trains from scratch on combined samples)
- Incremental: Medium, good (fine-tunes existing weights)
- Adaptive: Chooses based on heuristics

All strategies maintain:
- Idempotence: merge(x, x) = x
- Subset absorption: merge(A, A∪B) = A∪B
- TMS compatibility: kick_out/bring_in work correctly
""")
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    test_input = [1, 0]
    
    strategies = [
        ("Linear (fast, approximate)", LinearStrategy()),
        ("Retrain (slow, exact)", RetrainStrategy()),
        ("Incremental(500)", IncrementalStrategy(correction_epochs=500)),
        ("Incremental(1000)", IncrementalStrategy(correction_epochs=1000)),
        ("Adaptive(threshold=3)", AdaptiveStrategy(retrain_threshold=3)),
    ]
    
    print(f"\nTest input: {test_input} (expected: 1)")
    print("=" * 70)
    
    all_results = {}
    
    for name, strategy in strategies:
        print(f"\n### {name} ###")
        results, stats = demo_single_strategy(
            strategy, training_data, test_input, seed=42, verbose=True
        )
        all_results[name] = results
        print(f"  Stats: {stats}")
    
    # Comparison table
    print("\n" + "=" * 70)
    print("Comparison Table")
    print("=" * 70)
    
    configs = ["All 4", "1,2 (output=1)", "0,3 (output=0)", "Without 0"]
    
    print(f"\n{'Strategy':<25} | " + " | ".join(f"{c:<12}" for c in configs))
    print("-" * (25 + 4 + 15 * len(configs)))
    
    for name, _ in strategies:
        results = all_results[name]
        row = f"{name:<25} | "
        row += " | ".join(f"{results[c]:.4f}      " if results[c] else "None        " 
                         for c in configs)
        print(row)
    
    print("-" * (25 + 4 + 15 * len(configs)))
    print(f"{'Expected':<25} | {'~0.97':>12} | {'~0.99':>12} | {'~0.01':>12} | {'~0.97':>12}")


def demo_full_truth_table():
    """Compare strategies on the full XOR truth table."""
    print("\n" + "=" * 70)
    print("Full XOR Truth Table Comparison")
    print("=" * 70)
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    strategies = [
        ("Linear", LinearStrategy()),
        ("Retrain", RetrainStrategy()),
        ("Incr(500)", IncrementalStrategy(500)),
    ]
    
    print(f"\n{'Input':<8} | " + " | ".join(f"{n:<10}" for n, _ in strategies) + " | Expected")
    print("-" * 60)
    
    for inputs, targets in training_data:
        row = f"{inputs!s:<8} |"
        
        for name, strategy in strategies:
            nn = HybridPropagatorNetwork(
                training_data, strategy=strategy, seed=42, 
                epochs=5000, hidden_size=4
            )
            nn.setup(verbose=False)
            
            pred = nn.forward(inputs)
            row += f" {pred[0]:.4f}     |" if pred else " None       |"
        
        row += f" {targets[0]}"
        print(row)


def demo_adaptive_behavior():
    """Show how adaptive strategy switches between linear and retrain."""
    print("\n" + "=" * 70)
    print("Adaptive Strategy Behavior")
    print("=" * 70)
    print("""
The adaptive strategy uses linear approximation for small configurations
and switches to full retraining for larger ones.
""")
    
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    ]
    
    # Use adaptive with threshold=3 (retrain when 3+ samples)
    strategy = AdaptiveStrategy(retrain_threshold=3)
    
    nn = HybridPropagatorNetwork(
        training_data, strategy=strategy, seed=42, epochs=5000
    )
    nn.setup(verbose=False)
    
    test_input = [1, 0]
    
    configs = [
        [0],           # 1 sample - linear
        [0, 1],        # 2 samples - linear (merge)
        [0, 1, 2],     # 3 samples - RETRAIN (threshold)
        [0, 1, 2, 3],  # 4 samples - RETRAIN
    ]
    
    print(f"\nTest input: {test_input}")
    print(f"Strategy: Adaptive(retrain_threshold=3)")
    print("-" * 60)
    
    for believed in configs:
        for p in nn.sample_premises:
            kick_out(p)
        for idx in believed:
            bring_in(f"sample_{idx}")
        run()
        
        pred = nn.forward(test_input)
        print(f"Samples {believed}: prediction={pred[0]:.4f}")
        print(f"  → {strategy.name}")
    
    print("-" * 60)
    print(f"Final: {strategy.linear_count} linear merges, "
          f"{strategy.retrain_count} retrainings")


if __name__ == "__main__":
    demo()
    demo_full_truth_table()
    demo_adaptive_behavior()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This hybrid approach gives you the best of both worlds:

1. STRUCTURAL ELEGANCE
   - Weights conceptually live in propagator cells
   - TMS tracks which samples are believed
   - kick_out/bring_in navigate the configuration space

2. SEMANTIC CORRECTNESS
   - Merge strategy determines how to combine configurations
   - RetrainStrategy gives exact results
   - Caching prevents redundant computation

3. FLEXIBILITY
   - Swap strategies based on your needs
   - LinearStrategy for fast exploration
   - RetrainStrategy for final results
   - IncrementalStrategy for balance
   - AdaptiveStrategy for automatic switching

4. EXTENSIBILITY
   - Create custom strategies by subclassing MergeStrategy
   - Could add: ensemble methods, Bayesian updates, transfer learning, etc.

The key insight: MERGE IS PROGRAMMABLE.
The propagator framework doesn't dictate HOW configurations combine,
only THAT they combine according to the lattice structure.
""")
