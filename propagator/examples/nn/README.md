# Propagator NN Examples: Techniques, Tradeoffs, and When To Use Each

This directory contains multiple neural-network experiments built around propagator ideas.
They are intentionally different: some prioritize bidirectional constraints, some prioritize premise-aware provenance/retraction, and some prioritize practical SGD-style training behavior.

## Quick Map

- `recursive_nn_common.py`
  - Shared building blocks for recursive, bidirectional probabilistic networks.
  - Includes safe bidirectional multiplication for interval inputs, sigmoid/logit propagators, and recursive layer wiring.

- `bidirectional_recursive_neuron.py`
  - Smallest end-to-end bidirectional examples.
  - Demonstrates forward + inverse inference, recursive network construction, and interval-safe inverse arithmetic.
  - Use this first to understand constraint flow and inverse queries over input intervals.

- `xor_recursive_bidirectional.py`
  - Recursive XOR network on top of the shared blocks.
  - Includes:
    - fixed XOR parameters (forward + inverse checks),
    - branch-aware inverse queries (`y high` with `x2 near 0` vs `x2 near 1`),
    - direct output-layer fitting in logit space,
    - **Phase A** premise-aware fit retraction demo (`kick_out` / `bring_in`).
  - Best for understanding bidirectionality and premise toggling in a compact script.

- `xor_propagator_nn.py`
  - "Merge is training" design.
  - Stores trainable weights as propagator content and performs gradient descent during merge of training-support configurations.
  - Supports epoch/learning-rate knobs and premise toggling over sample configurations.
  - Best for studying true training dynamics inside merge semantics.

- `xor_retrain_on_query.py`
  - Lattice-of-configurations approach with caching.
  - Query-time retraining for the currently believed training-sample subset.
  - `kick_out`/`bring_in` moves in configuration space; cache avoids recomputing seen configs.
  - Best for exact "train on active sample subset" semantics.

- `xor_propagator_weights.py`
  - Gradient-contribution accumulation in weight cells.
  - Approximate compositional training: merges sample-wise gradient deltas (idempotent by source).
  - Fast and premise-aware, but not equivalent to full SGD ordering behavior.

- `xor_hybrid_strategies.py`
  - Strategy playground for merge behavior:
    - linear approximation,
    - retrain-on-merge,
    - incremental correction.
  - Best for comparing speed/accuracy tradeoffs under one framework.

- `xor_prop_net.py`
  - Historical mixed implementation.
  - Useful as a cautionary reference: shows why some architectures do **not** make `kick_out` affect trained weights.

- `true_xor_prop_net.py`
  - Richer comparison between conventional and propagator-centric orchestration.
  - Helpful for seeing practical integration patterns and trigger-based training flow.

## Two Main Families

1. Constraint-fit / direct solve family
- Representative: `xor_recursive_bidirectional.py` (logit-space fit), `bidirectional_recursive_neuron.py`.
- Idea: solve parameter constraints from specified equations/corners, then use the resulting network bidirectionally.
- Hyperparameters are target constraints (e.g., desired corner probabilities), not epoch loops.

2. Epoch-based gradient family
- Representatives: `xor_propagator_nn.py`, `xor_retrain_on_query.py`, `xor_hybrid_strategies.py`, `xor_propagator_weights.py`.
- Idea: optimize weights via repeated gradient updates over data.
- Hyperparameters include epochs, learning rate, and merge/retraining strategy.

## Logit Constraint Fit vs Gradient Descent: Intuition

### What logit-space fitting is doing
For a sigmoid output neuron:
- Forward model: `y = sigmoid(z)`
- Inverse transform: `z = logit(y) = ln(y / (1-y))`

If you choose target output probabilities at a few design points (for XOR corners), then in logit space those become linear constraints on the output-layer affine relation. In this example we fit:
- `z = w * (h1 + h2) + b`

Given two target corners (one low, one high), you get two equations in two unknowns (`w`, `b`) and can solve directly.

Benefits:
- No iterative optimizer loop.
- Deterministic (subject to interval tolerances).
- Easy to make premise-aware by attaching/retracting equations.

Limits:
- It is model-design fitting, not data-driven learning over noisy datasets.
- Scales poorly when many parameters are unconstrained or when constraints are inconsistent.

### What epoch-based gradient descent is doing
Given training data and a loss, repeatedly update weights:
- `w <- w - eta * dL/dw` (or equivalent sign convention)
- repeated for many epochs and samples.

Benefits:
- Handles many parameters and larger datasets.
- Naturally supports noisy/overdetermined settings.
- Standard in practical ML workflows.

Limits:
- Hyperparameter-sensitive (epochs, learning rate, init, strategy).
- Not exact; converges approximately.
- In propagator settings, provenance/retraction semantics need explicit architecture choices.

## Premise Retraction in Practice

In this directory, premise toggling is most meaningful when training/fitting evidence is represented explicitly as supported assumptions:
- Works clearly in:
  - `xor_recursive_bidirectional.py` Phase A fit retraction,
  - `xor_retrain_on_query.py`,
  - `xor_propagator_nn.py` (depending on merge strategy/state design).
- Beware of architectures where learned weights become plain floats disconnected from premise provenance; `kick_out` then cannot influence learned parameters.

## Suggested Learning Path

1. `bidirectional_recursive_neuron.py`
2. `xor_recursive_bidirectional.py`
3. `xor_retrain_on_query.py`
4. `xor_propagator_nn.py`
5. `xor_hybrid_strategies.py`
6. `xor_propagator_weights.py`
7. `xor_prop_net.py` and `true_xor_prop_net.py` as comparative references
