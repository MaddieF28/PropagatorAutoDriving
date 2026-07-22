"""Recursive bidirectional XOR network using shared propagator NN blocks.

This module includes:
1) Fixed-parameter XOR network checks (forward + inverse).
2) Branch-aware inverse queries for the two XOR modes when y is high.
3) Constraint-based fitting of the output layer for a trainable XOR variant.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from propagator import (
    Cell,
    Tms,
    bring_in,
    difference,
    initialize_scheduler,
    kick_out,
    make_interval,
    nothing_p,
    premise_in,
    run,
    tms_query,
)

from .recursive_nn_common import (
    affine_neuron,
    assert_high_probability,
    assert_interval,
    assert_low_probability,
    point_interval,
    probabilistic_neuron,
    safe_product,
    sigmoid_scalar,
    two_layer_prob_network,
)


@dataclass(frozen=True)
class XORConstraintFitConfig:
    """Hyperparameters for the constraint-fit XOR variant.

    Note: this network uses direct constraint fitting in logit space, not
    gradient-descent epochs. `low_prob`/`high_prob` control the desired corner
    probabilities and therefore how sharp the fitted separator is.
    """

    low_prob: float = 0.01
    high_prob: float = 0.99
    inverse_high_low: float = 0.9
    inverse_high_high: float = 1.0
    branch_anchor_eps: float = 0.01


def _resolved_content(value):
    """Resolve TMS/Supported wrappers to their usable value."""
    if nothing_p(value):
        return None
    if isinstance(value, Tms):
        resolved = tms_query(value)
        if resolved is None:
            return None
        if nothing_p(resolved):
            return None
        if hasattr(resolved, "value"):
            return resolved.value
        return resolved
    if hasattr(value, "value") and hasattr(value, "support"):
        return value.value
    return value


def clamp_fixed_hidden_xor_parameters(hidden_weights, hidden_biases):
    """Set hidden-layer parameters for XOR feature extraction.

    Hidden units:
      h1 approximates OR:    sigmoid(20*x1 + 20*x2 - 10)
      h2 approximates NAND:  sigmoid(-20*x1 - 20*x2 + 30)
    """
    hidden_weights[0][0].add_content(point_interval(20.0))
    hidden_weights[0][1].add_content(point_interval(20.0))
    hidden_biases[0].add_content(point_interval(-10.0))

    hidden_weights[1][0].add_content(point_interval(-20.0))
    hidden_weights[1][1].add_content(point_interval(-20.0))
    hidden_biases[1].add_content(point_interval(30.0))


def clamp_fixed_output_xor_parameters(output_weights, output_bias):
    """Set output-layer parameters implementing XOR on hidden features."""
    output_weights[0].add_content(point_interval(20.0))
    output_weights[1].add_content(point_interval(20.0))
    output_bias.add_content(point_interval(-30.0))


def build_xor_network_with_cells(
    x1,
    x2,
    y,
    hidden_weights,
    hidden_biases,
    output_weights,
    output_bias,
    bidirectional=True,
):
    two_layer_prob_network(
        [x1, x2],
        hidden_weights,
        hidden_biases,
        output_weights,
        output_bias,
        y,
        bidirectional=bidirectional,
    )


def build_fixed_xor_network(x1, x2, y, bidirectional=True):
    h1_w1 = Cell(name="h1_w1")
    h1_w2 = Cell(name="h1_w2")
    h2_w1 = Cell(name="h2_w1")
    h2_w2 = Cell(name="h2_w2")
    h1_b = Cell(name="h1_b")
    h2_b = Cell(name="h2_b")

    o_w1 = Cell(name="o_w1")
    o_w2 = Cell(name="o_w2")
    o_b = Cell(name="o_b")

    hidden_weights = [[h1_w1, h1_w2], [h2_w1, h2_w2]]
    hidden_biases = [h1_b, h2_b]
    output_weights = [o_w1, o_w2]

    build_xor_network_with_cells(
        x1,
        x2,
        y,
        hidden_weights,
        hidden_biases,
        output_weights,
        o_b,
        bidirectional=bidirectional,
    )
    clamp_fixed_hidden_xor_parameters(hidden_weights, hidden_biases)
    clamp_fixed_output_xor_parameters(output_weights, o_b)


def hidden_feature_values_for_input(x1, x2):
    """Compute scalar hidden features induced by fixed hidden params."""
    h1 = sigmoid_scalar(20.0 * x1 + 20.0 * x2 - 10.0)
    h2 = sigmoid_scalar(-20.0 * x1 - 20.0 * x2 + 30.0)
    return h1, h2


def fit_xor_output_layer_constraints(config: XORConstraintFitConfig):
    """Fit a tied output layer z = w*(h1+h2) + b in logit space.

    This is a constraint-based fit that enforces XOR symmetry (ow1 = ow2).
    """
    low_prob = config.low_prob
    high_prob = config.high_prob
    low_logit = math.log(low_prob / (1.0 - low_prob))
    high_logit = math.log(high_prob / (1.0 - high_prob))

    # Sums of hidden features for a low-output corner and a high-output corner.
    h1_low, h2_low = hidden_feature_values_for_input(0.0, 0.0)
    h1_high, h2_high = hidden_feature_values_for_input(0.0, 1.0)
    s_low = h1_low + h2_low
    s_high = h1_high + h2_high

    w_shared = Cell(name="fit_w_shared")
    b_shared = Cell(name="fit_b_shared")

    s_low_cell = Cell(name="fit_s_low")
    s_high_cell = Cell(name="fit_s_high")
    z_low_cell = Cell(name="fit_z_low")
    z_high_cell = Cell(name="fit_z_high")

    s_low_cell.add_content(point_interval(s_low))
    s_high_cell.add_content(point_interval(s_high))
    z_low_cell.add_content(point_interval(low_logit))
    z_high_cell.add_content(point_interval(high_logit))

    affine_neuron([s_low_cell], [w_shared], b_shared, z_low_cell)
    affine_neuron([s_high_cell], [w_shared], b_shared, z_high_cell)

    # Explicitly couple the two equations to make w directly inferable.
    delta_z = Cell(name="fit_delta_z")
    delta_s = Cell(name="fit_delta_s")
    difference(z_high_cell, z_low_cell, delta_z)
    difference(s_high_cell, s_low_cell, delta_s)
    # Linear slope relation: delta_z = w_shared * delta_s
    safe_product(w_shared, delta_s, delta_z)

    return w_shared.content, b_shared.content


def fit_xor_output_layer_constraints_with_premises(config: XORConstraintFitConfig):
    """Phase A: premise-aware fit state for output-layer retraining demo."""
    premise_low = "fit-sample-low"
    premise_high = "fit-sample-high"
    return {
        "premise_low": premise_low,
        "premise_high": premise_high,
    }


def _fit_xor_output_layer_from_active_premises(
    config: XORConstraintFitConfig,
    premise_low: str,
    premise_high: str,
):
    """Recompute output-layer fit using only currently believed premises."""
    low_prob = config.low_prob
    high_prob = config.high_prob
    low_logit = math.log(low_prob / (1.0 - low_prob))
    high_logit = math.log(high_prob / (1.0 - high_prob))

    h1_low, h2_low = hidden_feature_values_for_input(0.0, 0.0)
    h1_high, h2_high = hidden_feature_values_for_input(0.0, 1.0)
    s_low = h1_low + h2_low
    s_high = h1_high + h2_high

    use_low = premise_in(premise_low)
    use_high = premise_in(premise_high)

    w_shared = Cell(name="phaseA_worldview_w_shared")
    b_shared = Cell(name="phaseA_worldview_b_shared")

    s_low_cell = Cell(name="phaseA_worldview_s_low")
    s_high_cell = Cell(name="phaseA_worldview_s_high")
    z_low_cell = Cell(name="phaseA_worldview_z_low")
    z_high_cell = Cell(name="phaseA_worldview_z_high")

    if use_low:
        s_low_cell.add_content(point_interval(s_low))
        z_low_cell.add_content(point_interval(low_logit))
        affine_neuron([s_low_cell], [w_shared], b_shared, z_low_cell)

    if use_high:
        s_high_cell.add_content(point_interval(s_high))
        z_high_cell.add_content(point_interval(high_logit))
        affine_neuron([s_high_cell], [w_shared], b_shared, z_high_cell)

    if use_low and use_high:
        delta_z = Cell(name="phaseA_worldview_delta_z")
        delta_s = Cell(name="phaseA_worldview_delta_s")
        difference(z_high_cell, z_low_cell, delta_z)
        difference(s_high_cell, s_low_cell, delta_s)
        safe_product(w_shared, delta_s, delta_z)

    run()
    return _resolved_content(w_shared.content), _resolved_content(b_shared.content)


def evaluate_phase_a_premise_fit_retraction(
    config: XORConstraintFitConfig = XORConstraintFitConfig(),
):
    """Demonstrate premise retraction effects on constraint-fit parameters."""
    initialize_scheduler()
    fit_state = fit_xor_output_layer_constraints_with_premises(config)
    premise_low = fit_state["premise_low"]
    premise_high = fit_state["premise_high"]

    w_both, b_both = _fit_xor_output_layer_from_active_premises(
        config,
        premise_low,
        premise_high,
    )
    assert_interval(w_both, "phaseA_w_both")
    assert_interval(b_both, "phaseA_b_both")

    kick_out(premise_high)
    w_without_high, b_without_high = _fit_xor_output_layer_from_active_premises(
        config,
        premise_low,
        premise_high,
    )

    bring_in(premise_high)
    kick_out(premise_low)
    w_without_low, b_without_low = _fit_xor_output_layer_from_active_premises(
        config,
        premise_low,
        premise_high,
    )
    bring_in(premise_low)

    # With one premise removed, system is underdetermined and at least one
    # parameter should become unavailable (None).
    if w_without_high is not None and b_without_high is not None:
        raise AssertionError(
            "Expected underdetermined fit after removing high-premise equation"
        )
    if w_without_low is not None and b_without_low is not None:
        raise AssertionError(
            "Expected underdetermined fit after removing low-premise equation"
        )

    w_restored, b_restored = _fit_xor_output_layer_from_active_premises(
        config,
        premise_low,
        premise_high,
    )
    assert_interval(w_restored, "phaseA_w_restored")
    assert_interval(b_restored, "phaseA_b_restored")

    print("Phase A: premise-aware fit retraction")
    print("  both premises in   -> w", w_both, "b", b_both)
    print("  kick_out(high)     -> w", w_without_high, "b", b_without_high)
    print("  kick_out(low)      -> w", w_without_low, "b", b_without_low)
    print("  bring_in(all)      -> w", w_restored, "b", b_restored)


def build_fitted_xor_network(
    x1,
    x2,
    y,
    bidirectional=True,
    config: XORConstraintFitConfig = XORConstraintFitConfig(),
):
    """Build XOR network with fixed hidden layer and constraint-fitted output layer."""
    h1_w1 = Cell(name="fit_h1_w1")
    h1_w2 = Cell(name="fit_h1_w2")
    h2_w1 = Cell(name="fit_h2_w1")
    h2_w2 = Cell(name="fit_h2_w2")
    h1_b = Cell(name="fit_h1_b")
    h2_b = Cell(name="fit_h2_b")

    o_w1 = Cell(name="fit_o_w1")
    o_w2 = Cell(name="fit_o_w2")
    o_b = Cell(name="fit_o_b")

    hidden_weights = [[h1_w1, h1_w2], [h2_w1, h2_w2]]
    hidden_biases = [h1_b, h2_b]
    output_weights = [o_w1, o_w2]

    build_xor_network_with_cells(
        x1,
        x2,
        y,
        hidden_weights,
        hidden_biases,
        output_weights,
        o_b,
        bidirectional=bidirectional,
    )

    clamp_fixed_hidden_xor_parameters(hidden_weights, hidden_biases)

    fitted_w, fitted_b = fit_xor_output_layer_constraints(config)
    o_w1.add_content(fitted_w)
    o_w2.add_content(fitted_w)
    o_b.add_content(fitted_b)


def evaluate_forward_truth_table_case_recursive(cases, builder, index=0):
    if index >= len(cases):
        return

    x1v, x2v, expected = cases[index]
    x1 = Cell(name=f"forward_x1_{index}")
    x2 = Cell(name=f"forward_x2_{index}")
    y = Cell(name=f"forward_y_{index}")

    builder(x1, x2, y, bidirectional=False)
    x1.add_content(point_interval(x1v))
    x2.add_content(point_interval(x2v))

    if expected == 1:
        assert_high_probability(y.content, f"xor_forward_case_{index}")
    else:
        assert_low_probability(y.content, f"xor_forward_case_{index}")

    print(f"Forward case {index}: ({x1v}, {x2v}) -> {y.content}")
    evaluate_forward_truth_table_case_recursive(cases, builder, index + 1)


def evaluate_inverse_queries_basic():
    # Existing regression checks.
    xa = Cell(name="inverse_a_x1")
    x2a = Cell(name="inverse_a_x2")
    ya = Cell(name="inverse_a_y")
    build_fixed_xor_network(xa, x2a, ya, bidirectional=True)
    xa.add_content(make_interval(0.0, 1.0))
    x2a.add_content(point_interval(0.0))
    ya.add_content(make_interval(0.9, 1.0))
    assert_interval(xa.content, "inverse_a_x1")
    assert_interval(x2a.content, "inverse_a_x2")
    if xa.content.low < 0.5:
        raise AssertionError(f"inverse_a_x1 not narrowed high enough: {xa.content}")

    xb = Cell(name="inverse_b_x1")
    x2b = Cell(name="inverse_b_x2")
    yb = Cell(name="inverse_b_y")
    build_fixed_xor_network(xb, x2b, yb, bidirectional=True)
    xb.add_content(make_interval(0.0, 1.0))
    x2b.add_content(point_interval(1.0))
    yb.add_content(make_interval(0.9, 1.0))
    assert_interval(xb.content, "inverse_b_x1")
    assert_interval(x2b.content, "inverse_b_x2")
    if xb.content.high > 0.5:
        raise AssertionError(f"inverse_b_x1 not narrowed low enough: {xb.content}")

    print("Inverse basic A (x2=0, y high): x1", xa.content, "x2", x2a.content)
    print("Inverse basic B (x2=1, y high): x1", xb.content, "x2", x2b.content)


def branch_aware_inverse_xor_high_output(
    config: XORConstraintFitConfig = XORConstraintFitConfig(),
):
    """Explicitly return both XOR branches for y≈1.

    Branch 1: x2 near 0 -> infer x1 near 1.
    Branch 2: x2 near 1 -> infer x1 near 0.
    """
    b1_x1 = Cell(name="branch1_x1")
    b1_x2 = Cell(name="branch1_x2")
    b1_y = Cell(name="branch1_y")
    build_fixed_xor_network(b1_x1, b1_x2, b1_y, bidirectional=True)
    b1_x1.add_content(make_interval(0.0, 1.0))
    b1_x2.add_content(make_interval(0.0, config.branch_anchor_eps))
    b1_y.add_content(make_interval(config.inverse_high_low, config.inverse_high_high))

    b2_x1 = Cell(name="branch2_x1")
    b2_x2 = Cell(name="branch2_x2")
    b2_y = Cell(name="branch2_y")
    build_fixed_xor_network(b2_x1, b2_x2, b2_y, bidirectional=True)
    b2_x1.add_content(make_interval(0.0, 1.0))
    b2_x2.add_content(make_interval(1.0 - config.branch_anchor_eps, 1.0))
    b2_y.add_content(make_interval(config.inverse_high_low, config.inverse_high_high))

    assert_interval(b1_x1.content, "branch1_x1")
    assert_interval(b1_x2.content, "branch1_x2")
    assert_interval(b2_x1.content, "branch2_x1")
    assert_interval(b2_x2.content, "branch2_x2")
    if b1_x1.content.low < 0.48:
        raise AssertionError(f"branch1 not high-input mode: {b1_x1.content}")
    if b2_x1.content.high > 0.53:
        raise AssertionError(f"branch2 not low-input mode: {b2_x1.content}")

    print("Branch-aware inverse for y≈1")
    print("  Branch 1 (x2≈0) -> x1", b1_x1.content, "x2", b1_x2.content)
    print("  Branch 2 (x2≈1) -> x1", b2_x1.content, "x2", b2_x2.content)


def evaluate_trainable_xor_variant_regression(
    config: XORConstraintFitConfig = XORConstraintFitConfig(),
):
    """Run regression checks on the constraint-fitted XOR variant."""
    cases = [
        (0.0, 0.0, 0),
        (0.0, 1.0, 1),
        (1.0, 0.0, 1),
        (1.0, 1.0, 0),
    ]
    def fitted_builder(ix1, ix2, iy, bidirectional=True):
        return build_fitted_xor_network(
            ix1,
            ix2,
            iy,
            bidirectional=bidirectional,
            config=config,
        )

    evaluate_forward_truth_table_case_recursive(cases, fitted_builder)

    xq = Cell(name="fit_inverse_x1")
    yq = Cell(name="fit_inverse_y")
    x2q = Cell(name="fit_inverse_x2")
    build_fitted_xor_network(xq, x2q, yq, bidirectional=True, config=config)
    xq.add_content(make_interval(0.0, 1.0))
    x2q.add_content(make_interval(0.0, config.branch_anchor_eps))
    yq.add_content(make_interval(config.inverse_high_low, config.inverse_high_high))
    assert_interval(xq.content, "fit_inverse_x1")
    assert_interval(x2q.content, "fit_inverse_x2")

    print("Trainable variant inverse (x2≈0, y high): x1", xq.content, "x2", x2q.content)


def run_all_checks(config: XORConstraintFitConfig = XORConstraintFitConfig()):
    cases = [
        (0.0, 0.0, 0),
        (0.0, 1.0, 1),
        (1.0, 0.0, 1),
        (1.0, 1.0, 0),
    ]

    print("Fixed XOR forward regression")
    evaluate_forward_truth_table_case_recursive(cases, build_fixed_xor_network)
    print()

    print("Fixed XOR inverse regression")
    evaluate_inverse_queries_basic()
    print()

    branch_aware_inverse_xor_high_output(config=config)
    print()

    print("Constraint-fitted XOR variant regression")
    evaluate_trainable_xor_variant_regression(config=config)
    print()

    evaluate_phase_a_premise_fit_retraction(config=config)
    print("All XOR recursive bidirectional checks passed.")


if __name__ == "__main__":
    run_all_checks()
