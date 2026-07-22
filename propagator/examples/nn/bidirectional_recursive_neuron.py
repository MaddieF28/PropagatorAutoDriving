"""Recursive, bidirectional neural-network building blocks with propagators.

Tracks implemented and evaluated independently:
1) Nonlinear probabilistic neuron with sigmoid/logit bidirectional constraints.
2) Recursive two-layer probabilistic network wiring.
3) Zero-safe product relation for interval priors that include 0.
"""

from __future__ import annotations

from propagator import Cell, difference, make_interval

from .recursive_nn_common import (
    affine_neuron,
    assert_interval_contains,
    assert_interval_narrowed,
    point_interval,
    probabilistic_neuron,
    safe_product,
    sigmoid_scalar,
    two_layer_prob_network,
)


def build_trained_affine_neuron_from_samples(samples):
    """Infer w1, w2, b from three affine samples via constraints."""
    if len(samples) < 3:
        raise ValueError("Need at least three samples to infer w1, w2, and b")

    (x1_1, x2_1, y1) = samples[0]
    (x1_2, _x2_2, y2) = samples[1]
    (_x1_3, x2_3, y3) = samples[2]

    w1 = Cell(name="w1")
    w2 = Cell(name="w2")
    b = Cell(name="b")

    x1_1_cell = Cell(name="x1_1")
    x2_1_cell = Cell(name="x2_1")
    y1_cell = Cell(name="y1")
    x1_2_cell = Cell(name="x1_2")
    y2_cell = Cell(name="y2")
    x2_3_cell = Cell(name="x2_3")
    y3_cell = Cell(name="y3")

    x1_1_cell.add_content(point_interval(x1_1))
    x2_1_cell.add_content(point_interval(x2_1))
    y1_cell.add_content(point_interval(y1))
    x1_2_cell.add_content(point_interval(x1_2))
    y2_cell.add_content(point_interval(y2))
    x2_3_cell.add_content(point_interval(x2_3))
    y3_cell.add_content(point_interval(y3))

    delta_y12 = Cell(name="delta_y12")
    delta_x12 = Cell(name="delta_x12")
    difference(y2_cell, y1_cell, delta_y12)
    difference(x1_2_cell, x1_1_cell, delta_x12)
    safe_product(w1, delta_x12, delta_y12)

    delta_y31 = Cell(name="delta_y31")
    delta_x23 = Cell(name="delta_x23")
    difference(y3_cell, y1_cell, delta_y31)
    difference(x2_3_cell, x2_1_cell, delta_x23)
    safe_product(w2, delta_x23, delta_y31)

    affine_neuron([x1_1_cell, x2_1_cell], [w1, w2], b, y1_cell)
    return w1, w2, b


def evaluate_step_1_nonlinear_single_neuron():
    samples = [
        (0.1, 0.2, 0.26),
        (0.9, 0.2, 0.74),
        (0.1, 0.8, 0.44),
    ]
    w1, w2, b = build_trained_affine_neuron_from_samples(samples)

    x1_forward = Cell(name="x1_forward")
    x2_forward = Cell(name="x2_forward")
    y_forward = Cell(name="y_forward")
    probabilistic_neuron([x1_forward, x2_forward], [w1, w2], b, y_forward)
    x1_forward.add_content(point_interval(0.5))
    x2_forward.add_content(point_interval(0.4))

    expected_forward = sigmoid_scalar(0.6 * 0.5 + 0.3 * 0.4 + 0.14)
    assert_interval_contains(y_forward.content, expected_forward, "step1_forward_y")

    qx1 = Cell(name="step1_query_x1")
    qx2 = Cell(name="step1_query_x2")
    qy = Cell(name="step1_query_y")
    probabilistic_neuron([qx1, qx2], [w1, w2], b, qy)

    qx1.add_content(make_interval(0.0, 1.0))
    qx2.add_content(make_interval(0.4, 0.6))
    qy.add_content(make_interval(0.60, 0.64))

    assert_interval_narrowed(qx1.content, 0.7, "step1_inverse_x1")
    assert_interval_contains(qx2.content, 0.5, "step1_inverse_x2")

    print("Step 1: nonlinear single-neuron constraints")
    print("  learned w1:", w1.content)
    print("  learned w2:", w2.content)
    print("  learned b :", b.content)
    print("  forward y :", y_forward.content)
    print("  inverse x1:", qx1.content)
    print("  inverse x2:", qx2.content)


def evaluate_step_2_two_layer_recursive_network():
    x1 = Cell(name="step2_x1")
    x2 = Cell(name="step2_x2")
    y = Cell(name="step2_y")

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

    two_layer_prob_network([x1, x2], hidden_weights, hidden_biases, output_weights, o_b, y)

    h1_w1.add_content(point_interval(1.2))
    h1_w2.add_content(point_interval(0.8))
    h2_w1.add_content(point_interval(-0.7))
    h2_w2.add_content(point_interval(1.1))
    h1_b.add_content(point_interval(-0.3))
    h2_b.add_content(point_interval(0.2))
    o_w1.add_content(point_interval(1.4))
    o_w2.add_content(point_interval(-1.0))
    o_b.add_content(point_interval(0.1))

    expected_x1 = 0.35
    expected_x2 = 0.55

    h1 = sigmoid_scalar(1.2 * expected_x1 + 0.8 * expected_x2 - 0.3)
    h2 = sigmoid_scalar(-0.7 * expected_x1 + 1.1 * expected_x2 + 0.2)
    expected_y = sigmoid_scalar(1.4 * h1 - 1.0 * h2 + 0.1)

    x1.add_content(point_interval(expected_x1))
    x2.add_content(point_interval(expected_x2))
    assert_interval_contains(y.content, expected_y, "step2_forward_y")

    x1_inverse = Cell(name="step2_inverse_x1")
    x2_inverse = Cell(name="step2_inverse_x2")
    y_inverse = Cell(name="step2_inverse_y")
    two_layer_prob_network(
        [x1_inverse, x2_inverse],
        hidden_weights,
        hidden_biases,
        output_weights,
        o_b,
        y_inverse,
    )

    x1_inverse.add_content(make_interval(0.0, 1.0))
    x2_inverse.add_content(make_interval(0.5, 0.6))
    y_inverse.add_content(make_interval(expected_y - 0.02, expected_y + 0.02))

    assert_interval_contains(x1_inverse.content, expected_x1, "step2_inverse_x1")
    assert_interval_contains(x2_inverse.content, expected_x2, "step2_inverse_x2")

    print("Step 2: recursive two-layer network")
    print("  forward y:", y.content)
    print("  inverse x1:", x1_inverse.content)
    print("  inverse x2:", x2_inverse.content)


def evaluate_step_3_safe_product_zero_case():
    x = Cell(name="step3_x")
    y = Cell(name="step3_y")
    total = Cell(name="step3_total")

    safe_product(x, y, total)

    x.add_content(make_interval(-1.0, 1.0))
    y.add_content(make_interval(0.8, 1.2))
    assert_interval_contains(total.content, 0.0, "step3_total")

    total.add_content(make_interval(0.16, 0.24))
    assert_interval_contains(x.content, 0.2, "step3_x")

    print("Step 3: safe product with zero-crossing interval")
    print("  y:", y.content)
    print("  total:", total.content)
    print("  inferred x:", x.content)


def run_all_evaluations():
    evaluate_step_1_nonlinear_single_neuron()
    print()
    evaluate_step_2_two_layer_recursive_network()
    print()
    evaluate_step_3_safe_product_zero_case()


if __name__ == "__main__":
    run_all_evaluations()
