"""Shared recursive propagator NN building blocks.

This module centralizes reusable utilities for:
- safe bidirectional multiplication with interval priors
- affine and probabilistic neurons
- recursive layer/network construction
"""

from __future__ import annotations

import math

from propagator import (
    Cell,
    Interval,
    compound_propagator,
    constant,
    div_interval,
    function_to_propagator_constructor,
    make_generic_operator,
    make_interval,
    multiplier,
    sum_constraint,
    to_interval,
)


def point_interval(value, epsilon=1e-6):
    return make_interval(value - epsilon, value + epsilon)


def sigmoid_scalar(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def logit_scalar(p):
    return math.log(p / (1.0 - p))


def _sigmoid_value(x):
    if x is None:
        return None
    if isinstance(x, Interval):
        return make_interval(sigmoid_scalar(x.low), sigmoid_scalar(x.high))
    return sigmoid_scalar(x)


def _logit_value(p):
    if p is None:
        return None
    if isinstance(p, Interval):
        if p.low <= 0.0 or p.high >= 1.0:
            return None
        return make_interval(logit_scalar(p.low), logit_scalar(p.high))
    if p <= 0.0 or p >= 1.0:
        return None
    return logit_scalar(p)


generic_sigmoid = make_generic_operator(1, "recursive_sigmoid_bidirectional", _sigmoid_value)
generic_logit = make_generic_operator(1, "recursive_logit_bidirectional", _logit_value)
sigmoid_propagator = function_to_propagator_constructor(generic_sigmoid)
logit_propagator = function_to_propagator_constructor(generic_logit)


def _interval_contains_zero(value):
    return isinstance(value, Interval) and value.low <= 0.0 <= value.high


def _safe_divide_value(total, denominator):
    if total is None or denominator is None:
        return None
    if _interval_contains_zero(denominator):
        return None

    try:
        if isinstance(total, Interval) or isinstance(denominator, Interval):
            return div_interval(to_interval(total), to_interval(denominator))
        if denominator == 0:
            return None
        return total / denominator
    except Exception:
        return None


generic_safe_divide = make_generic_operator(2, "recursive_safe_divide", _safe_divide_value)
safe_divide_propagator = function_to_propagator_constructor(generic_safe_divide)


def safe_product(x, y, total):
    multiplier(x, y, total)
    safe_divide_propagator(total, x, y)
    safe_divide_propagator(total, y, x)


def weighted_sum_recursive(inputs, weights, output, index=0):
    if index >= len(inputs):
        constant(0.0, output)
        return

    term = Cell(name=f"term_{index}")
    rest = Cell(name=f"rest_{index}")

    safe_product(inputs[index], weights[index], term)
    weighted_sum_recursive(inputs, weights, rest, index + 1)
    sum_constraint(term, rest, output)


def affine_neuron(inputs, weights, bias, output):
    def affine_neuron_compute():
        weighted_sum = Cell(name="weighted_sum")
        weighted_sum_recursive(inputs, weights, weighted_sum)
        sum_constraint(weighted_sum, bias, output)

    compound_propagator(inputs + weights + [bias], affine_neuron_compute)


def sigmoid_constraint(pre_activation, probability_output, bidirectional=True):
    sigmoid_propagator(pre_activation, probability_output)
    if bidirectional:
        logit_propagator(probability_output, pre_activation)


def probabilistic_neuron(inputs, weights, bias, probability_output, bidirectional=True):
    def probabilistic_neuron_compute():
        pre_activation = Cell(name="pre_activation")
        affine_neuron(inputs, weights, bias, pre_activation)
        sigmoid_constraint(pre_activation, probability_output, bidirectional=bidirectional)

    compound_propagator(inputs + weights + [bias], probabilistic_neuron_compute)


def create_named_cells_recursive(prefix, count, index=0, cells=None):
    if cells is None:
        cells = []
    if index >= count:
        return cells
    cells.append(Cell(name=f"{prefix}_{index}"))
    return create_named_cells_recursive(prefix, count, index + 1, cells)


def build_layer_recursive(
    inputs,
    layer_weights,
    layer_biases,
    layer_outputs,
    bidirectional=True,
    index=0,
):
    if index >= len(layer_outputs):
        return
    probabilistic_neuron(
        inputs,
        layer_weights[index],
        layer_biases[index],
        layer_outputs[index],
        bidirectional=bidirectional,
    )
    build_layer_recursive(
        inputs,
        layer_weights,
        layer_biases,
        layer_outputs,
        bidirectional=bidirectional,
        index=index + 1,
    )


def two_layer_prob_network(
    inputs,
    hidden_weights,
    hidden_biases,
    output_weights,
    output_bias,
    output_probability,
    bidirectional=True,
):
    def two_layer_prob_network_compute():
        hidden_outputs = create_named_cells_recursive("hidden_output", len(hidden_biases))
        build_layer_recursive(
            inputs,
            hidden_weights,
            hidden_biases,
            hidden_outputs,
            bidirectional=bidirectional,
        )
        probabilistic_neuron(
            hidden_outputs,
            output_weights,
            output_bias,
            output_probability,
            bidirectional=bidirectional,
        )

    all_cells = inputs + hidden_biases + output_weights + [output_bias]
    compound_propagator(all_cells, two_layer_prob_network_compute)


def assert_interval_contains(interval_value, expected, label):
    if not isinstance(interval_value, Interval):
        raise AssertionError(f"{label} is not an interval: {interval_value}")
    if not (interval_value.low <= expected <= interval_value.high):
        raise AssertionError(
            f"{label}={interval_value} does not contain expected value {expected}"
        )


def assert_interval_narrowed(interval_value, max_width, label):
    if not isinstance(interval_value, Interval):
        raise AssertionError(f"{label} is not an interval: {interval_value}")
    if interval_value.high - interval_value.low > max_width:
        raise AssertionError(f"{label} not narrowed enough: {interval_value}")


def assert_interval(interval_value, label):
    if not isinstance(interval_value, Interval):
        raise AssertionError(f"{label} is not an interval: {interval_value}")


def assert_high_probability(interval_value, label, min_low=0.9):
    assert_interval(interval_value, label)
    if interval_value.low < min_low:
        raise AssertionError(f"{label} expected high probability, got {interval_value}")


def assert_low_probability(interval_value, label, max_high=0.1):
    assert_interval(interval_value, label)
    if interval_value.high > max_high:
        raise AssertionError(f"{label} expected low probability, got {interval_value}")
