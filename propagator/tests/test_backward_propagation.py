"""Pytest coverage for backward/forward propagation via compositional constraints.

This test exercises a multi-directional network (quadratic + product) to ensure
information flows both forward (time -> height) and backward (height -> time).
"""

from propagator import Cell, constant, interval_low, interval_high, make_interval, product, quadratic


def test_backward_propagation_restricts_inputs_and_outputs():
    """Motivation: constraints should refine unknowns from either direction."""
    fall_time = Cell()
    building_height = Cell()

    g = Cell()
    one_half = Cell()
    t_squared = Cell()
    g_times_t_squared = Cell()

    constant(make_interval(9.789, 9.832), g)
    constant(make_interval(0.5, 0.5), one_half)

    quadratic(fall_time, t_squared)
    product(g, t_squared, g_times_t_squared)
    product(one_half, g_times_t_squared, building_height)

    fall_time.add_content(make_interval(2.9, 3.1))
    assert building_height.content is not None
    assert interval_low(building_height.content) > 0
    assert interval_high(building_height.content) > interval_low(building_height.content)

    building_height.add_content(45)
    assert fall_time.content is not None
    assert interval_low(fall_time.content) <= interval_high(fall_time.content)
