"""
Interval propagators - propagator constructors that work with intervals.

These are created from interval arithmetic functions and can be used
in propagator networks.
"""
from .cell import function_to_propagator_constructor
from .intervals import (
    add_interval,
    mul_interval_complete,
    sub_interval,
    div_interval,
    square_interval,
    sqrt_interval,
)


# Create propagator constructors from interval arithmetic functions
# TODO: Should be unecessary due to the presence of generic operators supported by merge support for intervals
adder = function_to_propagator_constructor(add_interval)
multiplier = function_to_propagator_constructor(mul_interval_complete)
subtractor = function_to_propagator_constructor(sub_interval)
divider = function_to_propagator_constructor(div_interval)
squarer = function_to_propagator_constructor(square_interval)
sqrter = function_to_propagator_constructor(sqrt_interval)
