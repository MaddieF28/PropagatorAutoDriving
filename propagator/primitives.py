from .cell import function_to_propagator_constructor, Cell
from .merge import make_generic_operator
from .nothing import nothing

def constant(value, output_cell: Cell) -> None:
    """
    Propagator that immediately sets a constant value to an output cell.
    
    Unlike propagators created by function_to_propagator_constructor, this takes
    a raw value (not a cell) as its first argument.
    
    Args:
        value: The constant value to set
        output_cell: The cell to receive the constant value
        
    Example:
        >>> two = Cell()
        >>> constant(2, two)
        >>> print(two.content)  # 2
    """
    output_cell.add_content(value)

# Define generic arithmetic operators
# ====================================

generic_add = make_generic_operator(2, '+', lambda a, b: a + b)
generic_sub = make_generic_operator(2, '-', lambda a, b: a - b)
generic_mul = make_generic_operator(2, '*', lambda a, b: a * b)
generic_div = make_generic_operator(2, '/', lambda a, b: a / b)
generic_abs = make_generic_operator(1, 'abs', lambda a: abs(a))
generic_square = make_generic_operator(1, 'square', lambda a: a * a)
generic_sqrt = make_generic_operator(1, 'sqrt', lambda a: a ** 0.5)

# Generic comparison operators
generic_eq = make_generic_operator(2, '=', lambda a, b: a == b)
generic_lt = make_generic_operator(2, '<', lambda a, b: a < b)
generic_gt = make_generic_operator(2, '>', lambda a, b: a > b)
generic_lte = make_generic_operator(2, '<=', lambda a, b: a <= b)
generic_gte = make_generic_operator(2, '>=', lambda a, b: a >= b)

# Generic boolean operators
generic_not = make_generic_operator(1, 'not', lambda a: not a)
generic_and = make_generic_operator(2, 'and', lambda a, b: a and b)
generic_or = make_generic_operator(2, 'or', lambda a, b: a or b)

# Generic switch operator for conditional handling
# Scheme: (define (switch control input) (if control input nothing))
def _switch_function(control, input_val):
    """
    Switch function: return input if control is True, else nothing.

    This is the primitive function that handles plain values.
    For Supported/TMS values, handlers are registered in supported_values.py/tms.py.
    """
    if control:
        return input_val
    else:
        return nothing

generic_switch = make_generic_operator(2, 'switch', _switch_function)

# ========================================================
# These are unidirectional - they compute output from inputs

adder = function_to_propagator_constructor(generic_add)
subtractor = function_to_propagator_constructor(generic_sub)
multiplier = function_to_propagator_constructor(generic_mul)
divider = function_to_propagator_constructor(generic_div)
absolute_value = function_to_propagator_constructor(generic_abs)
squarer = function_to_propagator_constructor(generic_square)
sqrter = function_to_propagator_constructor(generic_sqrt)

# These remain unidirectional as they don't have meaningful reverse operations
absolute_value = function_to_propagator_constructor(generic_abs)

# Comparison propagators (unidirectional)
eq = function_to_propagator_constructor(generic_eq)
lt = function_to_propagator_constructor(generic_lt)
gt = function_to_propagator_constructor(generic_gt)
lte = function_to_propagator_constructor(generic_lte)
gte = function_to_propagator_constructor(generic_gte)

# Boolean propagators (unidirectional)
inverter = function_to_propagator_constructor(generic_not)
conjoiner = function_to_propagator_constructor(generic_and)
disjoiner = function_to_propagator_constructor(generic_or)


def switch(control: Cell, input_cell: Cell, output: Cell) -> None:
    """
    Switch propagator - outputs input when control is True, else nothing.

    This is the primitive conditional operation. For Supported/TMS values,
    it properly merges supports only when the control is True.

    Scheme equivalent:
        (define (switch control input)
          (if control input nothing))
        (propagatify switch)

    The key behavior:
    - When control is True: output receives input value
    - When control is False: output receives nothing (no change)
    - For Supported values: supports are merged when control is True

    Args:
        control: Cell containing boolean predicate
        input_cell: Cell containing value to pass through when True
        output: Cell to receive the result
    """
    # generic_switch already handles Supported/TMS values via handlers
    # registered in supported_values.py/tms.py.
    switch_propagator = function_to_propagator_constructor(generic_switch)
    switch_propagator(control, input_cell, output)


def conditional(p: Cell, if_true: Cell, if_false: Cell, output: Cell) -> None:
    """
    Conditional propagator that routes one of two values to output based on predicate.

    Implements the Scheme conditional using two switch propagators:
        (define-propagator (conditional control if-true if-false output)
          (switch control if-true output)
          (switch (e:not control) if-false output))

    This approach correctly handles Supported and TMS values:
    - switch(control, input) returns input if control is True, else nothing
    - For Supported values, the supports are properly merged
    - When control is False, switch returns nothing (not Supported(nothing, ...))

    Args:
        p: Cell containing a boolean predicate value (plain, Supported, or TMS)
        if_true: Cell to use when predicate is True
        if_false: Cell to use when predicate is False
        output: Cell to receive the selected value
    """
    # switch control if-true output
    switch(p, if_true, output)

    # switch (e:not control) if-false output
    # We need to create a cell for (not control) using the inverter propagator
    not_p = Cell()
    inverter(p, not_p)
    switch(not_p, if_false, output)

# Legacy names for backward compatibility
equal_to = eq
less_than = lt
greater_than = gt

# Additional aliases for common naming variations
# (helps avoid confusion - e.g., "absoluter" vs "absolute_value")
abs_value = absolute_value
absoluter = absolute_value  # Common alternative name
neg = inverter
negate = inverter
and_gate = conjoiner
or_gate = disjoiner


# ========================================================
# These are multidirectional - they compute based on which inputs are available, and merge information automatically
"""
Constraint propagators that impose relations rather than computing outputs.

These propagators stack mutual inverses on top of each other to create
multidirectional constraints. Whichever direction has enough inputs will
do its computation, and the cells merge information automatically.

Scheme equivalent:
    (define (product x y total)
      (multiplier x y total)
      (divider total x y)
      (divider total y x))
"""
def product(x: Cell, y: Cell, total: Cell) -> None:
    """
    Impose the constraint: total = x * y
    
    Works in all directions:
    - x, y known → total = x * y
    - total, x known → y = total / x
    - total, y known → x = total / y
    
    Scheme equivalent:
        (define (product x y total)
          (multiplier x y total)
          (divider total x y)
          (divider total y x))
    """
    multiplier(x, y, total)
    divider(total, x, y)
    divider(total, y, x)


def sum_constraint(x: Cell, y: Cell, total: Cell) -> None:
    """
    Impose the constraint: total = x + y
    
    Works in all directions:
    - x, y known → total = x + y
    - total, x known → y = total - x
    - total, y known → x = total - y
    
    Scheme equivalent:
        (define (sum x y total)
          (adder x y total)
          (subtractor total x y)
          (subtractor total y x))
    """
    adder(x, y, total)
    subtractor(total, x, y)
    subtractor(total, y, x)


def difference(x: Cell, y: Cell, diff: Cell) -> None:
    """
    Impose the constraint: diff = x - y
    
    Works in all directions:
    - x, y known → diff = x - y
    - diff, y known → x = diff + y
    - diff, x known → y = x - diff
    
    Scheme equivalent:
        (define (difference x y diff)
          (subtractor x y diff)
          (adder diff y x)
          (subtractor x diff y))
    """
    subtractor(x, y, diff)
    adder(diff, y, x)
    subtractor(x, diff, y)


def quadratic(x: Cell, x_squared: Cell) -> None:
    """
    Impose the constraint: x_squared = x^2
    
    Works in both directions:
    - x known → x_squared = x^2
    - x_squared known → x = sqrt(x_squared)
    
    Scheme equivalent:
        (define (quadratic x x^2)
          (squarer x x^2)
          (sqrter x^2 x))
    """
    squarer(x, x_squared)
    sqrter(x_squared, x)


# Export constraint propagators
__all__ = ['product', 'sum_constraint', 'difference', 'quadratic']
