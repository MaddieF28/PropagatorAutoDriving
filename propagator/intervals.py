"""
Interval arithmetic for propagators.
Intervals represent ranges of values [low, high] and support arithmetic operations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Interval:
    """
    Represents an interval [low, high].
    Scheme equivalent uses pairs (cons cells), but we use a class for clarity.
    """
    low: Any
    high: Any
    def __repr__(self):
        return f"Interval({self.low}, {self.high})"
    
    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return self.low == other.low and self.high == other.high


def make_interval(low, high) -> Interval:
    return Interval(low, high)


def interval_low(interval: Interval):
    """
    Get the lower bound of an interval.
    
    Scheme equivalent:
        (define (interval-low interval) (car interval))
    """
    return interval.low


def interval_high(interval: Interval):
    """
    Get the upper bound of an interval.
    
    Scheme equivalent:
        (define (interval-high interval) (cdr interval))
    """
    return interval.high


def mul_interval(x: Interval, y: Interval) -> Interval:
    """
    Multiply two intervals (simplified version).
    
    Note: This is the simple implementation that assumes positive intervals.
    For a complete implementation handling negative numbers, we'd need to
    consider all four products and take min/max.
    
    Args:
        x: First interval
        y: Second interval
        
    Returns:
        Product interval
        
    Scheme equivalent:
        (define (mul-interval x y)
          (make-interval (* (interval-low x) (interval-low y))
                         (* (interval-high x) (interval-high y))))
    """
    return make_interval(
        interval_low(x) * interval_low(y),
        interval_high(x) * interval_high(y)
    )


def mul_interval_complete(x: Interval, y: Interval) -> Interval:
    """
    Multiply two intervals (complete version handling negative bounds).
    
    Computes all four possible products and takes min/max to handle
    intervals with negative values correctly.
    """
    p1 = interval_low(x) * interval_low(y)
    p2 = interval_low(x) * interval_high(y)
    p3 = interval_high(x) * interval_low(y)
    p4 = interval_high(x) * interval_high(y)
    
    return make_interval(min(p1, p2, p3, p4), max(p1, p2, p3, p4))


def add_interval(x: Interval, y: Interval) -> Interval:
    """    
    [a, b] + [c, d] = [a + c, b + d]
    """
    return make_interval(
        interval_low(x) + interval_low(y),
        interval_high(x) + interval_high(y)
    )


def sub_interval(x: Interval, y: Interval) -> Interval:
    """    
    [a, b] - [c, d] = [a - d, b - c]
    """
    return make_interval(
        interval_low(x) - interval_high(y),
        interval_high(x) - interval_low(y)
    )


def div_interval(x: Interval, y: Interval) -> Interval:
    zero = _zero_for(interval_low(y))
    if interval_low(y) <= zero <= interval_high(y):
        raise ValueError("Cannot divide by interval containing zero")  
    one = _one_for(interval_low(y))
    return mul_interval_complete(
        x,
        make_interval(one / interval_high(y), one / interval_low(y))
    )

def square_interval(x: Interval) -> Interval:
    """
    Square an interval [a, b].
    
    If both a and b are non-negative, then [a^2, b^2].
    If both a and b are non-positive, then [b^2, a^2].
    If a < 0 < b, then [0, max(a^2, b^2)].
    """
    if interval_low(x) >= 0:
        return make_interval(interval_low(x)**2, interval_high(x)**2)
    elif interval_high(x) <= 0:
        return make_interval(interval_high(x)**2, interval_low(x)**2)
    else:
        return make_interval(_zero_for(interval_low(x)), max(interval_low(x)**2, interval_high(x)**2))

def sqrt_interval(x: Interval) -> Interval:
    zero = _zero_for(interval_low(x))
    if interval_low(x) < zero:
        raise ValueError("Cannot take square root of interval containing negative values")
    return make_interval(_sqrt_value(interval_low(x)), _sqrt_value(interval_high(x)))


def empty_interval() -> Interval:
    """
    Create an empty interval.
    
    An empty interval is represented by having low > high.
    """
    return Interval(1, 0)  # low > high indicates empty interval


def empty_interval_p(x: Interval) -> bool:
    """
    Check if an interval is empty.
    
    An interval is empty if its low bound is greater than its high bound.
    
    Scheme equivalent:
        (define (empty-interval? x)
          (> (interval-low x) (interval-high x)))
    
    Note: Using '_p' suffix for predicates (Python convention)
    instead of '?' which isn't valid in Python identifiers.
    """
    return interval_low(x) > interval_high(x)


def intersect_intervals(x: Interval, y: Interval) -> Interval:
    """
    Manual implementation of interval intersection. Replaced by merge handler in merge.py.
    Intersect two intervals.
    
    The intersection of [a, b] and [c, d] is [max(a, c), min(b, d)].
    If max(a, c) > min(b, d), then the intervals do not overlap and we return an empty interval.
    """
    new_low = max(interval_low(x), interval_low(y))
    new_high = min(interval_high(x), interval_high(y))
    
    if new_low > new_high:
        return empty_interval()  # No overlap
    else:
        return make_interval(new_low, new_high)


def ensure_inside(interval: Interval, number):
    """
    Check if a number is within an interval's bounds.
    
    If the number is within [low, high], returns the number.
    Otherwise, returns the_contradiction.
    
    Scheme equivalent:
        (define (ensure-inside interval number)
          (if (<= (interval-low interval) number (interval-high interval))
              number
              the-contradiction))
    """
    from .merge import the_contradiction
    
    if interval_low(interval) <= number <= interval_high(interval):
        return number
    else:
        return the_contradiction


# Register interval-specific merge handler
# =========================================
# This allows intervals to use custom intersection-based merging

def _merge_intervals(content: Interval, increment: Interval):
    """
    Merge two intervals using intersection.
    
    Scheme equivalent:
        (assign-operation 'merge
          (lambda (content increment)
            (let ((new-range (intersect-intervals content increment)))
              (cond ((interval-equal? new-range content) content)
                    ((interval-equal? new-range increment) increment)
                    ((empty-interval? new-range) the-contradiction)
                    (else new-range))))
          interval? interval?)
    """
    from .merge import the_contradiction
    
    new_range = intersect_intervals(content, increment)
    
    # Check if the range is the same as content
    if new_range == content:
        return content
    # Check if the range is the same as increment
    elif new_range == increment:
        return increment
    # Check if the range is empty (contradiction)
    elif empty_interval_p(new_range):
        return the_contradiction
    # Otherwise, return the new refined range
    else:
        return new_range


def _is_both_intervals(content) -> bool:
    """Predicate to check if value is an Interval."""
    return isinstance(content, Interval)


def _is_number(content) -> bool:
    """
    Predicate to check if a value can participate in interval arithmetic.

    This is intentionally structural rather than int/float-only so custom
    scalar types (for example Decimal, Fraction, or project-specific numeric
    wrappers) can opt in by providing the expected arithmetic/comparison ops.
    """
    if isinstance(content, Interval):
        return False

    required_ops = (
        "__add__", "__sub__", "__mul__", "__truediv__",
        "__lt__", "__le__", "__gt__", "__ge__",
    )
    return all(hasattr(content, op) for op in required_ops)


def _merge_number_with_interval(content, increment: Interval):
    """
    Merge a number with an interval by ensuring the number is inside the interval.
    
    Scheme equivalent:
        (assign-operation 'merge
          (lambda (content increment)
            (ensure-inside increment content))
          number? interval?)
    """
    return ensure_inside(increment, content)


def _merge_interval_with_number(content: Interval, increment):
    """
    Merge an interval with a number by ensuring the number is inside the interval.
    
    Scheme equivalent:
        (assign-operation 'merge
          (lambda (content increment)
            (ensure-inside content increment))
          interval? number?)
    """
    return ensure_inside(content, increment)


# Coercion utilities for mixed number/interval arithmetic
# ========================================================

def to_interval(x):
    """
    Convert a value to an interval.
    
    If x is already an interval, return it unchanged.
    If x is a number, convert it to an interval [x, x].
    
    Scheme equivalent:
        (define (->interval x)
          (if (interval? x) x
              (make-interval x x)))
    """
    if isinstance(x, Interval):
        return x
    if _is_number(x):
        return make_interval(x, x)
    raise TypeError(f"Cannot coerce value of type {type(x).__name__} to interval")


def _coerce_like(value, literal):
    """Best-effort conversion of a literal (0 or 1) to value's scalar type."""
    try:
        return type(value)(literal)
    except Exception:
        return literal


def _zero_for(value):
    """Return a zero literal compatible with the scalar domain of value."""
    return _coerce_like(value, 0)


def _one_for(value):
    """Return a one literal compatible with the scalar domain of value."""
    return _coerce_like(value, 1)


def _sqrt_value(value):
    """Compute scalar sqrt while preserving non-float domains when possible."""
    sqrt_method = getattr(value, "sqrt", None)
    if callable(sqrt_method):
        return sqrt_method()
    return value ** 0.5


def coercing(coercer, f):
    """
    Create a function that coerces its arguments before applying f.
    
    Scheme equivalent:
        (define (coercing coercer f)
          (lambda args
            (apply f (map coercer args))))
    
    Args:
        coercer: Function to apply to each argument
        f: Function to call with coerced arguments
        
    Returns:
        A new function that coerces arguments then calls f
    """
    def wrapper(*args):
        coerced_args = [coercer(arg) for arg in args]
        return f(*coerced_args)
    return wrapper


# Register the interval merge handlers using the unified generic operator system
from .merge import assign_merge_operation
assign_merge_operation(_merge_intervals, _is_both_intervals, _is_both_intervals)
assign_merge_operation(_merge_number_with_interval, _is_number, _is_both_intervals)
assign_merge_operation(_merge_interval_with_number, _is_both_intervals, _is_number)


# Register interval operations with generic arithmetic operators
# ===============================================================
# Scheme equivalent:
#   (assign-operation '* mul-interval interval? interval?)
#   (assign-operation '/ div-interval interval? interval?)
#   (assign-operation 'square square-interval interval?)
#   (assign-operation 'sqrt sqrt-interval interval?)

from .primitives import (
    generic_add, generic_sub, generic_mul, generic_div,
    generic_square, generic_sqrt
)

# (assign-operation '+ add-interval interval? interval?)
generic_add.assign_operation(add_interval, _is_both_intervals, _is_both_intervals)
# Mixed number/interval addition
generic_add.assign_operation(coercing(to_interval, add_interval), _is_number, _is_both_intervals)
generic_add.assign_operation(coercing(to_interval, add_interval), _is_both_intervals, _is_number)

# (assign-operation '- sub-interval interval? interval?)
generic_sub.assign_operation(sub_interval, _is_both_intervals, _is_both_intervals)
# Mixed number/interval subtraction
generic_sub.assign_operation(coercing(to_interval, sub_interval), _is_number, _is_both_intervals)
generic_sub.assign_operation(coercing(to_interval, sub_interval), _is_both_intervals, _is_number)

# (assign-operation '* mul-interval interval? interval?)
generic_mul.assign_operation(mul_interval_complete, _is_both_intervals, _is_both_intervals)
# (assign-operation '* (coercing ->interval mul-interval) number? interval?)
generic_mul.assign_operation(coercing(to_interval, mul_interval_complete), _is_number, _is_both_intervals)
# (assign-operation '* (coercing ->interval mul-interval) interval? number?)
generic_mul.assign_operation(coercing(to_interval, mul_interval_complete), _is_both_intervals, _is_number)

# (assign-operation '/ div-interval interval? interval?)
generic_div.assign_operation(div_interval, _is_both_intervals, _is_both_intervals)
# (assign-operation '/ (coercing ->interval div-interval) number? interval?)
generic_div.assign_operation(coercing(to_interval, div_interval), _is_number, _is_both_intervals)
# (assign-operation '/ (coercing ->interval div-interval) interval? number?)
generic_div.assign_operation(coercing(to_interval, div_interval), _is_both_intervals, _is_number)

# (assign-operation 'square square-interval interval?)
generic_square.assign_operation(square_interval, _is_both_intervals)

# (assign-operation 'sqrt sqrt-interval interval?)
generic_sqrt.assign_operation(sqrt_interval, _is_both_intervals)