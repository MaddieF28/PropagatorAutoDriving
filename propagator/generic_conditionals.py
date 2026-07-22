"""
Generic conditional operations for proper handling of supported values.

This module implements the Scheme procedures:
  - generic-true?  : Check truthiness handling supported values correctly
  - generic-ignore-first : Combine support from predicate with branch result

The conditional propagator is nontrivial because, when given a supported value,
it must both:
1. Branch correctly even if given a supported #f (which would read as a true 
   value if naively passed to Python's native `if`, since Supported(False, [...])
   is a truthy object)
2. Attach the support of the predicate to the support of the result produced 
   by that branch

We accomplish this by introducing two additional generic operations and adding
appropriate methods to them.

Scheme reference:
    (define true? (lambda (x) (not (not x))))
    
    (define generic-true? (make-generic-operator 1 'true? true?))
    (assign-operation 'true? (lambda (v&s) (generic-true? (v&s-value v&s))) v&s?)
    (assign-operation 'true? (lambda (tms) (generic-true? (tms-query tms))) tms?)
    
    (define generic-ignore-first (make-generic-operator 2 'ignore-first ignore-first))
"""

from __future__ import annotations

from typing import Any


def true_p(x: Any) -> bool:
    """
    Default truth check - like Scheme's (not (not x)).
    
    Scheme equivalent:
        (define true? (lambda (x) (not (not x))))
    """
    return not (not x)


def ignore_first(x: Any, y: Any) -> Any:
    """
    Return the second argument, ignoring the first.
    
    This is the default operation for generic_ignore_first.
    For supported values, the support from x must be merged into the result.
    
    Scheme equivalent:
        (define (ignore-first x y) y)
    """
    return y


# Late-binding initialization to avoid circular imports
generic_true = None
generic_ignore_first = None
_initialized = False


def _init_generic_conditionals():
    """
    Initialize the generic operators.
    Called after all dependent modules are loaded.
    """
    global generic_true, generic_ignore_first, _initialized
    
    if _initialized:
        return
    
    from .merge import make_generic_operator
    from .supported_values import (
        Supported,
        supported,
        supported_p,
        flat_p,
        merge_supports,
        to_supported,
        coercing,
    )
    from .nothing import nothing_p
    from .tms import tms_p, tms_query, full_tms_unpacking
    
    # Create generic_true operator
    # Scheme: (define generic-true? (make-generic-operator 1 'true? true?))
    generic_true = make_generic_operator(1, 'true?', true_p)
    
    # For Supported values: unwrap and recursively check the inner value
    # Scheme: (assign-operation 'true? (lambda (v&s) (generic-true? (v&s-value v&s))) v&s?)
    def supported_true(vs: Supported) -> bool:
        return generic_true(vs.value)
    
    generic_true.assign_operation(supported_true, supported_p)
    
    # For TMS: query the TMS first, then check truthiness
    # Scheme: (assign-operation 'true? (lambda (tms) (generic-true? (tms-query tms))) tms?)
    def tms_true(tms) -> bool:
        queried = tms_query(tms)
        if nothing_p(queried):
            return False  # Nothing is not true
        return generic_true(queried)
    
    generic_true.assign_operation(tms_true, tms_p)
    
    # Create generic_ignore_first operator
    # Scheme: (define generic-ignore-first (make-generic-operator 2 'ignore-first ignore-first))
    generic_ignore_first = make_generic_operator(2, 'ignore-first', ignore_first)
    
    # For supported values, we need to merge the support from the predicate (first)
    # with the support of the value (second)
    # Scheme equivalent of v&s-unpacking for ignore-first:
    #   (lambda (pred val)
    #     (supported (v&s-value val)
    #               (merge-supports pred val)))
    
    def supported_ignore_first(pred: Supported, val: Supported) -> Supported:
        """
        Combine support from predicate with value.
        The result has the value from val, but support from both pred and val.
        """
        return supported(val.value, merge_supports(pred, val))
    
    # Register all combinations for supported values
    # Scheme: (assign-operation name (v&s-unpacking underlying-operation) v&s? v&s?)
    generic_ignore_first.assign_operation(supported_ignore_first, supported_p, supported_p)
    
    # Scheme: (assign-operation name (coercing ->v&s underlying-operation) v&s? flat?)
    generic_ignore_first.assign_operation(
        coercing(to_supported, supported_ignore_first),
        supported_p,
        flat_p,
    )
    
    # Scheme: (assign-operation name (coercing ->v&s underlying-operation) flat? v&s?)
    generic_ignore_first.assign_operation(
        coercing(to_supported, supported_ignore_first),
        flat_p,
        supported_p,
    )
    
    # For TMS values, use full_tms_unpacking
    # Scheme: (assign-operation name (full-tms-unpacking underlying-operation) tms? tms?)
    generic_ignore_first.assign_operation(full_tms_unpacking(ignore_first), tms_p, tms_p)
    
    # Mixed TMS/Supported combinations
    # Scheme: (assign-operation name (coercing ->tms underlying-operation) tms? v&s?)
    from .tms import to_tms
    
    def tms_ignore_first_unpacked(pred, val):
        """Use full_tms_unpacking on coerced TMS values."""
        return full_tms_unpacking(ignore_first)(pred, val)
    
    generic_ignore_first.assign_operation(
        coercing(to_tms, tms_ignore_first_unpacked),
        tms_p,
        supported_p,
    )
    
    # Scheme: (assign-operation name (coercing ->tms underlying-operation) v&s? tms?)
    generic_ignore_first.assign_operation(
        coercing(to_tms, tms_ignore_first_unpacked),
        supported_p,
        tms_p,
    )
    
    # Scheme: (assign-operation name (coercing ->tms underlying-operation) tms? flat?)
    generic_ignore_first.assign_operation(
        coercing(to_tms, tms_ignore_first_unpacked),
        tms_p,
        flat_p,
    )
    
    # Scheme: (assign-operation name (coercing ->tms underlying-operation) flat? tms?)
    generic_ignore_first.assign_operation(
        coercing(to_tms, tms_ignore_first_unpacked),
        flat_p,
        tms_p,
    )
    
    _initialized = True


def get_generic_true():
    """Get the generic_true operator, initializing if needed."""
    if not _initialized:
        _init_generic_conditionals()
    return generic_true


def get_generic_ignore_first():
    """Get the generic_ignore_first operator, initializing if needed."""
    if not _initialized:
        _init_generic_conditionals()
    return generic_ignore_first
