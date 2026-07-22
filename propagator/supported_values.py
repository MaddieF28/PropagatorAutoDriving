"""
Dependency (support) utilities for supported values.

This module implements the Scheme procedures:
  - supported (v&s) structure
  - more-informative-support?
  - merge-supports

PERFORMANCE NOTE:
    Supports are stored as frozenset for O(1) membership testing instead of
    O(n) list scanning. Since Scheme uses identity-based equality (eq?) for
    premises, we wrap premise objects in an IdentityWrapper that uses id()
    for hashing and identity comparison for equality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, List, Set

from .intervals import Interval
from .merge import assign_merge_operation, generic_merge, merge, contradictory, any_p
from .nothing import nothing, nothing_p
from .primitives import (
	generic_abs,
	generic_add,
	generic_and,
	generic_div,
	generic_eq,
	generic_gt,
	generic_gte,
	generic_lt,
	generic_lte,
	generic_mul,
	generic_not,
	generic_or,
	generic_sqrt,
	generic_square,
	generic_sub,
)


# =============================================================================
# Identity-based Support Sets
# =============================================================================
# Scheme uses identity equality (eq?) for premises in support sets.
# Python's frozenset uses == for membership, so we need a wrapper.

class IdentityWrapper:
    """
    Wrapper that provides identity-based hashing and equality.
    
    This allows using object identity (like Scheme's eq?) with Python's
    set/frozenset which normally use == for membership testing.
    
    The wrapper is transparent: use .obj to get the original object.
    """
    __slots__ = ('obj', '_hash')
    
    def __init__(self, obj: Any):
        self.obj = obj
        self._hash = id(obj)
    
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, IdentityWrapper):
            return self.obj is other.obj
        return False
    
    def __repr__(self) -> str:
        return f"IdWrap({self.obj!r})"


# Type alias for clarity
Support = FrozenSet[IdentityWrapper]


def _make_support(items: Iterable[Any]) -> Support:
    """
    Create a support set from an iterable of premises.
    
    Wraps each premise in IdentityWrapper for O(1) identity-based membership.
    """
    return frozenset(IdentityWrapper(item) for item in items)


def _unwrap_support(support: Support) -> List[Any]:
    """
    Unwrap a support set back to a list of premises.
    
    Use this when you need to iterate over premises or return them to
    code that expects a list.
    """
    return [w.obj for w in support]


def _support_contains(support: Support, item: Any) -> bool:
    """
    Identity-based membership check for support sets. O(1).
    """
    return IdentityWrapper(item) in support


def support_contains(vs_or_support: Any, premise: Any) -> bool:
    """
    Check if a premise is in a Supported value's support set.
    
    This is the public API for checking premise membership. It handles:
    - Supported values (extracts .support automatically)
    - Support frozensets directly
    - Uses identity-based comparison (like Scheme's eq?)
    
    Args:
        vs_or_support: Either a Supported value or a Support frozenset
        premise: The premise to check for
    
    Returns:
        True if premise is in the support, False otherwise
    
    Example:
        >>> p = hypothetical()
        >>> vs = supported(42, [p])
        >>> support_contains(vs, p)  # True
        >>> support_contains(vs.support, p)  # Also True
    """
    if isinstance(vs_or_support, Supported):
        return IdentityWrapper(premise) in vs_or_support.support
    elif isinstance(vs_or_support, frozenset):
        return IdentityWrapper(premise) in vs_or_support
    else:
        # Fall back to list-style membership for backward compatibility
        return _contains_eq(vs_or_support, premise)


def _support_issubset(a: Support, b: Support) -> bool:
    """
    Check if support a is a subset of support b. O(|a|).
    """
    return a.issubset(b)


def _support_union(*supports: Support) -> Support:
    """
    Compute the union of multiple support sets. O(sum of sizes).
    """
    if not supports:
        return frozenset()
    result = supports[0]
    for s in supports[1:]:
        result = result | s
    return result


# Legacy functions for backward compatibility (used in tms.py)
def _contains_eq(items: Iterable[Any], target: Any) -> bool:
	"""
	Identity-based membership check (Scheme eq?).
	
	DEPRECATED: Use _support_contains with Support type for O(1) lookup.
	This O(n) version is kept for backward compatibility with code that
	hasn't migrated to frozenset-based supports.
	"""
	return any(item is target for item in items)


def _lset_leq_eq(a: Iterable[Any], b: Iterable[Any]) -> bool:
	"""
	Identity-based subset check (Scheme lset<= with eq?).
	
	DEPRECATED: Use _support_issubset with Support type for better performance.
	"""
	b_list = list(b)
	return all(_contains_eq(b_list, item) for item in a)


def _lset_equal_eq(a: Iterable[Any], b: Iterable[Any]) -> bool:
	"""
	Identity-based set equality (Scheme lset= with eq?).
	
	DEPRECATED: Use == on Support (frozenset) for O(n) equality.
	"""
	a_list = list(a)
	b_list = list(b)
	return _lset_leq_eq(a_list, b_list) and _lset_leq_eq(b_list, a_list)


def _lset_union_eq(*lists: Iterable[Any]) -> List[Any]:
	"""
	Identity-based set union, preserving first-seen order.
	
	DEPRECATED: Use _support_union with Support type for O(n) union.
	This O(n²) version is kept for backward compatibility.
	"""
	result: List[Any] = []
	for lst in lists:
		for item in lst:
			if not _contains_eq(result, item):
				result.append(item)
	return result


@dataclass(frozen=True)
class Supported:
	"""
	Represents a supported value with its dependency support.
	
	The support is a frozenset of IdentityWrapper objects for O(1)
	membership testing (vs O(n) with lists).

	Scheme equivalent:
		(define-structure (v&s (named 'supported) (type vector)
								(constructor supported))
		  value support)
	"""
	value: Any
	support: Support  # frozenset[IdentityWrapper] for O(1) membership

	@staticmethod
	def _premise_label(premise: Any) -> str:
		"""Format a premise for human-readable output."""
		if hasattr(premise, 'describe') and callable(getattr(premise, 'describe')):
			try:
				return str(premise.describe())
			except Exception:
				pass
		if hasattr(premise, 'name') and getattr(premise, 'name'):
			return str(getattr(premise, 'name'))
		return str(premise)

	def _sorted_premise_labels(self) -> List[str]:
		labels = [self._premise_label(wrapper.obj) for wrapper in self.support]
		return sorted(labels)

	def __str__(self) -> str:
		premises = self._sorted_premise_labels()
		support_text = 'none' if not premises else ', '.join(premises)
		return f"Supported(value={self.value}, support=[{support_text}])"

	def __repr__(self) -> str:
		# Keep repr aligned with str() for cleaner interactive/debug printing.
		return self.__str__()


def supported(value: Any, support: Iterable[Any]) -> Supported:
	"""
	Constructor for Supported values.
	
	Args:
		value: The actual value
		support: Iterable of premises (will be wrapped for identity-based lookup)
	
	Note: If support is already a Support (frozenset[IdentityWrapper]), it will
	be used directly. Otherwise, each item is wrapped in IdentityWrapper.
	"""
	if isinstance(support, frozenset) and (not support or isinstance(next(iter(support)), IdentityWrapper)):
		# Already a proper Support frozenset
		return Supported(value=value, support=support)
	return Supported(value=value, support=_make_support(support))


def supported_p(value: Any) -> bool:
	"""Predicate for Supported values."""
	return isinstance(value, Supported)


def get_support_premises(vs: Supported) -> List[Any]:
	"""
	Get the list of premises from a Supported value's support.
	
	This unwraps the IdentityWrapper objects to return the original premises.
	Use this when you need to iterate over or display premises.
	"""
	return _unwrap_support(vs.support)


def more_informative_support(vs1: Supported, vs2: Supported) -> bool:
	"""
	Return True if vs1 has strictly more informative support than vs2.
	
	More informative means: vs1's support is a strict subset of vs2's support.
	(Fewer premises = more informative because it's a stronger conclusion.)

	Scheme equivalent:
		(and (not (lset= eq? (v&s-support v&s1) (v&s-support v&s2)))
			 (lset<= eq? (v&s-support v&s1) (v&s-support v&s2)))
	
	Performance: O(|vs1.support|) with frozenset vs O(n²) with lists.
	"""
	return (
		vs1.support != vs2.support  # O(n) frozenset comparison
		and vs1.support.issubset(vs2.support)  # O(|vs1.support|)
	)


def merge_supports(*vss: Supported) -> Support:
	"""
	Merge supports using identity-based union.
	
	Returns a Support (frozenset[IdentityWrapper]) for O(1) membership testing.

	Scheme equivalent:
		(apply lset-union eq? (map v&s-support v&ss))
	
	Performance: O(total premises) with frozenset vs O(n²) with lists.
	"""
	return _support_union(*(vs.support for vs in vss))


def flat_p(thing: Any) -> bool:
	"""
	Predicate for flat (non-supported) values.

	A "flat" value is any value that is NOT a Supported or TMS structure.
	This includes numbers, booleans, strings, intervals, and other basic types.

	Scheme equivalent:
		(define (flat? thing) (or (interval? thing) (number? thing) (boolean? thing)))
	
	Note: In Python, we extend this to include strings and other basic types
	that might be used in propagator networks.
	"""
	# Avoid circular import by checking class name
	if isinstance(thing, Supported):
		return False
	# Check for TMS by class name to avoid import
	if thing.__class__.__name__ == 'Tms':
		return False
	# Include common basic types
	return isinstance(thing, (Interval, int, float, bool, str, type(None)))


def to_supported(thing: Any) -> Supported:
	"""
	Coerce a value to Supported.

	Scheme equivalent:
		(define (->v&s thing) (if (v&s? thing) thing (supported thing '())))
	"""
	return thing if supported_p(thing) else supported(thing, [])


def coercing(coercer: Callable[[Any], Any], f: Callable) -> Callable:
	"""
	Apply a coercer to each argument before calling f.
	"""
	def wrapper(*args):
		return f(*[coercer(arg) for arg in args])
	return wrapper


def supported_unpacking(f: Callable) -> Callable:
	"""
	Lift a function to operate on Supported values.

	Scheme equivalent:
		(define (v&s-unpacking f)
		  (lambda args
		    (supported (apply f (map v&s-value args))
		              (apply merge-supports args))))
	"""
	def wrapper(*args: Supported):
		return supported(
			f(*[arg.value for arg in args]),
			merge_supports(*args),
		)
	return wrapper


def implies(v1: Any, v2: Any) -> bool:
	"""
	Return True if v1 implies v2, using merge semantics.

	Scheme equivalent:
		(define (implies? v1 v2)
		  (eq? v1 (merge v1 v2)))
	
	Optimization: For simple types (bool, int, float, str), we can check
	equality directly instead of going through the generic merge machinery.
	This is a significant performance win since 99%+ of implies() calls
	in typical TMS workloads are comparing booleans or integers.
	"""
	# Fast path for common simple types - avoid GenericOperator dispatch
	# For these types, implies(v1, v2) iff v1 == v2 (since merge returns v1 if equal)
	t1, t2 = type(v1), type(v2)
	if t1 is t2 and t1 in (bool, int, float, str):
		return v1 == v2
	
	# Also handle nothing - nothing implies everything
	if nothing_p(v1):
		return True
	if nothing_p(v2):
		return True  # merge(v1, nothing) returns v1
	
	# Fall back to full merge for complex types (Supported, Interval, TMS, etc.)
	return merge(v1, v2) is v1


def supported_merge(vs1: Supported, vs2: Supported) -> Supported:
	"""
	Merge two Supported values.

	Scheme equivalent:
		(define (v&s-merge v&s1 v&s2) ...)
	"""
	vs1_value = vs1.value
	vs2_value = vs2.value
	value_merge = merge(vs1_value, vs2_value)

	if value_merge is vs1_value:
		if implies(vs2_value, value_merge):
			return vs2 if more_informative_support(vs2, vs1) else vs1
		return vs1
	if value_merge is vs2_value:
		return vs2
	return supported(value_merge, merge_supports(vs1, vs2))


# Register merge and contradictory? handlers for Supported values
assign_merge_operation(supported_merge, supported_p, supported_p)
contradictory.assign_operation(lambda vs: contradictory(vs.value), supported_p)


# Register generic operator support for Supported values
_binary_ops = [
	(generic_add, '+'),
	(generic_sub, '-'),
	(generic_mul, '*'),
	(generic_div, '/'),
	(generic_eq, '='),
	(generic_lt, '<'),
	(generic_gt, '>'),
	(generic_lte, '<='),
	(generic_gte, '>='),
	(generic_and, 'and'),
	(generic_or, 'or'),
]

for op, _ in _binary_ops:
	op.assign_operation(supported_unpacking(op), supported_p, supported_p)
	op.assign_operation(coercing(to_supported, op), supported_p, flat_p)
	op.assign_operation(coercing(to_supported, op), flat_p, supported_p)

_unary_ops = [
	(generic_abs, 'abs'),
	(generic_square, 'square'),
	(generic_sqrt, 'sqrt'),
	(generic_not, 'not'),
]

for op, _ in _unary_ops:
	op.assign_operation(supported_unpacking(op), supported_p)

# Merge with mixed supported/flat values
# Note: Use generic_merge (the GenericOperator) for assigning operations,
# not merge (which is wrapped with equivalent? short-circuit)
generic_merge.assign_operation(coercing(to_supported, supported_merge), supported_p, flat_p)
generic_merge.assign_operation(coercing(to_supported, supported_merge), flat_p, supported_p)


# Register generic_switch for Supported values
# The switch function: (if control input nothing)
# CRITICAL: Returns the nothing sentinel if control is False, NOT Supported(nothing, ...)
from .primitives import generic_switch

def supported_switch(control: Supported, input_val: Supported) -> Any:
	"""
	Switch for Supported values.

	Scheme equivalent (from v&s-binary-map):
		(v&s-> (supported (f (v&s-value v&s1) (v&s-value v&s2))
		                 (merge-supports v&s1 v&s2)))

	Where v&s-> converts Supported(nothing, ...) to nothing.
	"""
	# Extract values and apply switch logic
	if control.value:
		# Control is True: return input with merged supports
		return supported(input_val.value, merge_supports(control, input_val))
	else:
		# Control is False: return nothing (NOT Supported(nothing, ...))
		return nothing

# Register for all combinations of Supported/flat values
generic_switch.assign_operation(supported_switch, supported_p, supported_p)
generic_switch.assign_operation(
	lambda c, i: supported_switch(to_supported(c), to_supported(i)),
	supported_p, flat_p
)
generic_switch.assign_operation(
	lambda c, i: supported_switch(to_supported(c), to_supported(i)),
	flat_p, supported_p
)
generic_switch.assign_operation(
	lambda c, i: supported_switch(to_supported(c), to_supported(i)),
	supported_p, any_p
)
generic_switch.assign_operation(
	lambda c, i: supported_switch(to_supported(c), to_supported(i)),
	any_p, supported_p
)

