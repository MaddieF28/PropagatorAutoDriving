"""
Merge operations for propagator values.

Handles merging of information from different sources, detecting contradictions.
Supports extensible merge strategies for different data types via a registry.
"""

# Import nothing from dedicated module (no circular dependency risk)
from .nothing import nothing, nothing_p, _Nothing


# ============================================================================
# The Contradiction Singleton
# ============================================================================
# In Scheme: (define the-contradiction (list 'contradiction))
# A unique object representing a contradiction/conflict in information

class _Contradiction:
    """
    Singleton class representing a contradiction.
    
    This is used when two pieces of information cannot be merged
    (e.g., a cell receives conflicting values).
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return "Contradiction"
    
    def __eq__(self, other):
        # Only equal to itself (identity comparison)
        return self is other


# The single instance of contradiction
the_contradiction = _Contradiction()


def contradictory_p(x) -> bool:
    """
    Check if a value is a contradiction.
    
    Args:
        x: The value to check
        
    Returns:
        True if x is a contradiction, False otherwise
        
    Scheme equivalent:
        (define (contradictory? x)
          (eq? x the-contradiction))
    
    Note: Using '_p' suffix for predicates (Python convention)
    instead of '?' which isn't valid in Python identifiers.
    """
    return contradictory(x)


# Generic Merge System
# ====================
# Merge as a GenericOperator for consistency with Scheme

# Generic Operator System
# ========================
# Generic operators dispatch based on argument types, similar to Scheme's make-generic-operator

class GenericOperator:
    """
    A generic operator that dispatches to type-specific implementations.
    
    Scheme equivalent:
        (define generic-+ (make-generic-operator 2 '+ +))
    
    The operator maintains a list of handlers, each with a predicate and implementation.
    When called, it tries each handler until one matches the argument types.
    
    Handler Coverage:
        Use get_handler_coverage() to inspect which type combinations are covered.
        Use check_handler_coverage() to verify that expected type combinations have handlers.
        Missing handlers will silently fall through to default_op, which may cause
        subtle bugs - use these methods to detect such issues in tests.
    """
    def __init__(self, arity: int, name: str, default_op):
        """
        Args:
            arity: Number of arguments the operator takes
            name: Name of the operator (for debugging)
            default_op: Default operation for base types (numbers)
        """
        self.arity = arity
        self.name = name
        self.default_op = default_op
        self._handlers = []
        # Track which handlers were actually used (for coverage analysis)
        self._handler_hit_counts = []
    
    def assign_operation(self, operation, *type_predicates):
        """
        Register a type-specific handler.
        
        Args:
            operation: The function to call for this type combination
            type_predicates: Predicates to check argument types (one per argument)
        """
        if len(type_predicates) != self.arity:
            raise ValueError(f"Expected {self.arity} predicates, got {len(type_predicates)}")
        self._handlers.append((type_predicates, operation))
        self._handler_hit_counts.append(0)
    
    def __call__(self, *args):
        """
        Invoke the generic operator with the given arguments.
        
        Tries each registered handler in order. If none match, uses default operation.
        """
        if len(args) != self.arity:
            raise ValueError(f"{self.name} expects {self.arity} arguments, got {len(args)}")
        
        # Try each handler
        for i, (predicates, operation) in enumerate(self._handlers):
            if all(pred(arg) for pred, arg in zip(predicates, args)):
                self._handler_hit_counts[i] += 1
                return operation(*args)
        
        # Fall back to default operation
        return self.default_op(*args)
    
    def get_handler_coverage(self) -> list:
        """
        Get coverage information for all registered handlers.
        
        Returns:
            List of dicts with 'predicates', 'operation', 'hit_count' for each handler.
            Handlers with hit_count=0 were never invoked (potential dead code or
            missing test coverage).
        """
        return [
            {
                'predicates': [p.__name__ if hasattr(p, '__name__') else str(p) for p in preds],
                'operation': op.__name__ if hasattr(op, '__name__') else str(op),
                'hit_count': count,
            }
            for (preds, op), count in zip(self._handlers, self._handler_hit_counts)
        ]
    
    def get_unused_handlers(self) -> list:
        """
        Get handlers that have never been invoked.
        
        Useful for detecting dead handler registrations or missing test coverage.
        
        Returns:
            List of (predicates, operation) tuples for handlers with hit_count=0.
        """
        return [
            (preds, op)
            for (preds, op), count in zip(self._handlers, self._handler_hit_counts)
            if count == 0
        ]
    
    def check_handler_coverage(self, expected_predicate_names: list) -> dict:
        """
        Check that expected type combinations have handlers registered.
        
        Args:
            expected_predicate_names: List of tuples of predicate names that should
                have handlers. E.g., [('supported_p', 'supported_p'), ('flat_p', 'supported_p')]
        
        Returns:
            Dict with:
                'missing': List of expected combinations without handlers
                'extra': List of registered combinations not in expected list
                'covered': List of expected combinations that have handlers
        
        Example:
            >>> result = generic_add.check_handler_coverage([
            ...     ('supported_p', 'supported_p'),
            ...     ('supported_p', 'flat_p'),
            ...     ('flat_p', 'supported_p'),
            ... ])
            >>> assert result['missing'] == [], f"Missing handlers: {result['missing']}"
        """
        registered = set()
        for preds, _ in self._handlers:
            pred_names = tuple(
                p.__name__ if hasattr(p, '__name__') else str(p)
                for p in preds
            )
            registered.add(pred_names)
        
        expected_set = set(tuple(e) for e in expected_predicate_names)
        
        return {
            'missing': list(expected_set - registered),
            'extra': list(registered - expected_set),
            'covered': list(expected_set & registered),
        }
    
    def reset_coverage(self) -> None:
        """
        Reset all handler hit counts to zero.
        
        Call this at the start of a test to measure coverage for that test.
        """
        self._handler_hit_counts = [0] * len(self._handlers)


def make_generic_operator(arity: int, name: str, default_op):
    """
    Create a generic operator.
    
    Scheme equivalent:
        (define (make-generic-operator arity name default-op) ...)
    """
    return GenericOperator(arity, name, default_op)


# Generic contradictory? operator
contradictory = make_generic_operator(
    1,
    'contradictory?',
    lambda thing: thing is the_contradiction,
)


# ============================================================================
# Generic Equivalent Operator
# ============================================================================
#
# Scheme equivalent from cells.scm:
#     (define (equivalent? info1 info2)
#       (or (eqv? info1 info2)
#           (generic-equivalent? info1 info2)))
#
#     (define generic-equivalent?
#       (make-generic-operator 2 'equivalent? (lambda (a b) #f)))
#
# OPTIMIZATION RATIONALE:
# The generic_equivalent operator provides a fast short-circuit in merge:
# if two values are equivalent, we can return either one immediately without
# invoking the full merge machinery. This is especially valuable for:
#
# 1. COMPOUND TYPES: Two intervals or supported values that represent the
#    same information can be detected quickly without full merge computation.
#
# 2. REDUCING GARBAGE: Returning the existing value avoids creating a new
#    merged object when the information is already the same.
#
# 3. CONSISTENCY: Multiple sources providing equivalent info should not
#    trigger unnecessary propagation cycles.

def _default_equivalent(a, b):
    """
    Default equivalence check for unknown types.
    
    For safety with custom types, we use identity check only as the default.
    This matches Scheme's conservative approach:
        (define generic-equivalent? (make-generic-operator 2 'equivalent? (lambda (a b) #f)))
    
    Custom types should register their own equivalence handlers if they want
    merge short-circuiting to work for equivalent (but not identical) values.
    
    For built-in immutable types (int, float, str, bool), we use == since
    they are safe and commonly used in propagator networks.
    """
    # Fast path for common immutable types where == is safe and meaningful
    t1, t2 = type(a), type(b)
    if t1 is t2 and t1 in (bool, int, float, str, tuple):
        return a == b
    # For unknown types, be conservative - require identity
    # This prevents issues with types where == might not return bool
    # or might have side effects
    return False


generic_equivalent = make_generic_operator(
    2,
    'equivalent?',
    _default_equivalent,
)


def equivalent(info1, info2) -> bool:
    """
    Check if two pieces of information are equivalent.
    
    Uses identity check first (fast), then falls back to generic_equivalent.
    
    Scheme equivalent:
        (define (equivalent? info1 info2)
          (or (eqv? info1 info2)
              (generic-equivalent? info1 info2)))
              
    Returns:
        True if info1 and info2 represent the same information.
    """
    # Fast identity check (like Scheme's eqv?)
    if info1 is info2:
        return True
    # Fall back to generic equivalence
    return generic_equivalent(info1, info2)


def assign_equivalent_operation(operation, *predicates):
    """
    Register a type-specific equivalence handler.
    
    Args:
        operation: The equivalence function to call
        predicates: Type predicates (should be 2 for binary equivalence)
    
    Example:
        >>> assign_equivalent_operation(intervals_equivalent, is_interval, is_interval)
    """
    generic_equivalent.assign_operation(operation, *predicates)


# Predicates for merge operations
def any_p(x) -> bool:
    """Accept any value (always returns True)."""
    return True


def default_merge(content, increment):
    """
    Default merge operation: return content if equal, otherwise contradiction.
    
    Scheme equivalent:
        (lambda (content increment)
          (if (default-equal? content increment)
              content
              the-contradiction))
    """
    if content == increment:
        return content
    else:
        return the_contradiction


# ============================================================================
# Merge with Equivalent Short-Circuit
# ============================================================================
#
# The generic_merge operator handles type-specific merging.
# We wrap it with an equivalent? check for optimization.
#
# Scheme pattern from cells.scm:
#     (define (merge info1 info2)
#       (if (equivalent? info1 info2)
#           info1
#           (let ((answer (generic-merge info1 info2)))
#             ...)))

# The raw generic merge (without equivalent short-circuit)
generic_merge = None  # Will be initialized

def _init_merge():
    """
    Initialize merge as a GenericOperator with equivalent short-circuit.
    Called after primitives module is loaded to avoid circular import.
    """
    global generic_merge, merge
    
    # Create the raw generic merge operator
    generic_merge = make_generic_operator(2, 'generic-merge', default_merge)
    
    # Register nothing/any handlers on the raw merge
    # (assign-operation 'merge (lambda (content increment) increment) nothing? any?)
    generic_merge.assign_operation(lambda content, increment: increment, nothing_p, any_p)
    
    # (assign-operation 'merge (lambda (content increment) content) any? nothing?)
    generic_merge.assign_operation(lambda content, increment: content, any_p, nothing_p)
    
    # The public merge function with equivalent? short-circuit
    # This is an OPTIMIZATION: if two values are equivalent, return immediately
    # without invoking the full merge machinery
    def merge_with_equivalent_shortcircuit(content, increment):
        """
        Merge two pieces of information with equivalent? short-circuit.
        
        Scheme equivalent from cells.scm:
            (define (merge info1 info2)
              (if (equivalent? info1 info2)
                  info1
                  (generic-merge info1 info2)))
        
        OPTIMIZATION: The equivalent? check avoids:
        1. Unnecessary GenericOperator dispatch for identical values
        2. Creating new merged objects when information is unchanged
        3. Triggering downstream propagation for no-op merges
        """
        # Short-circuit: if equivalent, return existing content
        if equivalent(content, increment):
            return content
        # Otherwise, use full generic merge
        return generic_merge(content, increment)
    
    merge = merge_with_equivalent_shortcircuit

# Initialize merge immediately
_init_merge() 




def assign_merge_operation(operation, *predicates):
    """
    Register a merge handler using the unified generic operator system.
    
    Note: Operations are registered on generic_merge (the raw operator),
    not on merge (which adds the equivalent? short-circuit).
    
    Scheme equivalent:
        (assign-operation 'merge operation pred1 pred2)
    
    Args:
        operation: The merge function to call
        predicates: Type predicates (should be 2 for binary merge)
    
    Example:
        >>> assign_merge_operation(merge_intervals, is_interval, is_interval)
    """
    if generic_merge is None:
        raise RuntimeError("generic_merge not initialized. This should not happen.")
    generic_merge.assign_operation(operation, *predicates)


if __name__ == "__main__":
    # Test the merge function
    print("Testing merge function:")
    print()
    
    # Merging with nothing (the dedicated sentinel)
    print("=== Merging with nothing (dedicated sentinel) ===")
    result_nothing1 = merge(nothing, 42)
    print(f"merge(nothing, 42) = {result_nothing1}")  # 42
    
    result_nothing2 = merge(42, nothing)
    print(f"merge(42, nothing) = {result_nothing2}")  # 42
    
    result_nothing3 = merge(nothing, nothing)
    print(f"merge(nothing, nothing) = {result_nothing3}")  # nothing
    print()
    
    # None is a real, storable value now -- not an alias for nothing -- so
    # merging it against a different value is a genuine conflict, same as
    # merging any other two different values.
    print("=== None is real data, not nothing ===")
    result_none1 = merge(None, 42)
    print(f"merge(None, 42) = {result_none1}")  # Contradiction
    print(f"contradictory_p(...) = {contradictory_p(result_none1)}")  # True

    result_none2 = merge(None, None)
    print(f"merge(None, None) = {result_none2}")  # None (equal values merge fine)
    print()
    
    # Same values - should merge successfully
    print("=== Merging same values ===")
    result1 = merge(5, 5)
    print(f"merge(5, 5) = {result1}")  # 5
    print()
    
    # Different values - should produce contradiction
    print("=== Merging different values ===")
    result2 = merge(5, 10)
    print(f"merge(5, 10) = {result2}")  # Contradiction
    print()
    
    # Check if contradiction
    print("=== Testing contradictory_p ===")
    print(f"contradictory_p(result1) = {contradictory_p(result1)}")  # False
    print(f"contradictory_p(result2) = {contradictory_p(result2)}")  # True
    print()
    
    # The contradiction is a singleton
    print("=== Contradiction is a singleton ===")
    result3 = merge(7, 8)
    print(f"result2 is result3 = {result2 is result3}")  # True (same object)
    print(f"result2 is the_contradiction = {result2 is the_contradiction}")  # True
    print()
    
    # Test equivalent? optimization
    print("=== Testing equivalent? optimization ===")
    print(f"equivalent(5, 5) = {equivalent(5, 5)}")  # True (fast identity fails, falls back to ==)
    x = [1, 2, 3]
    print(f"equivalent(x, x) = {equivalent(x, x)}")  # True (fast identity check)
    print(f"equivalent(nothing, nothing) = {equivalent(nothing, nothing)}")  # True
    print(f"equivalent(5, 10) = {equivalent(5, 10)}")  # False
    print()
    
    # Test nothing sentinel
    print("=== Testing nothing sentinel ===")
    print(f"nothing = {nothing}")  # Nothing
    print(f"nothing_p(nothing) = {nothing_p(nothing)}")  # True
    print(f"nothing_p(None) = {nothing_p(None)}")  # False (None is a real value)
    print(f"nothing_p(0) = {nothing_p(0)}")  # False
    print(f"bool(nothing) = {bool(nothing)}")  # False (falsy like None)
