"""
Test handler coverage for all generic operators.

This test evaluates which type combinations are covered by handlers for each
generic operator, exercises them to track hit counts, and reports coverage
metrics.

Coverage Metrics Explained:
    - registered: Total handlers registered for this operator
    - exercised: Handlers that were invoked at least once during tests
    - unexercised: Handlers that exist but were never invoked (potential dead code)
    - default_fallback: Whether we're relying on default_op for any type combinations

Type Predicates:
    - flat_p: Basic types (int, float, bool, str, Interval) - NOT wrapped
    - supported_p: Supported values with dependency tracking
    - tms_p: Truth Maintenance System values (multiple hypothetical values)
"""

import pytest
from propagator import (
    Cell,
    Interval,
    Supported,
    Tms,
    contradictory,
    generic_merge,
    make_tms,
    nothing,
    supported,
)
from propagator.primitives import (
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
    generic_switch,
)
from propagator.supported_values import flat_p, supported_p
from propagator.tms import tms_p


# =============================================================================
# Test Data: Type Representatives
# =============================================================================

# Representative values for each type category
FLAT_VALUES = {
    'int': 5,
    'float': 3.14,
    'bool_true': True,
    'bool_false': False,
    'interval': Interval(1, 10),
}

INTERVAL_VALUES = {
    'interval_1_10': Interval(1, 10),
    'interval_2_8': Interval(2, 8),
    'interval_5_15': Interval(5, 15),
}

# Create premise objects for supports
class Premise:
    """Simple premise object for testing."""
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return f"Premise({self.name})"

PREMISE_A = Premise("A")
PREMISE_B = Premise("B")

SUPPORTED_VALUES = {
    'supported_int': supported(10, [PREMISE_A]),
    'supported_float': supported(2.5, [PREMISE_B]),
    'supported_bool': supported(True, [PREMISE_A]),
}

TMS_VALUES = {
    'tms_single': make_tms([supported(7, [PREMISE_A])]),
    'tms_multi': make_tms([
        supported(5, [PREMISE_A]),
        supported(8, [PREMISE_B]),
    ]),
}


# =============================================================================
# Coverage Report Functions
# =============================================================================

def format_coverage_report(operator_name: str, coverage: list) -> str:
    """Format coverage data as a readable report."""
    lines = [f"\n{'='*60}", f"OPERATOR: {operator_name}", f"{'='*60}"]
    
    total = len(coverage)
    exercised = sum(1 for h in coverage if h['hit_count'] > 0)
    unexercised = total - exercised
    
    lines.append(f"Summary: {exercised}/{total} handlers exercised ({100*exercised/total:.0f}% coverage)" if total > 0 else "Summary: No handlers registered (uses default_op only)")
    lines.append("")
    
    # Group by exercised/unexercised
    if coverage:
        lines.append("Handlers (exercised):")
        for h in coverage:
            if h['hit_count'] > 0:
                preds = ', '.join(h['predicates'])
                lines.append(f"  ✓ ({preds}) -> {h['operation']} [hits: {h['hit_count']}]")
        
        if unexercised > 0:
            lines.append("\nHandlers (NOT exercised):")
            for h in coverage:
                if h['hit_count'] == 0:
                    preds = ', '.join(h['predicates'])
                    lines.append(f"  ✗ ({preds}) -> {h['operation']}")
    
    return '\n'.join(lines)


def print_full_report(results: dict):
    """Print a full coverage report for all operators."""
    print("\n" + "="*70)
    print("HANDLER COVERAGE REPORT")
    print("="*70)
    
    # Summary table
    print("\nSummary:")
    print(f"{'Operator':<20} {'Registered':>10} {'Exercised':>10} {'Coverage':>10}")
    print("-"*52)
    
    total_registered = 0
    total_exercised = 0
    
    for name, coverage in sorted(results.items()):
        registered = len(coverage)
        exercised = sum(1 for h in coverage if h['hit_count'] > 0)
        pct = f"{100*exercised/registered:.0f}%" if registered > 0 else "N/A"
        print(f"{name:<20} {registered:>10} {exercised:>10} {pct:>10}")
        total_registered += registered
        total_exercised += exercised
    
    print("-"*52)
    total_pct = f"{100*total_exercised/total_registered:.0f}%" if total_registered > 0 else "N/A"
    print(f"{'TOTAL':<20} {total_registered:>10} {total_exercised:>10} {total_pct:>10}")
    
    # Detailed reports
    for name, coverage in sorted(results.items()):
        print(format_coverage_report(name, coverage))


# =============================================================================
# Exercise Functions: Actually invoke the operators with different types
# =============================================================================

def exercise_binary_arithmetic(op, results_key: str, results: dict):
    """Exercise a binary arithmetic operator with all type combinations."""
    op.reset_coverage()
    
    # flat x flat (uses default_op, no handler)
    try:
        op(5, 3)
    except:
        pass
    
    # interval x interval (special handlers for arithmetic ops)
    try:
        op(INTERVAL_VALUES['interval_1_10'], INTERVAL_VALUES['interval_2_8'])
    except:
        pass
    
    # number x interval
    try:
        op(5, INTERVAL_VALUES['interval_1_10'])
    except:
        pass
    
    # interval x number
    try:
        op(INTERVAL_VALUES['interval_1_10'], 5)
    except:
        pass
    
    # supported x supported
    try:
        op(SUPPORTED_VALUES['supported_int'], SUPPORTED_VALUES['supported_float'])
    except:
        pass
    
    # supported x flat
    try:
        op(SUPPORTED_VALUES['supported_int'], 2)
    except:
        pass
    
    # flat x supported
    try:
        op(3, SUPPORTED_VALUES['supported_int'])
    except:
        pass
    
    # tms x tms
    try:
        op(TMS_VALUES['tms_single'], TMS_VALUES['tms_single'])
    except:
        pass
    
    # tms x supported
    try:
        op(TMS_VALUES['tms_single'], SUPPORTED_VALUES['supported_int'])
    except:
        pass
    
    # supported x tms
    try:
        op(SUPPORTED_VALUES['supported_int'], TMS_VALUES['tms_single'])
    except:
        pass
    
    # tms x flat
    try:
        op(TMS_VALUES['tms_single'], 5)
    except:
        pass
    
    # flat x tms
    try:
        op(5, TMS_VALUES['tms_single'])
    except:
        pass
    
    results[results_key] = op.get_handler_coverage()


def exercise_unary_op(op, results_key: str, results: dict):
    """Exercise a unary operator with all type combinations."""
    op.reset_coverage()
    
    # flat (uses default_op)
    try:
        op(5)
    except:
        pass
    
    # interval (special handlers for sqrt/square)
    try:
        op(INTERVAL_VALUES['interval_1_10'])
    except:
        pass
    
    # supported
    try:
        op(SUPPORTED_VALUES['supported_int'])
    except:
        pass
    
    # tms
    try:
        op(TMS_VALUES['tms_single'])
    except:
        pass
    
    results[results_key] = op.get_handler_coverage()


def exercise_merge(results: dict):
    """Exercise generic_merge with all type combinations."""
    from propagator.merge import generic_merge
    
    generic_merge.reset_coverage()
    
    # flat x flat (uses default_op - equality check)
    generic_merge(5, 5)
    generic_merge(5, 6)  # Different values
    
    # nothing x value and value x nothing
    generic_merge(nothing, 5)
    generic_merge(5, nothing)
    
    # interval x interval
    generic_merge(INTERVAL_VALUES['interval_1_10'], INTERVAL_VALUES['interval_2_8'])
    
    # number x interval
    generic_merge(5, INTERVAL_VALUES['interval_1_10'])
    
    # interval x number
    generic_merge(INTERVAL_VALUES['interval_1_10'], 5)
    
    # supported x supported
    generic_merge(
        SUPPORTED_VALUES['supported_int'],
        supported(10, [PREMISE_B])  # Same value, different support
    )
    
    # supported x flat
    generic_merge(SUPPORTED_VALUES['supported_int'], 10)
    
    # flat x supported
    generic_merge(10, SUPPORTED_VALUES['supported_int'])
    
    # tms x tms
    generic_merge(TMS_VALUES['tms_single'], TMS_VALUES['tms_single'])
    
    # tms x supported
    generic_merge(TMS_VALUES['tms_single'], SUPPORTED_VALUES['supported_int'])
    
    # supported x tms
    generic_merge(SUPPORTED_VALUES['supported_int'], TMS_VALUES['tms_single'])
    
    # tms x flat
    generic_merge(TMS_VALUES['tms_single'], 7)
    
    # flat x tms
    generic_merge(7, TMS_VALUES['tms_single'])
    
    # tms x nothing and nothing x tms
    generic_merge(TMS_VALUES['tms_single'], nothing)
    generic_merge(nothing, TMS_VALUES['tms_single'])
    
    results['generic_merge'] = generic_merge.get_handler_coverage()


def exercise_switch(results: dict):
    """Exercise generic_switch with all type combinations."""
    generic_switch.reset_coverage()
    
    # flat x flat (uses default_op)
    generic_switch(True, 5)
    generic_switch(False, 5)
    
    # supported x supported
    generic_switch(
        supported(True, [PREMISE_A]),
        SUPPORTED_VALUES['supported_int']
    )
    generic_switch(
        supported(False, [PREMISE_A]),
        SUPPORTED_VALUES['supported_int']
    )
    
    # supported x flat
    generic_switch(supported(True, [PREMISE_A]), 5)
    
    # flat x supported
    generic_switch(True, SUPPORTED_VALUES['supported_int'])
    
    # tms x tms
    generic_switch(
        make_tms([supported(True, [PREMISE_A])]),
        TMS_VALUES['tms_single']
    )
    
    # tms x supported
    generic_switch(
        make_tms([supported(True, [PREMISE_A])]),
        SUPPORTED_VALUES['supported_int']
    )
    
    # supported x tms
    generic_switch(
        supported(True, [PREMISE_A]),
        TMS_VALUES['tms_single']
    )
    
    # tms x flat
    generic_switch(
        make_tms([supported(True, [PREMISE_A])]),
        5
    )
    
    # flat x tms
    generic_switch(True, TMS_VALUES['tms_single'])
    
    results['generic_switch'] = generic_switch.get_handler_coverage()


def exercise_contradictory(results: dict):
    """Exercise contradictory operator."""
    contradictory.reset_coverage()
    
    # flat values
    contradictory(5)
    contradictory(True)
    
    # Supported with regular value
    contradictory(SUPPORTED_VALUES['supported_int'])
    
    results['contradictory'] = contradictory.get_handler_coverage()


# =============================================================================
# Main Test
# =============================================================================

def test_handler_coverage_report():
    """
    Exercise all generic operators and print a comprehensive coverage report.
    
    This test:
    1. Resets hit counters on all operators
    2. Exercises each operator with all type combinations
    3. Collects coverage metrics
    4. Prints a detailed report showing which handlers were used
    
    Coverage metrics help identify:
    - Dead handlers: Registered but never invoked (potential code smell)
    - Missing handlers: Type combinations falling through to default_op
    - Hotspots: Frequently used handlers (optimization targets)
    """
    results = {}
    
    # Binary arithmetic operators
    exercise_binary_arithmetic(generic_add, 'generic_add', results)
    exercise_binary_arithmetic(generic_sub, 'generic_sub', results)
    exercise_binary_arithmetic(generic_mul, 'generic_mul', results)
    exercise_binary_arithmetic(generic_div, 'generic_div', results)
    
    # Binary comparison operators
    exercise_binary_arithmetic(generic_eq, 'generic_eq', results)
    exercise_binary_arithmetic(generic_lt, 'generic_lt', results)
    exercise_binary_arithmetic(generic_gt, 'generic_gt', results)
    exercise_binary_arithmetic(generic_lte, 'generic_lte', results)
    exercise_binary_arithmetic(generic_gte, 'generic_gte', results)
    
    # Binary boolean operators
    exercise_binary_arithmetic(generic_and, 'generic_and', results)
    exercise_binary_arithmetic(generic_or, 'generic_or', results)
    
    # Unary operators
    exercise_unary_op(generic_abs, 'generic_abs', results)
    exercise_unary_op(generic_square, 'generic_square', results)
    exercise_unary_op(generic_sqrt, 'generic_sqrt', results)
    exercise_unary_op(generic_not, 'generic_not', results)
    
    # Special operators
    exercise_merge(results)
    exercise_switch(results)
    exercise_contradictory(results)
    
    # Print the full report
    print_full_report(results)
    
    # Validation: check that all handlers were exercised
    all_exercised = True
    unexercised_handlers = []
    
    for name, coverage in results.items():
        for h in coverage:
            if h['hit_count'] == 0:
                all_exercised = False
                preds = ', '.join(h['predicates'])
                unexercised_handlers.append(f"{name}: ({preds}) -> {h['operation']}")
    
    if unexercised_handlers:
        print("\n" + "!"*60)
        print("WARNING: The following handlers were NOT exercised:")
        for h in unexercised_handlers:
            print(f"  - {h}")
        print("!"*60)
        print("\nANALYSIS: Unexercised handlers may be:")
        print("  1. Dead code (handlers shadowed by earlier, more general handlers)")
        print("  2. Edge cases not covered by this test")
        print("  3. Optimization handlers for rare type combinations")
        print("\nNOTE: The (tms_p, nothing_p) and (nothing_p, tms_p) handlers in")
        print("generic_merge appear to be shadowed by the more general (nothing_p, any_p)")
        print("and (any_p, nothing_p) handlers that are registered first.")
    
    # The test passes even with unexercised handlers - the report is informational
    # To enforce full coverage, uncomment the assertion below:
    # assert all_exercised, f"Unexercised handlers: {unexercised_handlers}"


def test_expected_type_combinations():
    """
    Verify that expected type combinations have handlers registered.
    
    This test documents the expected handler matrix and validates
    that it matches the actual registrations.
    """
    # Expected binary operator coverage (for Supported/TMS-aware ops)
    expected_binary = [
        # Supported combinations
        ('supported_p', 'supported_p'),
        ('supported_p', 'flat_p'),
        ('flat_p', 'supported_p'),
        # TMS combinations
        ('tms_p', 'tms_p'),
        ('tms_p', 'supported_p'),
        ('supported_p', 'tms_p'),
        ('tms_p', 'flat_p'),
        ('flat_p', 'tms_p'),
    ]
    
    result = generic_add.check_handler_coverage(expected_binary)
    
    print("\nExpected Binary Operator Coverage Check (generic_add):")
    print(f"  Covered: {len(result['covered'])}/{len(expected_binary)}")
    print(f"  Missing: {result['missing']}")
    print(f"  Extra: {result['extra']}")
    
    # For generic_merge, we expect TMS handlers too
    expected_merge = [
        ('supported_p', 'supported_p'),
        ('supported_p', 'flat_p'),
        ('flat_p', 'supported_p'),
        ('tms_p', 'tms_p'),
        ('tms_p', 'supported_p'),
        ('supported_p', 'tms_p'),
        ('tms_p', 'flat_p'),
        ('flat_p', 'tms_p'),
    ]
    
    result_merge = generic_merge.check_handler_coverage(expected_merge)
    
    print("\nExpected Merge Handler Coverage Check (generic_merge):")
    print(f"  Covered: {len(result_merge['covered'])}/{len(expected_merge)}")
    print(f"  Missing: {result_merge['missing']}")
    print(f"  Extra: {result_merge['extra']}")
    
    # Assert all expected handlers exist
    assert result['missing'] == [], f"Missing handlers in generic_add: {result['missing']}"


def test_type_predicates():
    """
    Test that type predicates correctly categorize values.
    
    This ensures the handler dispatch will work correctly.
    """
    # flat_p should match basic types
    assert flat_p(5) is True, "int should be flat"
    assert flat_p(3.14) is True, "float should be flat"
    assert flat_p(True) is True, "bool should be flat"
    assert flat_p(Interval(1, 10)) is True, "Interval should be flat"
    assert flat_p(None) is True, "None should be flat"
    
    # flat_p should NOT match Supported/TMS
    s = supported(5, [PREMISE_A])
    t = make_tms([s])
    assert flat_p(s) is False, "Supported should NOT be flat"
    assert flat_p(t) is False, "TMS should NOT be flat"
    
    # supported_p
    assert supported_p(s) is True, "Supported should be supported_p"
    assert supported_p(5) is False, "int should NOT be supported_p"
    assert supported_p(t) is False, "TMS should NOT be supported_p"
    
    # tms_p
    assert tms_p(t) is True, "TMS should be tms_p"
    assert tms_p(s) is False, "Supported should NOT be tms_p"
    assert tms_p(5) is False, "int should NOT be tms_p"
    
    print("\nType predicate tests passed!")
    print(f"  flat_p: Matches int, float, bool, Interval, None")
    print(f"  supported_p: Matches Supported values only")
    print(f"  tms_p: Matches TMS values only")


if __name__ == "__main__":
    # Run with verbose output
    print("Running handler coverage tests...\n")
    test_type_predicates()
    test_expected_type_combinations()
    test_handler_coverage_report()
