"""Pytest coverage for merge behavior across numbers and intervals.

These tests protect the core merge semantics that all propagators rely on:
consistent values should stabilize, and contradictions should raise.

Also tests the Nothing sentinel and equivalent operator that are part of merge.py.
"""

import pytest

from propagator import (
    Cell,
    Log,
    LogEntry,
    append_log_entry,
    count_log,
    count_window,
    detect_bursts_in_log,
    entries_between,
    entry_to_log,
    filter_log,
    filter_after_timestamp,
    filter_before_timestamp,
    latest,
    latest_payload,
    latest_before,
    make_interval,
    make_log,
    make_log_entry,
    map_log,
    map_payload_values,
    singleton_log,
    timestamps,
    window_log,
    # Nothing sentinel
    nothing,
    nothing_p,
    # Equivalence system  
    equivalent,
    generic_equivalent,
    assign_equivalent_operation,
    # Merge system
    merge,
    the_contradiction,
    contradictory_p,
)
from propagator.nothing import _Nothing


# =============================================================================
# Core Merge Semantics
# =============================================================================

def test_merge_keeps_consistent_number():
    """Motivation: repeated identical facts should be idempotent."""
    cell = Cell()
    cell.add_content(42)
    cell.add_content(42)
    assert cell.content == 42


def test_merge_intersects_intervals():
    """Motivation: interval information should narrow via intersection."""
    cell = Cell()
    cell.add_content(make_interval(1, 5))
    cell.add_content(make_interval(3, 7))
    assert cell.content == make_interval(3, 5)


def test_merge_contradiction_for_numbers():
    """Motivation: conflicting concrete facts must be rejected."""
    cell = Cell()
    cell.add_content(10)
    with pytest.raises(Exception):
        cell.add_content(20)


def test_merge_contradiction_for_intervals():
    """Motivation: disjoint interval evidence should raise a contradiction."""
    cell = Cell()
    cell.add_content(make_interval(1, 3))
    with pytest.raises(Exception):
        cell.add_content(make_interval(5, 7))


# =============================================================================
# Nothing Sentinel Tests
# =============================================================================

class TestNothingSentinel:
    """Tests for the dedicated Nothing sentinel."""
    
    def test_nothing_is_singleton(self):
        """Nothing should be a singleton - same object everywhere."""
        n1 = _Nothing()
        n2 = _Nothing()
        assert n1 is n2
        assert n1 is nothing
    
    def test_nothing_repr(self):
        """Nothing should have a clear string representation."""
        assert repr(nothing) == "Nothing"
        assert str(nothing) == "Nothing"
    
    def test_nothing_is_falsy(self):
        """Nothing should be falsy like None for convenience."""
        assert not nothing
        assert bool(nothing) is False
    
    def test_nothing_p_with_nothing(self):
        """nothing_p should return True for the nothing sentinel."""
        assert nothing_p(nothing) is True
    
    def test_nothing_p_is_strict_about_none(self):
        """nothing_p should NOT treat None as nothing -- None is a storable value."""
        assert nothing_p(None) is False
    
    def test_nothing_p_rejects_other_values(self):
        """nothing_p should return False for normal values."""
        assert nothing_p(0) is False
        assert nothing_p("") is False
        assert nothing_p(False) is False
        assert nothing_p([]) is False
    
    def test_merge_with_nothing(self):
        """Merging with nothing should return the other value."""
        assert merge(nothing, 42) == 42
        assert merge(42, nothing) == 42
        assert merge(nothing, nothing) is nothing
    
    def test_merge_treats_none_as_real_data(self):
        """None is a genuine value now: merging it with a different value conflicts."""
        assert contradictory_p(merge(None, 42)) is True
        assert contradictory_p(merge(42, None)) is True
        # But merging None with itself is fine, like any other equal value.
        assert merge(None, None) is None


# =============================================================================
# Equivalent Operator Tests
# =============================================================================

class TestEquivalentOperator:
    """Tests for the generic_equivalent operator."""
    
    def test_equivalent_with_identity(self):
        """Equivalent should return True for same object (identity check)."""
        x = [1, 2, 3]
        assert equivalent(x, x) is True
    
    def test_equivalent_with_equal_immutable_values(self):
        """Equivalent should return True for equal immutable values."""
        assert equivalent(5, 5) is True
        assert equivalent("hello", "hello") is True
        assert equivalent(nothing, nothing) is True
        assert equivalent((1, 2), (1, 2)) is True
    
    def test_equivalent_with_different_values(self):
        """Equivalent should return False for different values."""
        assert equivalent(5, 10) is False
        assert equivalent("hello", "world") is False
    
    def test_equivalent_conservative_for_unknown_types(self):
        """Equivalent should be conservative (return False) for unknown types."""
        # For custom types without registered equivalence, we require identity
        class CustomType:
            def __init__(self, v):
                self.v = v
            def __eq__(self, other):
                return isinstance(other, CustomType) and self.v == other.v
        
        a = CustomType(5)
        b = CustomType(5)
        # Even though a == b, equivalent returns False for safety
        # This ensures custom types must opt-in to equivalence short-circuiting
        assert equivalent(a, b) is False
        # But identity still works
        assert equivalent(a, a) is True
    
    def test_assign_equivalent_operation(self):
        """Custom equivalence handlers can be registered."""
        class MyType:
            def __init__(self, v):
                self.v = v
        
        def my_type_p(x):
            return isinstance(x, MyType)
        
        def my_type_equivalent(a, b):
            return a.v == b.v
        
        # Register custom equivalence
        assign_equivalent_operation(my_type_equivalent, my_type_p, my_type_p)
        
        # Now equivalent should use our handler
        a = MyType(5)
        b = MyType(5)
        c = MyType(10)
        
        assert equivalent(a, b) is True
        assert equivalent(a, c) is False


# =============================================================================
# Merge + Equivalent Integration Tests  
# =============================================================================

class TestMergeEquivalentIntegration:
    """Tests for merge using equivalent short-circuit correctly."""
    
    def test_merge_same_value_returns_first(self):
        """Merging same values should return the first value."""
        result = merge(42, 42)
        assert result == 42
    
    def test_merge_different_values_gives_contradiction(self):
        """Merging different values should give contradiction."""
        result = merge(5, 10)
        assert contradictory_p(result) is True
        assert result is the_contradiction
    
    def test_merge_does_not_burden_new_types(self):
        """
        Custom types should work with merge without needing to register equivalence.
        
        The equivalent short-circuit should be conservative for unknown types,
        falling through to generic_merge which handles type dispatch properly.
        """
        from propagator import assign_merge_operation
        
        class Vector:
            def __init__(self, x, y):
                self.x, self.y = x, y
            def __eq__(self, other):
                return isinstance(other, Vector) and self.x == other.x and self.y == other.y
        
        def vector_p(x):
            return isinstance(x, Vector)
        
        def merge_vectors(v1, v2):
            if v1.x == v2.x and v1.y == v2.y:
                return v1
            return the_contradiction
        
        # Register merge operation for Vector type
        assign_merge_operation(merge_vectors, vector_p, vector_p)
        
        # Now merge should work - equivalent returns False for unknown types,
        # so it falls through to generic_merge which uses our handler
        v1 = Vector(1, 2)
        v2 = Vector(1, 2)
        v3 = Vector(3, 4)
        
        assert merge(v1, v2) == v1  # Same content
        assert contradictory_p(merge(v1, v3))  # Different content


# =============================================================================
# Log Semilattice Tests
# =============================================================================

class TestLogMerge:
    """Logs should merge monotonically as partially ordered growing sets."""

    def test_log_merge_unions_entries(self):
        left = make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
        ])
        right = make_log([
            make_log_entry("b", 2.0),
            make_log_entry("c", 3.0),
        ])

        merged = merge(left, right)
        assert isinstance(merged, Log)
        assert merged == make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
            make_log_entry("c", 3.0),
        ])

    def test_log_merge_is_order_independent(self):
        e1 = make_log_entry("a", 1.0)
        e2 = make_log_entry("b", 2.0)
        e3 = make_log_entry("c", 3.0)

        l1 = make_log([e1, e2])
        l2 = make_log([e3])

        assert merge(l1, l2) == merge(l2, l1)

    def test_log_cell_can_ingest_entry_incrementally(self):
        cell = Cell()
        cell.add_content(make_log())

        cell.add_content(make_log_entry("first", 10.0))
        cell.add_content(make_log_entry("second", 20.0))
        cell.add_content(make_log_entry("second", 20.0))

        assert cell.content == make_log([
            make_log_entry("first", 10.0),
            make_log_entry("second", 20.0),
        ])

    def test_append_log_entry_helper(self):
        log = make_log()
        log = append_log_entry(log, "v1", 1.0)
        log = append_log_entry(log, "v1", 1.0)
        log = append_log_entry(log, "v2", 2.0)

        assert log.values == (
            LogEntry(value="v1", timestamp=1.0),
            LogEntry(value="v2", timestamp=2.0),
        )

    def test_singleton_log_creates_one_entry_increment(self):
        entry = make_log_entry("v1", 1.0)
        assert singleton_log(entry) == make_log([entry])

    def test_entry_to_log_accepts_plain_log_entry(self):
        entry = make_log_entry("v1", 1.0)
        assert entry_to_log(entry) == make_log([entry])

    def test_log_window_and_count_operators(self):
        log = make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
            make_log_entry("c", 4.0),
            make_log_entry("d", 9.0),
        ])

        assert count_log(log) == 4
        assert timestamps(log) == (1.0, 2.0, 4.0, 9.0)
        assert entries_between(log, 2.0, 4.0) == make_log([
            make_log_entry("b", 2.0),
            make_log_entry("c", 4.0),
        ])
        assert window_log(log, 4.0, 3.0) == make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
            make_log_entry("c", 4.0),
        ])
        assert count_window(log, 4.0, 3.0) == 3

    def test_log_latest_filter_and_map_operators(self):
        log = make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
            make_log_entry("c", 4.0),
        ])

        assert latest(log) == make_log_entry("c", 4.0)
        assert latest_payload(log) == "c"
        assert latest_before(log, 2.5) == make_log_entry("b", 2.0)
        assert latest_before(log, 0.5) is None

        filtered = filter_log(log, lambda entry: entry.timestamp >= 2.0)
        assert filtered == make_log([
            make_log_entry("b", 2.0),
            make_log_entry("c", 4.0),
        ])
        assert filter_after_timestamp(log, 2.0) == make_log([
            make_log_entry("c", 4.0),
        ])
        assert filter_before_timestamp(log, 2.0) == make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
        ])

        mapped = map_log(
            log,
            lambda entry: make_log_entry(str(entry.value).upper(), entry.timestamp + 10.0),
        )
        assert mapped == make_log([
            make_log_entry("A", 11.0),
            make_log_entry("B", 12.0),
            make_log_entry("C", 14.0),
        ])
        assert map_payload_values(log, str.upper) == make_log([
            make_log_entry("A", 1.0),
            make_log_entry("B", 2.0),
            make_log_entry("C", 4.0),
        ])

    def test_detect_bursts_in_log_operator(self):
        log = make_log([
            make_log_entry("a", 1.0),
            make_log_entry("b", 2.0),
            make_log_entry("c", 4.0),
            make_log_entry("d", 9.0),
            make_log_entry("e", 10.0),
            make_log_entry("f", 11.0),
        ])

        bursts = detect_bursts_in_log(
            log,
            5.0,
            3,
            lambda count, min_events, window_seconds, event_timestamp: (
                "burst",
                count,
                min_events,
                window_seconds,
                event_timestamp,
            ),
        )

        assert bursts == make_log([
            make_log_entry(("burst", 3, 3, 5.0, 4.0), 4.0),
            make_log_entry(("burst", 3, 3, 5.0, 11.0), 11.0),
        ])
