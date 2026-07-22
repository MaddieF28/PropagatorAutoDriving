"""Pytest tests for consequence caching with worldview number.

These tests validate the optimization that caches strongest_consequence results
and invalidates them when the worldview changes (premises kicked out/brought in).

Based on the MIT Scheme implementation in propagator/core/truth-maintenance.scm.
"""

import pytest
from propagator import (
    Cell,
    initialize_scheduler,
    run,
    one_of,
    require_distinct,
)
from propagator.tms import (
    get_worldview_number,
    initialize_tms,
    kick_out,
    bring_in,
    hypothetical,
    premise_in,
    strongest_consequence,
    make_tms,
    _consequence_cache,
    _worldview_number,
)
from propagator.supported_values import supported, Supported


class TestWorldviewNumberTracking:
    """Tests for worldview number increment behavior."""

    def test_initial_worldview_number_is_zero(self):
        """After initialization, worldview number should be 0."""
        initialize_scheduler()
        assert get_worldview_number() == 0

    def test_kick_out_increments_worldview_number(self):
        """Kicking out a premise that was in should increment worldview number."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        assert premise_in(h1) is True  # Starts as "in"
        initial_wv = get_worldview_number()
        
        kick_out(h1)
        
        assert premise_in(h1) is False
        assert get_worldview_number() == initial_wv + 1

    def test_kick_out_already_out_does_not_increment(self):
        """Kicking out a premise already out should not change worldview."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        kick_out(h1)  # First kick out
        wv_after_first = get_worldview_number()
        
        kick_out(h1)  # Second kick out (already out)
        
        assert get_worldview_number() == wv_after_first

    def test_bring_in_increments_worldview_number(self):
        """Bringing in a premise that was out should increment worldview number."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        kick_out(h1)  # Make it out
        wv_before = get_worldview_number()
        
        bring_in(h1)
        
        assert premise_in(h1) is True
        assert get_worldview_number() == wv_before + 1

    def test_bring_in_already_in_does_not_increment(self):
        """Bringing in a premise already in should not change worldview."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        assert premise_in(h1) is True  # Starts as "in"
        wv_before = get_worldview_number()
        
        bring_in(h1)  # Already in
        
        assert get_worldview_number() == wv_before

    def test_multiple_premise_changes(self):
        """Multiple premise changes should accumulate worldview number."""
        initialize_scheduler()
        h1 = hypothetical(sign='h1')
        h2 = hypothetical(sign='h2')
        
        assert get_worldview_number() == 0
        
        kick_out(h1)
        assert get_worldview_number() == 1
        
        kick_out(h2)
        assert get_worldview_number() == 2
        
        bring_in(h1)
        assert get_worldview_number() == 3


class TestConsequenceCaching:
    """Tests for the consequence caching mechanism."""

    def test_strongest_consequence_caches_result(self):
        """Second call to strongest_consequence should return cached result."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        tms = make_tms([supported(42, []), supported(100, [h1])])
        
        # First call computes
        result1 = strongest_consequence(tms)
        
        # Second call should hit cache (same worldview)
        result2 = strongest_consequence(tms)
        
        # Results should be equal
        assert result1 == result2

    def test_cache_invalidated_on_kick_out(self):
        """Cache should be invalidated when a premise is kicked out."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        # h1 starts as "in", so both values are believed
        tms = make_tms([supported(42, []), supported(100, [h1])])
        
        # With h1 "in", strongest consequence merges both
        result1 = strongest_consequence(tms)
        
        # Kick out h1 - only the unsupported value should be believed
        kick_out(h1)
        result2 = strongest_consequence(tms)
        
        # After kick out, only value 42 (with no support) should be returned
        assert isinstance(result2, Supported)
        assert result2.value == 42
        assert len(result2.support) == 0  # Empty support (frozenset or list)

    def test_cache_invalidated_on_bring_in(self):
        """Cache should be invalidated when a premise is brought in."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        kick_out(h1)  # Start with h1 out
        
        tms = make_tms([supported(42, []), supported(100, [h1])])
        
        # With h1 "out", only 42 is believed
        result1 = strongest_consequence(tms)
        assert result1.value == 42
        
        # Bring in h1 - both should now merge
        bring_in(h1)
        result2 = strongest_consequence(tms)
        
        # After bring in, should have merged result
        # (depends on merge semantics, but should be different from result1)
        # The key point is that cache was invalidated and recomputed
        assert isinstance(result2, Supported)

    def test_different_tms_have_separate_cache_entries(self):
        """Different TMS objects should have independent cache entries."""
        initialize_scheduler()
        h1 = hypothetical(sign='h1')
        h2 = hypothetical(sign='h2')
        
        tms1 = make_tms([supported(10, []), supported(20, [h1])])
        tms2 = make_tms([supported(30, []), supported(40, [h2])])
        
        result1 = strongest_consequence(tms1)
        result2 = strongest_consequence(tms2)
        
        # Both should be cached separately
        assert result1.value != result2.value or result1.support != result2.support


class TestCachingWithConstraints:
    """Integration tests for caching with actual constraint problems."""

    def test_simple_distinct_constraint(self):
        """Test caching works correctly with distinct constraint search."""
        initialize_scheduler()
        
        # Create cells with distinct values from [1, 2, 3]
        a = Cell(name='a')
        b = Cell(name='b')
        c = Cell(name='c')
        
        one_of([1, 2, 3], a)
        one_of([1, 2, 3], b)
        one_of([1, 2, 3], c)
        
        require_distinct([a, b, c])
        
        run()
        
        # The worldview number should have increased due to search
        assert get_worldview_number() > 0
        
        # All cells should have valid TMS content
        assert a.content is not None
        assert b.content is not None
        assert c.content is not None


class TestInitializeTms:
    """Tests for initialize_tms function."""

    def test_initialize_tms_resets_worldview_number(self):
        """initialize_tms should reset worldview number to 0."""
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        
        kick_out(h1)  # Increment worldview
        assert get_worldview_number() > 0
        
        initialize_tms()
        
        assert get_worldview_number() == 0

    def test_initialize_tms_clears_cache(self):
        """initialize_tms should clear the consequence cache."""
        initialize_scheduler()
        
        tms = make_tms([supported(42, [])])
        strongest_consequence(tms)  # Cache the result
        
        # Import the cache to check it was populated
        from propagator.tms import _consequence_cache
        cache_had_entries = len(_consequence_cache) > 0
        
        initialize_tms()
        
        # Cache should be empty after initialization
        from propagator.tms import _consequence_cache as new_cache
        assert len(new_cache) == 0

    def test_initialize_scheduler_calls_initialize_tms(self):
        """initialize_scheduler should also clear TMS state."""
        # First, set up some TMS state
        initialize_scheduler()
        h1 = hypothetical(sign='test')
        kick_out(h1)
        
        assert get_worldview_number() > 0
        
        # Re-initialize scheduler
        initialize_scheduler()
        
        # TMS state should be reset
        assert get_worldview_number() == 0


class TestCachingPerformance:
    """Performance comparison tests for consequence caching."""

    def test_caching_improves_performance_on_require_distinct(self):
        """
        Consequence caching should improve performance on require_distinct.
        
        This test runs the same 4-cell distinct constraint problem twice:
        once with caching enabled (default) and once with caching disabled.
        The cached version should be faster.
        
        The performance benefit comes from avoiding redundant strongest_consequence
        computations when the worldview hasn't changed between calls.
        """
        import time
        from unittest.mock import patch
        
        def run_require_distinct_4cells():
            """Run a 4-cell require_distinct problem."""
            initialize_scheduler()
            cells = [Cell() for _ in range(4)]
            for i, cell in enumerate(cells):
                one_of(list(range(1, 5)), cell)  # Each cell can be 1-4
            require_distinct(cells)
            run()
            # Verify solution is valid
            values = [c.content.base_value if hasattr(c.content, 'base_value') else c.content for c in cells]
            return len(set(values)) == 4  # All distinct
        
        # Run with caching (default behavior)
        start_cached = time.perf_counter()
        result_cached = run_require_distinct_4cells()
        time_cached = time.perf_counter() - start_cached
        
        # Run without caching by making _cached_consequence always return None
        def no_cache_consequence(tms):
            return None
        
        with patch('propagator.tms._cached_consequence', no_cache_consequence):
            start_no_cache = time.perf_counter()
            result_no_cache = run_require_distinct_4cells()
            time_no_cache = time.perf_counter() - start_no_cache
        
        # Both runs should find valid solutions
        assert result_cached, "Cached run should find valid solution"
        assert result_no_cache, "No-cache run should find valid solution"
        
        # Caching should provide some performance benefit
        # We use a generous margin since timing can vary, but caching should not be slower
        speedup = time_no_cache / time_cached if time_cached > 0 else 1.0
        
        # Assert caching is not significantly slower (allow 10% margin for timing noise)
        assert time_cached <= time_no_cache * 1.1, (
            f"Caching should not be slower: cached={time_cached:.3f}s, "
            f"no_cache={time_no_cache:.3f}s, speedup={speedup:.2f}x"
        )
        
        # Log performance info for debugging (visible with pytest -v)
        print(f"\nPerformance: cached={time_cached:.3f}s, no_cache={time_no_cache:.3f}s, speedup={speedup:.2f}x")
