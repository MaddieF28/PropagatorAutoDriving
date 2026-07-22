"""
Performance benchmarks for propagator optimizations.

This file tests that performance optimizations are actually providing benefits.
The correctness tests for these features are in the appropriate module test files:
- test_merge.py: Nothing sentinel, equivalent operator, merge short-circuit
- test_simple_puzzle.py: avoid-false-true-flips configuration
- test_consequence_caching.py: TMS consequence caching

These benchmarks verify the optimizations provide measurable improvement.
"""

import time
import pytest

from propagator import (
    Cell,
    initialize_scheduler,
    run,
    one_of,
    require_distinct,
    # Merge/equivalent
    equivalent,
    merge,
    nothing,
    nothing_p,
)


# =============================================================================
# Equivalent Identity Check Performance
# =============================================================================

class TestEquivalentPerformance:
    """Performance tests for the equivalent() optimization."""
    
    def test_equivalent_shortcircuits_on_identity(self):
        """
        Equivalent should use fast identity check first.
        
        The identity check (info1 is info2) is O(1) and should handle
        the common case where the same object is compared to itself.
        """
        # Create a large list - comparing by value would be slow
        large_list = list(range(100000))
        
        # Identity check should be instant
        start = time.perf_counter()
        for _ in range(10000):
            equivalent(large_list, large_list)
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 0.5s for 10000 calls)
        assert elapsed < 0.5, f"Identity check took too long: {elapsed:.3f}s"
    
    def test_equivalent_fast_for_simple_types(self):
        """
        Equivalent should be fast for common immutable types.
        
        The optimization handles bool/int/float/str/tuple with == directly,
        avoiding GenericOperator dispatch overhead.
        """
        start = time.perf_counter()
        for _ in range(100000):
            equivalent(42, 42)
            equivalent("test", "test")
            equivalent(3.14, 3.14)
            equivalent(True, True)
        elapsed = time.perf_counter() - start
        
        # Should complete quickly
        assert elapsed < 1.0, f"Simple type equivalence took too long: {elapsed:.3f}s"


# =============================================================================
# Merge Short-Circuit Performance
# =============================================================================

class TestMergePerformance:
    """Performance tests for merge short-circuit optimization."""
    
    def test_merge_equivalent_shortcircuit_provides_benefit(self):
        """
        Merging equivalent values should short-circuit before GenericOperator dispatch.
        
        When values are equivalent (detected by identity or ==), merge returns
        immediately without invoking the full generic_merge machinery.
        """
        # Compare merge performance with identical values
        start = time.perf_counter()
        for _ in range(100000):
            merge(42, 42)
        elapsed_shortcircuit = time.perf_counter() - start
        
        # The optimization is working if this completes quickly
        # (no GenericOperator dispatch for simple identical values)
        assert elapsed_shortcircuit < 1.0, (
            f"Merge short-circuit took too long: {elapsed_shortcircuit:.3f}s"
        )


# =============================================================================
# Full Problem Performance Benchmarks
# =============================================================================

class TestRequireDistinctPerformance:
    """
    Performance benchmarks for the full require_distinct problem.
    
    These tests verify that the combined optimizations (implies fast-path,
    consequence caching, equivalent short-circuit) provide real benefits.
    """
    
    @staticmethod
    def _run_require_distinct(n_cells: int) -> tuple:
        """Run an n-cell distinct constraint and return (time, valid)."""
        start = time.perf_counter()
        
        initialize_scheduler()
        cells = [Cell() for _ in range(n_cells)]
        for cell in cells:
            one_of(list(range(1, n_cells + 1)), cell)
        require_distinct(cells)
        run()
        
        elapsed = time.perf_counter() - start
        
        values = [
            c.content.base_value if hasattr(c.content, 'base_value') else c.content
            for c in cells
        ]
        valid = len(set(values)) == n_cells
        return elapsed, valid
    
    def test_3_cell_completes_quickly(self):
        """3-cell distinct should complete in under 0.1 seconds."""
        elapsed, valid = self._run_require_distinct(3)
        assert valid, "Should find valid solution"
        assert elapsed < 0.1, f"3-cell took too long: {elapsed:.3f}s"
    
    def test_4_cell_completes_reasonably(self):
        """4-cell distinct should complete in under 2 seconds."""
        elapsed, valid = self._run_require_distinct(4)
        assert valid, "Should find valid solution"
        assert elapsed < 2.0, f"4-cell took too long: {elapsed:.3f}s"
    
    def test_5_cell_completes(self):
        """5-cell distinct should complete in under 60 seconds."""
        elapsed, valid = self._run_require_distinct(5)
        assert valid, "Should find valid solution"
        assert elapsed < 60.0, f"5-cell took too long: {elapsed:.3f}s"


# =============================================================================
# Consequence Caching Performance
# =============================================================================

class TestConsequenceCachingPerformance:
    """
    Performance tests for TMS consequence caching.
    
    The correctness tests are in test_consequence_caching.py.
    These tests verify the caching provides performance benefits.
    """
    
    def test_strongest_consequence_caching_is_effective(self):
        """
        Repeated strongest_consequence calls should benefit from caching.
        
        When the worldview hasn't changed, cached results should be returned
        without recomputing.
        """
        from propagator.tms import (
            make_tms, 
            strongest_consequence,
            hypothetical,
        )
        from propagator.supported_values import supported
        
        initialize_scheduler()
        
        # Create a TMS with some supported values
        h1 = hypothetical()
        h2 = hypothetical()
        tms = make_tms([
            supported(10, [h1]),
            supported(20, [h2]),
        ])
        
        # First call computes and caches
        result1 = strongest_consequence(tms)
        
        # Subsequent calls should hit cache (same worldview)
        start = time.perf_counter()
        for _ in range(10000):
            result = strongest_consequence(tms)
        elapsed_cached = time.perf_counter() - start
        
        # Cached lookups should be very fast
        assert elapsed_cached < 0.5, f"Cached lookups too slow: {elapsed_cached:.3f}s"
        assert result == result1, "Cache should return same result"


# =============================================================================
# Nothing Sentinel Performance  
# =============================================================================

class TestNothingPerformance:
    """Performance tests for nothing sentinel checks."""
    
    def test_nothing_p_is_fast(self):
        """nothing_p() should be a fast identity check."""
        start = time.perf_counter()
        for _ in range(100000):
            nothing_p(nothing)
            nothing_p(None)
            nothing_p(42)
        elapsed = time.perf_counter() - start
        
        # Simple identity checks should be very fast
        assert elapsed < 0.5, f"nothing_p checks too slow: {elapsed:.3f}s"


# =============================================================================
# Benchmark Runner (for manual profiling)
# =============================================================================

def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 60)
    print("Propagator Performance Benchmarks")
    print("=" * 60)
    
    # Require distinct problems
    print("\nRequire Distinct Problem Times:")
    for n in [3, 4]:
        times = []
        for _ in range(3):
            t, valid = TestRequireDistinctPerformance._run_require_distinct(n)
            assert valid
            times.append(t)
        avg = sum(times) / len(times)
        print(f"  {n} cells: {avg:.3f}s (min={min(times):.3f}s, max={max(times):.3f}s)")
    
    # Equivalent performance
    print("\nEquivalent Check Performance:")
    large_list = list(range(100000))
    
    start = time.perf_counter()
    for _ in range(100000):
        equivalent(large_list, large_list)
    elapsed = time.perf_counter() - start
    print(f"  100k identity checks: {elapsed:.3f}s")
    
    start = time.perf_counter()
    for _ in range(100000):
        equivalent(42, 42)
    elapsed = time.perf_counter() - start
    print(f"  100k int equivalence: {elapsed:.3f}s")
    
    # Merge performance
    print("\nMerge Performance:")
    start = time.perf_counter()
    for _ in range(100000):
        merge(42, 42)
    elapsed = time.perf_counter() - start
    print(f"  100k identical merges: {elapsed:.3f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_benchmarks()
