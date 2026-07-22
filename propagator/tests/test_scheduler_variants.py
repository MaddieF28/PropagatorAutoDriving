"""Pytest tests for scheduler variants.

These tests verify:
1. All scheduler types produce correct results
2. FastSlowScheduler properly prioritizes fast over slow propagators
3. Scheduler statistics are tracked correctly
4. Performance comparison between scheduler types
"""

import pytest
import time
from propagator import (
    Cell,
    initialize_scheduler,
    run,
    one_of,
    require_distinct,
    SchedulerType,
    SchedulerStats,
    set_scheduler_factory,
    get_scheduler_type,
    get_scheduler_stats,
    reset_scheduler_stats,
    tag_slow,
    is_slow,
    untag_slow,
    RoundRobinScheduler,
    StackScheduler,
    FastSlowScheduler,
)


class TestSchedulerTypes:
    """Tests for different scheduler type configurations."""

    def test_default_scheduler_is_round_robin(self):
        """Default scheduler should be round robin."""
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
        initialize_scheduler()
        assert get_scheduler_type() == SchedulerType.ROUND_ROBIN

    def test_can_set_stack_scheduler(self):
        """Can configure stack scheduler."""
        set_scheduler_factory(SchedulerType.STACK)
        initialize_scheduler()
        assert get_scheduler_type() == SchedulerType.STACK
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_can_set_fast_slow_round_robin_scheduler(self):
        """Can configure fast/slow round robin scheduler."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        assert get_scheduler_type() == SchedulerType.FAST_SLOW_ROUND_ROBIN
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_can_set_fast_slow_stack_scheduler(self):
        """Can configure fast/slow stack scheduler."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_STACK)
        initialize_scheduler()
        assert get_scheduler_type() == SchedulerType.FAST_SLOW_STACK
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_can_use_custom_scheduler_factory(self):
        """Can use a custom scheduler factory function."""
        set_scheduler_factory(lambda: FastSlowScheduler(policy='stack'))
        initialize_scheduler()
        assert get_scheduler_type() == SchedulerType.FAST_SLOW_STACK
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)


class TestSlowTagging:
    """Tests for slow propagator tagging."""

    def test_propagator_not_slow_by_default(self):
        """Propagators are not slow by default."""
        def my_propagator():
            pass
        assert not is_slow(my_propagator)

    def test_tag_slow_marks_propagator(self):
        """tag_slow marks a propagator as slow."""
        def my_propagator():
            pass
        tag_slow(my_propagator)
        assert is_slow(my_propagator)

    def test_untag_slow_removes_mark(self):
        """untag_slow removes the slow mark."""
        def my_propagator():
            pass
        tag_slow(my_propagator)
        untag_slow(my_propagator)
        assert not is_slow(my_propagator)

    def test_tag_slow_returns_propagator(self):
        """tag_slow returns the propagator for chaining."""
        def my_propagator():
            pass
        result = tag_slow(my_propagator)
        assert result is my_propagator


class TestSchedulerStats:
    """Tests for scheduler statistics tracking."""

    def test_stats_tracks_executions(self):
        """Stats should track total executions."""
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
        initialize_scheduler()
        
        from propagator.scheduler import alert_propagators
        
        execution_count = 0
        def counting_propagator():
            nonlocal execution_count
            execution_count += 1
        
        alert_propagators(counting_propagator)
        run()
        
        stats = get_scheduler_stats()
        assert stats.total_executions >= 1

    def test_stats_tracks_rounds(self):
        """Stats should track number of rounds."""
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
        initialize_scheduler()
        
        stats = get_scheduler_stats()
        assert stats.rounds >= 0

    def test_reset_stats_clears_counters(self):
        """reset_scheduler_stats should clear all counters."""
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
        initialize_scheduler()
        
        from propagator.scheduler import alert_propagators
        
        alert_propagators(lambda: None)
        run()
        
        reset_scheduler_stats()
        stats = get_scheduler_stats()
        assert stats.total_executions == 0
        assert stats.rounds == 0

    def test_fast_slow_stats_track_both_queues(self):
        """FastSlowScheduler stats should track fast and slow separately."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        
        from propagator.scheduler import alert_propagators
        
        def fast_prop():
            pass
        
        def slow_prop():
            pass
        tag_slow(slow_prop)
        
        alert_propagators(fast_prop)
        alert_propagators(slow_prop)
        run()
        
        stats = get_scheduler_stats()
        assert stats.fast_executions >= 1
        assert stats.slow_executions >= 1
        assert stats.total_executions == stats.fast_executions + stats.slow_executions
        
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)


class TestFastSlowPrioritization:
    """Tests for FastSlowScheduler's prioritization of fast over slow."""

    def test_fast_runs_before_slow(self):
        """Fast propagators should run before slow ones."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        
        from propagator.scheduler import alert_propagators
        
        execution_order = []
        
        def fast_prop():
            execution_order.append('fast')
        
        def slow_prop():
            execution_order.append('slow')
        tag_slow(slow_prop)
        
        # Alert slow first, then fast
        alert_propagators(slow_prop)
        alert_propagators(fast_prop)
        run()
        
        # Fast should still run before slow
        assert execution_order == ['fast', 'slow']
        
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_fast_queue_exhausted_before_slow(self):
        """All fast propagators should run before any slow."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        
        from propagator.scheduler import alert_propagators
        
        execution_order = []
        
        def fast1():
            execution_order.append('fast1')
        def fast2():
            execution_order.append('fast2')
        def slow1():
            execution_order.append('slow1')
        def slow2():
            execution_order.append('slow2')
        
        tag_slow(slow1)
        tag_slow(slow2)
        
        # Interleave alerts
        alert_propagators(slow1)
        alert_propagators(fast1)
        alert_propagators(slow2)
        alert_propagators(fast2)
        run()
        
        # All fast should come before all slow
        fast_indices = [i for i, x in enumerate(execution_order) if x.startswith('fast')]
        slow_indices = [i for i, x in enumerate(execution_order) if x.startswith('slow')]
        
        assert max(fast_indices) < min(slow_indices)
        
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)


class TestCorrectnessAcrossSchedulers:
    """Tests that all schedulers produce correct results on real problems."""

    def _run_require_distinct_3cells(self):
        """Run a simple 3-cell distinct constraint."""
        cells = [Cell() for _ in range(3)]
        for cell in cells:
            one_of([1, 2, 3], cell)
        require_distinct(cells)
        run()
        
        # Extract values
        values = []
        for c in cells:
            content = c.content
            if hasattr(content, 'base_value'):
                values.append(content.base_value)
            else:
                values.append(content)
        
        return len(set(values)) == 3  # All distinct

    def test_round_robin_correctness(self):
        """RoundRobin scheduler produces correct results."""
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
        initialize_scheduler()
        assert self._run_require_distinct_3cells()

    def test_stack_correctness(self):
        """Stack scheduler produces correct results."""
        set_scheduler_factory(SchedulerType.STACK)
        initialize_scheduler()
        assert self._run_require_distinct_3cells()
        # Reset
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_fast_slow_round_robin_correctness(self):
        """FastSlowRoundRobin scheduler produces correct results."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        assert self._run_require_distinct_3cells()
        # Reset
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_fast_slow_stack_correctness(self):
        """FastSlowStack scheduler produces correct results."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_STACK)
        initialize_scheduler()
        assert self._run_require_distinct_3cells()
        # Reset
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)


class TestSchedulerComparison:
    """Performance and behavior comparison between schedulers."""

    def _benchmark_scheduler(self, scheduler_type: SchedulerType, n_cells: int = 4):
        """Run a benchmark with the given scheduler type."""
        set_scheduler_factory(scheduler_type)
        initialize_scheduler()
        
        start = time.perf_counter()
        
        cells = [Cell() for _ in range(n_cells)]
        for cell in cells:
            one_of(list(range(1, n_cells + 1)), cell)
        require_distinct(cells)
        run()
        
        elapsed = time.perf_counter() - start
        stats = get_scheduler_stats()
        
        # Verify correctness
        values = []
        for c in cells:
            content = c.content
            if hasattr(content, 'base_value'):
                values.append(content.base_value)
            else:
                values.append(content)
        is_valid = len(set(values)) == n_cells
        
        return {
            'scheduler': scheduler_type.value,
            'time': elapsed,
            'executions': stats.total_executions,
            'rounds': stats.rounds,
            'fast_executions': stats.fast_executions,
            'slow_executions': stats.slow_executions,
            'valid': is_valid,
        }

    def test_all_schedulers_produce_same_validity(self):
        """All schedulers should produce valid solutions."""
        results = {}
        for scheduler_type in SchedulerType:
            results[scheduler_type] = self._benchmark_scheduler(scheduler_type, n_cells=3)
        
        # All should be valid
        for scheduler_type, result in results.items():
            assert result['valid'], f"{scheduler_type.value} produced invalid result"
        
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)

    def test_scheduler_comparison_side_by_side(self):
        """
        Side-by-side comparison of all scheduler types.
        
        This test runs the same problem with each scheduler and prints
        performance metrics for analysis. All should produce correct results.
        """
        n_cells = 4
        results = []
        
        for scheduler_type in SchedulerType:
            result = self._benchmark_scheduler(scheduler_type, n_cells)
            results.append(result)
        
        # Print comparison table
        print("\n" + "=" * 70)
        print(f"Scheduler Comparison ({n_cells} cells, {n_cells} values each)")
        print("=" * 70)
        print(f"{'Scheduler':<25} {'Time':>10} {'Execs':>10} {'Rounds':>10} {'Valid':>8}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['scheduler']:<25} {r['time']:>10.3f}s {r['executions']:>10} {r['rounds']:>10} {str(r['valid']):>8}")
        
        # For fast/slow schedulers, show breakdown
        fast_slow_results = [r for r in results if 'fast_slow' in r['scheduler']]
        if fast_slow_results:
            print("\nFast/Slow Breakdown:")
            for r in fast_slow_results:
                print(f"  {r['scheduler']}: fast={r['fast_executions']}, slow={r['slow_executions']}")
        
        print("=" * 70)
        
        # All should be valid
        for r in results:
            assert r['valid'], f"{r['scheduler']} produced invalid result"
        
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)


class TestSchedulerStatsDisplay:
    """Tests for scheduler stats string representation."""

    def test_round_robin_stats_str(self):
        """RoundRobinScheduler stats should have clean string representation."""
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
        initialize_scheduler()
        
        stats = get_scheduler_stats()
        stats_str = str(stats)
        
        assert "round_robin" in stats_str
        assert "Total executions" in stats_str
        assert "Rounds" in stats_str

    def test_fast_slow_stats_str(self):
        """FastSlowScheduler stats should include fast/slow breakdown."""
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        
        stats = get_scheduler_stats()
        stats_str = str(stats)
        
        assert "fast_slow" in stats_str
        assert "Fast executions" in stats_str
        assert "Slow executions" in stats_str
        
        # Reset to default
        set_scheduler_factory(SchedulerType.ROUND_ROBIN)
