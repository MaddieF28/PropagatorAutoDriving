"""Pytest coverage for a minimal distinct-values puzzle.

Simpler than multiple-dwelling; tests the require_distinct search loop
without extra constraints. Also tests guessing_machine configuration options.
"""

import pytest

from propagator import initialize_scheduler, run
from propagator.cell import Cell
from propagator import guessing_machine
from propagator.guessing_machine import one_of, require_distinct
from propagator.tms import tms_query


def test_require_distinct_finds_assignment():
    """Motivation: 4 variables from 4 values should produce a permutation."""
    initialize_scheduler()

    cells = [Cell() for _ in range(4)]
    for c in cells:
        one_of([1, 2, 3, 4], c)
    run()

    require_distinct(cells)
    run()

    values = []
    for c in cells:
        q = tms_query(c.content)
        val = q.value if hasattr(q, "value") else q
        values.append(val)

    assert len(set(values)) == 4
    assert set(values) == {1, 2, 3, 4}


# =============================================================================
# Avoid False-True Flips Configuration Tests
# =============================================================================

class TestAvoidFalseTrueFlips:
    """Tests for the avoid-false-true-flips optimization."""
    
    def test_flag_exists(self):
        """The configuration flag should exist with correct default."""
        # Default is False (matching Scheme)
        assert guessing_machine._avoid_false_true_flips is False
    
    def test_amb_works_with_flag_disabled(self):
        """AMB should work correctly with the flag disabled (default)."""
        initialize_scheduler()
        cells = [Cell() for _ in range(3)]
        for cell in cells:
            one_of([1, 2, 3], cell)
        require_distinct(cells)
        run()
        
        # Should find a valid solution
        values = [
            c.content.base_value if hasattr(c.content, 'base_value') else c.content
            for c in cells
        ]
        assert len(set(values)) == 3  # All distinct
    
    def test_amb_works_with_flag_enabled(self):
        """AMB should work correctly with the flag enabled."""
        # Enable the optimization
        original = guessing_machine._avoid_false_true_flips
        guessing_machine._avoid_false_true_flips = True
        
        try:
            initialize_scheduler()
            cells = [Cell() for _ in range(3)]
            for cell in cells:
                one_of([1, 2, 3], cell)
            require_distinct(cells)
            run()
            
            # Should still find a valid solution
            values = [
                c.content.base_value if hasattr(c.content, 'base_value') else c.content
                for c in cells
            ]
            assert len(set(values)) == 3  # All distinct
        finally:
            # Restore original setting
            guessing_machine._avoid_false_true_flips = original
