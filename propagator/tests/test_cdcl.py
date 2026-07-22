#!/usr/bin/env python3
"""
Test suite for CDCL implementation.

Verifies:
1. CDCL produces correct solutions (same as DDB when deterministic)
2. Both methods find valid solutions for CSP problems
3. CDCL statistics are tracked correctly
4. API functions work as expected

Run with:
    python3 -m propagator.tests.test_cdcl
"""

import sys


def test_cdcl_api():
    """Test the CDCL enable/disable API."""
    from propagator import (
        initialize_scheduler, enable_cdcl, disable_cdcl, 
        cdcl_enabled, get_cdcl_engine, reset_cdcl, full_reset_cdcl,
    )
    
    # Fresh start
    initialize_scheduler()
    
    # Should be disabled by default
    assert not cdcl_enabled(), "CDCL should be disabled by default"
    
    # Enable
    engine = enable_cdcl()
    assert cdcl_enabled(), "CDCL should be enabled after enable_cdcl()"
    assert engine is get_cdcl_engine(), "enable_cdcl should return the engine"
    
    # Disable
    disable_cdcl()
    assert not cdcl_enabled(), "CDCL should be disabled after disable_cdcl()"
    
    # Reset functions
    enable_cdcl()
    engine.stats.conflicts = 100
    reset_cdcl()
    assert engine.stats.conflicts == 0, "reset_cdcl should reset stats"
    
    engine._activities[123] = 5.0
    full_reset_cdcl()
    assert len(engine._activities) == 0, "full_reset_cdcl should reset activities"
    
    print("✓ test_cdcl_api passed")


def test_simple_constraint():
    """Test a simple constraint with both methods."""
    from propagator import (
        Cell, initialize_scheduler, enable_cdcl, disable_cdcl,
        get_contradictions,
    )
    from propagator.primitives import constant, eq
    from propagator.guessing_machine import one_of, abhor
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    # Test with DDB
    initialize_scheduler()
    disable_cdcl()
    
    x = Cell(name='x')
    one_of([1, 2, 3], x)
    
    # Forbid 1 and 2
    eq1 = Cell()
    one = Cell()
    constant(1, one)
    eq(x, one, eq1)
    abhor(eq1)
    
    eq2 = Cell()
    two = Cell()
    constant(2, two)
    eq(x, two, eq2)
    abhor(eq2)
    
    run()
    
    content = x.content
    ddb_result = tms_query(content)
    ddb_value = ddb_result.value if hasattr(ddb_result, 'value') else ddb_result
    ddb_conflicts = len(get_contradictions())
    
    # Test with CDCL
    initialize_scheduler()
    enable_cdcl()
    
    x = Cell(name='x')
    one_of([1, 2, 3], x)
    
    eq1 = Cell()
    one = Cell()
    constant(1, one)
    eq(x, one, eq1)
    abhor(eq1)
    
    eq2 = Cell()
    two = Cell()
    constant(2, two)
    eq(x, two, eq2)
    abhor(eq2)
    
    run()
    
    content = x.content
    cdcl_result = tms_query(content)
    cdcl_value = cdcl_result.value if hasattr(cdcl_result, 'value') else cdcl_result
    cdcl_conflicts = len(get_contradictions())
    
    # Both should find x = 3
    assert ddb_value == 3, f"DDB should find x=3, got {ddb_value}"
    assert cdcl_value == 3, f"CDCL should find x=3, got {cdcl_value}"
    
    print(f"✓ test_simple_constraint passed (DDB: {ddb_conflicts} conflicts, CDCL: {cdcl_conflicts} conflicts)")


def test_multiple_dwelling_correctness():
    """Test that both methods solve Multiple Dwelling correctly."""
    from propagator import (
        Cell, initialize_scheduler, enable_cdcl, disable_cdcl,
    )
    from propagator.primitives import eq, constant, gt, subtractor, absolute_value
    from propagator.guessing_machine import one_of, require, abhor, require_distinct
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    def setup_multiple_dwelling():
        baker = Cell(name='baker')
        cooper = Cell(name='cooper')
        fletcher = Cell(name='fletcher')
        miller = Cell(name='miller')
        smith = Cell(name='smith')
        
        floors = [1, 2, 3, 4, 5]
        
        one_of(floors, baker)
        one_of(floors, fletcher)
        one_of(floors, smith)
        one_of(floors, cooper)
        one_of(floors, miller)
        
        require_distinct([baker, fletcher, smith, cooper, miller])
        
        # Constraints
        five = Cell(); constant(5, five)
        one = Cell(); constant(1, one)
        
        b_eq_5 = Cell(); eq(five, baker, b_eq_5); abhor(b_eq_5)
        c_eq_1 = Cell(); eq(one, cooper, c_eq_1); abhor(c_eq_1)
        f_eq_5 = Cell(); eq(five, fletcher, f_eq_5); abhor(f_eq_5)
        f_eq_1 = Cell(); eq(one, fletcher, f_eq_1); abhor(f_eq_1)
        
        m_gt_c = Cell(); gt(miller, cooper, m_gt_c); require(m_gt_c)
        
        s_f = Cell(); subtractor(smith, fletcher, s_f)
        as_f = Cell(); absolute_value(s_f, as_f)
        sf = Cell(); eq(one, as_f, sf); abhor(sf)
        
        f_c = Cell(); subtractor(fletcher, cooper, f_c)
        af_c = Cell(); absolute_value(f_c, af_c)
        fc = Cell(); eq(one, af_c, fc); abhor(fc)
        
        return [baker, cooper, fletcher, miller, smith]
    
    def extract_solution(cells):
        solution = {}
        names = ['baker', 'cooper', 'fletcher', 'miller', 'smith']
        for name, cell in zip(names, cells):
            content = cell.content
            if tms_p(content):
                result = tms_query(content)
                if result is not None:
                    val = result.value if hasattr(result, 'value') else result
                    solution[name] = val
        return solution
    
    def is_valid_solution(sol):
        """Verify the solution satisfies all constraints."""
        if len(sol) != 5:
            return False
        
        b, c, f, m, s = sol['baker'], sol['cooper'], sol['fletcher'], sol['miller'], sol['smith']
        
        # All different
        if len(set([b, c, f, m, s])) != 5:
            return False
        
        # All in 1-5
        if not all(1 <= v <= 5 for v in [b, c, f, m, s]):
            return False
        
        # Baker != 5
        if b == 5:
            return False
        
        # Cooper != 1
        if c == 1:
            return False
        
        # Fletcher != 1, 5
        if f == 1 or f == 5:
            return False
        
        # Miller > Cooper
        if not m > c:
            return False
        
        # Smith not adjacent to Fletcher
        if abs(s - f) == 1:
            return False
        
        # Fletcher not adjacent to Cooper
        if abs(f - c) == 1:
            return False
        
        return True
    
    # Test DDB
    initialize_scheduler()
    disable_cdcl()
    cells = setup_multiple_dwelling()
    run()
    ddb_solution = extract_solution(cells)
    
    # Test CDCL
    initialize_scheduler()
    enable_cdcl()
    cells = setup_multiple_dwelling()
    run()
    cdcl_solution = extract_solution(cells)
    
    assert is_valid_solution(ddb_solution), f"DDB solution invalid: {ddb_solution}"
    assert is_valid_solution(cdcl_solution), f"CDCL solution invalid: {cdcl_solution}"
    
    print(f"✓ test_multiple_dwelling_correctness passed")
    print(f"  DDB solution: {ddb_solution}")
    print(f"  CDCL solution: {cdcl_solution}")


def test_cdcl_statistics():
    """Test that CDCL statistics are tracked correctly."""
    from propagator import (
        Cell, initialize_scheduler, enable_cdcl, get_cdcl_engine,
        cdcl_stats, cdcl_conflicts, cdcl_backjumps,
    )
    from propagator.guessing_machine import one_of, require_distinct
    from propagator.scheduler import run
    
    initialize_scheduler()
    engine = enable_cdcl()
    
    # Initial stats should be zero
    assert engine.stats.conflicts == 0
    assert engine.stats.decisions == 0
    assert engine.stats.backjumps == 0
    
    # Create a small problem that will have conflicts
    x = Cell(name='x')
    y = Cell(name='y')
    z = Cell(name='z')
    
    one_of([1, 2], x)
    one_of([1, 2], y)
    one_of([1, 2], z)
    
    require_distinct([x, y, z])  # Impossible with only 2 values for 3 vars
    
    run()  # Will explore and find contradictions
    
    # There should be some stats recorded
    stats_report = cdcl_stats()
    assert "CDCL Statistics" in stats_report
    
    print("✓ test_cdcl_statistics passed")
    print(f"  Stats report:\n{stats_report}")


def test_4queens_valid_solutions():
    """Test that both methods find valid 4-queens solutions."""
    from propagator import (
        Cell, initialize_scheduler, enable_cdcl, disable_cdcl,
    )
    from propagator.primitives import constant, eq, subtractor, absolute_value
    from propagator.guessing_machine import one_of, abhor, require_distinct
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    def setup_nqueens(n):
        queens = [Cell(name=f'queen_{i}') for i in range(n)]
        columns = list(range(n))
        
        for q in queens:
            one_of(columns, q)
        
        require_distinct(queens)
        
        for i in range(n):
            for j in range(i + 1, n):
                row_diff = j - i
                col_diff = Cell()
                abs_col_diff = Cell()
                row_const = Cell()
                diag_eq = Cell()
                
                subtractor(queens[i], queens[j], col_diff)
                absolute_value(col_diff, abs_col_diff)
                constant(row_diff, row_const)
                eq(row_const, abs_col_diff, diag_eq)
                abhor(diag_eq)
        
        return queens
    
    def extract_solution(queens):
        solution = []
        for q in queens:
            content = q.content
            if tms_p(content):
                result = tms_query(content)
                if result is not None:
                    val = result.value if hasattr(result, 'value') else result
                    solution.append(val)
        return solution
    
    def is_valid_nqueens(cols, n):
        if len(cols) != n:
            return False
        if len(set(cols)) != n:
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if abs(cols[i] - cols[j]) == j - i:
                    return False
        return True
    
    # Test DDB
    initialize_scheduler()
    disable_cdcl()
    queens = setup_nqueens(4)
    run()
    ddb_solution = extract_solution(queens)
    
    # Test CDCL
    initialize_scheduler()
    enable_cdcl()
    queens = setup_nqueens(4)
    run()
    cdcl_solution = extract_solution(queens)
    
    assert is_valid_nqueens(ddb_solution, 4), f"DDB 4-queens invalid: {ddb_solution}"
    assert is_valid_nqueens(cdcl_solution, 4), f"CDCL 4-queens invalid: {cdcl_solution}"
    
    print(f"✓ test_4queens_valid_solutions passed")
    print(f"  DDB solution: {ddb_solution}")
    print(f"  CDCL solution: {cdcl_solution}")


def run_all_tests():
    """Run all CDCL tests."""
    print("=" * 60)
    print("CDCL Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_cdcl_api,
        test_simple_constraint,
        test_cdcl_statistics,
        test_4queens_valid_solutions,
        test_multiple_dwelling_correctness,  # This one takes longer
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
