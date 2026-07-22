#!/usr/bin/env python3
"""
Test each execution approach for propagator correctness.

This tests that all solving approaches produce valid, correct solutions
for canonical constraint satisfaction problems.

Approaches tested:
1. Propagator with Dependency-Directed Backtracking (DDB)
2. Propagator with CDCL enhancement
3. Direct SMT (z3)
4. Translated SMT (propagator → SMT translation)
5. Hybrid Propagator+SMT
"""

import pytest
from typing import Dict, Optional, Tuple


# =============================================================================
# Validation Functions
# =============================================================================

def validate_nqueens(n: int, solution: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    """Validate an N-Queens solution."""
    if not solution:
        return False, "Empty solution"
    
    # Extract queen positions
    queens = []
    for i in range(n):
        key = f'Q{i}'
        if key not in solution:
            return False, f"Missing queen {key}"
        queens.append(solution[key])
    
    # Check all queens have valid row values
    for i, q in enumerate(queens):
        if not (0 <= q < n):
            return False, f"Q{i} has invalid row {q}"
    
    # Check no two queens in same row
    if len(set(queens)) != n:
        return False, "Two queens in same row"
    
    # Check diagonals
    for i in range(n):
        for j in range(i + 1, n):
            row_diff = abs(queens[i] - queens[j])
            col_diff = j - i
            if row_diff == col_diff:
                return False, f"Q{i} and Q{j} on same diagonal"
    
    return True, None


def validate_dwelling(solution: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    """Validate a Multiple Dwelling solution."""
    if not solution:
        return False, "Empty solution"
    
    required = ['baker', 'cooper', 'fletcher', 'miller', 'smith']
    for name in required:
        if name not in solution:
            return False, f"Missing {name}"
    
    baker = solution['baker']
    cooper = solution['cooper']
    fletcher = solution['fletcher']
    miller = solution['miller']
    smith = solution['smith']
    
    # All different floors 1-5
    floors = [baker, cooper, fletcher, miller, smith]
    if set(floors) != {1, 2, 3, 4, 5}:
        return False, f"Floors must be 1-5, got {floors}"
    
    # Baker != 5
    if baker == 5:
        return False, "Baker on top floor"
    
    # Cooper != 1
    if cooper == 1:
        return False, "Cooper on bottom floor"
    
    # Fletcher != 1 and != 5
    if fletcher in (1, 5):
        return False, "Fletcher on top or bottom"
    
    # Miller > Cooper
    if miller <= cooper:
        return False, "Miller not above Cooper"
    
    # Smith not adjacent to Fletcher
    if abs(smith - fletcher) == 1:
        return False, "Smith adjacent to Fletcher"
    
    # Fletcher not adjacent to Cooper
    if abs(fletcher - cooper) == 1:
        return False, "Fletcher adjacent to Cooper"
    
    return True, None


def validate_send_more_money(solution: Dict[str, int]) -> Tuple[bool, Optional[str]]:
    """Validate SEND + MORE = MONEY."""
    if not solution:
        return False, "Empty solution"
    
    required = ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']
    for letter in required:
        if letter not in solution:
            return False, f"Missing {letter}"
    
    S, E, N, D = solution['S'], solution['E'], solution['N'], solution['D']
    M, O, R, Y = solution['M'], solution['O'], solution['R'], solution['Y']
    
    # All different
    values = [S, E, N, D, M, O, R, Y]
    if len(set(values)) != 8:
        return False, "Not all different"
    
    # All 0-9
    for v in values:
        if not (0 <= v <= 9):
            return False, f"Value {v} out of range"
    
    # S and M non-zero
    if S == 0:
        return False, "S is zero"
    if M == 0:
        return False, "M is zero"
    
    # Arithmetic
    send = 1000*S + 100*E + 10*N + D
    more = 1000*M + 100*O + 10*R + E
    money = 10000*M + 1000*O + 100*N + 10*E + Y
    
    if send + more != money:
        return False, f"{send} + {more} != {money}"
    
    return True, None


# =============================================================================
# Approach 1: Propagator with DDB
# =============================================================================

class TestPropagatorDDB:
    """Test propagator solving with Dependency-Directed Backtracking."""
    
    def test_nqueens_4_ddb(self):
        """4-Queens with DDB - uses existing test infrastructure."""
        from propagator.examples.performance.benchmark_solver_approaches import run_propagator_nqueens
        
        result = run_propagator_nqueens(4, use_cdcl=False)
        assert result.correct, f"Solution invalid: {result.error}"
    
    def test_dwelling_ddb(self):
        """Multiple Dwelling with DDB."""
        from propagator.examples.performance.benchmark_solver_approaches import run_propagator_dwelling
        
        result = run_propagator_dwelling(use_cdcl=False)
        assert result.correct, f"Solution invalid: {result.error}"


# =============================================================================
# Approach 2: Propagator with CDCL
# =============================================================================

class TestPropagatorCDCL:
    """Test propagator solving with CDCL enhancement."""
    
    def test_nqueens_4_cdcl(self):
        """4-Queens with CDCL."""
        from propagator.examples.performance.benchmark_solver_approaches import run_propagator_nqueens
        
        result = run_propagator_nqueens(4, use_cdcl=True)
        assert result.correct, f"Solution invalid: {result.error}"
    
    def test_dwelling_cdcl(self):
        """Multiple Dwelling with CDCL."""
        from propagator.examples.performance.benchmark_solver_approaches import run_propagator_dwelling
        
        result = run_propagator_dwelling(use_cdcl=True)
        assert result.correct, f"Solution invalid: {result.error}"


# =============================================================================
# Approach 3: Direct SMT (z3)
# =============================================================================

class TestDirectSMT:
    """Test direct SMT solving with Z3."""
    
    @pytest.fixture(autouse=True)
    def check_z3(self):
        """Skip if z3 not installed."""
        pytest.importorskip("z3")
    
    def test_nqueens_4_smt(self):
        """4-Queens with direct SMT."""
        from z3 import Int, Solver, Distinct, Abs, And, sat
        
        n = 4
        queens = [Int(f'Q{i}') for i in range(n)]
        
        solver = Solver()
        
        # Domain: 0 to n-1
        for q in queens:
            solver.add(And(q >= 0, q < n))
        
        # All different rows
        solver.add(Distinct(queens))
        
        # No diagonals
        for i in range(n):
            for j in range(i + 1, n):
                solver.add(Abs(queens[i] - queens[j]) != j - i)
        
        assert solver.check() == sat
        
        model = solver.model()
        solution = {f'Q{i}': model[queens[i]].as_long() for i in range(n)}
        
        valid, error = validate_nqueens(4, solution)
        assert valid, f"Invalid solution: {error}"
    
    def test_dwelling_smt(self):
        """Multiple Dwelling with direct SMT."""
        from z3 import Int, Solver, Distinct, Abs, And, sat
        
        solver = Solver()
        
        baker = Int('baker')
        cooper = Int('cooper')
        fletcher = Int('fletcher')
        miller = Int('miller')
        smith = Int('smith')
        people = [baker, cooper, fletcher, miller, smith]
        
        # Floors 1-5
        for p in people:
            solver.add(And(p >= 1, p <= 5))
        
        solver.add(Distinct(people))
        solver.add(baker != 5)
        solver.add(cooper != 1)
        solver.add(fletcher != 1)
        solver.add(fletcher != 5)
        solver.add(miller > cooper)
        solver.add(Abs(smith - fletcher) != 1)
        solver.add(Abs(fletcher - cooper) != 1)
        
        assert solver.check() == sat
        
        model = solver.model()
        solution = {
            'baker': model[baker].as_long(),
            'cooper': model[cooper].as_long(),
            'fletcher': model[fletcher].as_long(),
            'miller': model[miller].as_long(),
            'smith': model[smith].as_long(),
        }
        
        valid, error = validate_dwelling(solution)
        assert valid, f"Invalid solution: {error}"
    
    def test_send_more_money_smt(self):
        """SEND+MORE=MONEY with direct SMT."""
        from z3 import Int, Solver, Distinct, And, sat
        
        solver = Solver()
        
        S = Int('S')
        E = Int('E')
        N = Int('N')
        D = Int('D')
        M = Int('M')
        O = Int('O')
        R = Int('R')
        Y = Int('Y')
        letters = [S, E, N, D, M, O, R, Y]
        
        for l in letters:
            solver.add(And(l >= 0, l <= 9))
        
        solver.add(Distinct(letters))
        solver.add(S != 0)
        solver.add(M != 0)
        
        # SEND + MORE = MONEY
        send = 1000*S + 100*E + 10*N + D
        more = 1000*M + 100*O + 10*R + E
        money = 10000*M + 1000*O + 100*N + 10*E + Y
        solver.add(send + more == money)
        
        assert solver.check() == sat
        
        model = solver.model()
        solution = {
            'S': model[S].as_long(),
            'E': model[E].as_long(),
            'N': model[N].as_long(),
            'D': model[D].as_long(),
            'M': model[M].as_long(),
            'O': model[O].as_long(),
            'R': model[R].as_long(),
            'Y': model[Y].as_long(),
        }
        
        valid, error = validate_send_more_money(solution)
        assert valid, f"Invalid solution: {error}"


# =============================================================================
# Approach 4: Translated SMT
# =============================================================================

class TestTranslatedSMT:
    """Test propagator to SMT translation.
    
    NOTE: The PropagatorTranslator is an incomplete/experimental feature.
    These tests are marked as expected failures until the translator is complete.
    """
    
    @pytest.fixture(autouse=True)
    def check_z3(self):
        """Skip if z3 not installed."""
        pytest.importorskip("z3")
    
    # @pytest.mark.xfail(reason="PropagatorTranslator incomplete - doesn't encode all constraints")
    def test_nqueens_4_translated(self):
        """4-Queens via propagator → SMT translation."""
        from propagator.examples.performance.benchmark_solver_approaches import run_translated_smt_nqueens
        
        result = run_translated_smt_nqueens(4)
        assert result.correct, f"Invalid solution: {result.error}"
    
    # @pytest.mark.xfail(reason="PropagatorTranslator incomplete - doesn't encode all constraints")
    def test_dwelling_translated(self):
        """Multiple Dwelling via propagator → SMT translation."""
        from propagator.examples.performance.benchmark_solver_approaches import run_translated_smt_dwelling
        
        result = run_translated_smt_dwelling()
        assert result.correct, f"Invalid solution: {result.error}"
    
    # @pytest.mark.xfail(reason="PropagatorTranslator incomplete - doesn't encode all constraints")
    def test_send_more_money_translated(self):
        """SEND+MORE=MONEY via propagator → SMT translation."""
        from propagator.examples.performance.benchmark_solver_approaches import run_translated_smt_send_more_money
        
        result = run_translated_smt_send_more_money()
        assert result.correct, f"Invalid solution: {result.error}"


# =============================================================================
# Approach 5: Hybrid Propagator + SMT
# =============================================================================

class TestHybridApproach:
    """Test hybrid propagator + SMT solving."""
    
    @pytest.fixture(autouse=True)
    def check_z3(self):
        """Skip if z3 not installed."""
        pytest.importorskip("z3")
    
    def test_nqueens_4_hybrid(self):
        """4-Queens with hybrid approach."""
        from propagator.examples.performance.benchmark_solver_approaches import run_hybrid_nqueens
        
        result = run_hybrid_nqueens(4)
        assert result.correct, f"Invalid solution: {result.error}"
    
    def test_dwelling_hybrid(self):
        """Multiple Dwelling with hybrid approach."""
        from propagator.examples.performance.benchmark_solver_approaches import run_hybrid_dwelling
        
        result = run_hybrid_dwelling()
        assert result.correct, f"Invalid solution: {result.error}"
    
    def test_send_more_money_hybrid(self):
        """SEND+MORE=MONEY with hybrid approach."""
        from propagator.examples.performance.benchmark_solver_approaches import run_hybrid_send_more_money

        result = run_hybrid_send_more_money()
        assert result.correct, f"Invalid solution: {result.error}"


# =============================================================================
# Approach 5b: Roots-First (solve an ordinarily-built network from its roots)
# =============================================================================

class TestRootsFirstApproach:
    """Test solving ordinarily-built propagator networks via solve_from_roots.

    These go through the benchmark's solve_with_roots_first, which wires each
    problem with the exact same _build_propagator_network used by the native
    Propagator-DDB approach -- no translation-only rewiring -- then hands the
    root cells to solve_from_roots.
    """

    @pytest.fixture(autouse=True)
    def check_z3(self):
        """Skip if z3 not installed."""
        pytest.importorskip("z3")

    def test_nqueens_4_roots_first(self):
        """4-Queens: network self-solves during construction (default auto-run),
        then solve_from_roots re-derives a valid solution from the roots."""
        from propagator.examples.performance.benchmark_solver_approaches import run_roots_first_nqueens

        result = run_roots_first_nqueens(4)
        assert result.correct, f"Invalid solution: {result.error}"

    def test_nqueens_4_roots_first_cold(self):
        """4-Queens with native search deferred: solve_from_roots does all the solving."""
        from propagator.examples.performance.benchmark_solver_approaches import run_roots_first_cold_nqueens

        result = run_roots_first_cold_nqueens(4)
        assert result.correct, f"Invalid solution: {result.error}"

    def test_dwelling_roots_first_cold(self):
        """Multiple Dwelling, cold: the case where roots-first beats native
        search by orders of magnitude (native DDB takes seconds here)."""
        from propagator.examples.performance.benchmark_solver_approaches import run_roots_first_cold_dwelling

        result = run_roots_first_cold_dwelling()
        assert result.correct, f"Invalid solution: {result.error}"

    def test_send_more_money_roots_first_cold(self):
        """SEND+MORE=MONEY, cold: impractical for native propagator search
        (COLUMN_ADD x4), but the same wiring solves via roots-first SMT."""
        from propagator.examples.performance.benchmark_solver_approaches import (
            send_more_money_problem, solve_with_roots_first,
        )

        result = solve_with_roots_first(send_more_money_problem(), defer_native_search=True)
        assert result.correct, f"Invalid solution: {result.error}"
        assert result.stats.get('skipped_constraints') == 0


# =============================================================================
# Approach 6: Direct SAT (also tests SEND+MORE=MONEY)
# =============================================================================

class TestDirectSAT:
    """Test direct SAT solving for cryptarithmetic."""
    
    @pytest.fixture(autouse=True)
    def check_z3(self):
        """Skip if z3 not installed."""
        pytest.importorskip("z3")
    
    def test_send_more_money_sat(self):
        """SEND+MORE=MONEY with SAT (uses column enumeration, can be slow)."""
        from propagator.examples.performance.benchmark_solver_approaches import run_direct_sat_send_more_money
        
        result = run_direct_sat_send_more_money()
        assert result.correct, f"Invalid solution: {result.error}"


# =============================================================================
# Cross-Approach Consistency
# =============================================================================

class TestCrossApproachConsistency:
    """Test that different approaches give consistent results.
    
    NOTE: Excludes Translated SMT which is an incomplete feature.
    """
    
    @pytest.fixture(autouse=True)
    def check_z3(self):
        """Skip if z3 not installed."""
        pytest.importorskip("z3")
    
    def test_all_approaches_solve_nqueens(self):
        """All working approaches should produce valid N-Queens solutions."""
        from propagator.examples.performance.benchmark_solver_approaches import (
            run_propagator_nqueens,
            run_direct_smt_nqueens,
            run_hybrid_nqueens,
        )
        
        results = {
            'DDB': run_propagator_nqueens(4, use_cdcl=False),
            'CDCL': run_propagator_nqueens(4, use_cdcl=True),
            'Direct SMT': run_direct_smt_nqueens(4),
            'Hybrid': run_hybrid_nqueens(4),
        }
        
        for name, result in results.items():
            assert result.correct, f"{name} failed: {result.error}"
    
    def test_all_approaches_solve_dwelling(self):
        """All working approaches should produce valid Dwelling solutions."""
        from propagator.examples.performance.benchmark_solver_approaches import (
            run_propagator_dwelling,
            run_direct_smt_dwelling,
            run_hybrid_dwelling,
        )
        
        results = {
            'DDB': run_propagator_dwelling(use_cdcl=False),
            'CDCL': run_propagator_dwelling(use_cdcl=True),
            'Direct SMT': run_direct_smt_dwelling(),
            'Hybrid': run_hybrid_dwelling(),
        }
        
        for name, result in results.items():
            assert result.correct, f"{name} failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
