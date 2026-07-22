"""
Propagator Network to SAT/SMT Solver Export

This module provides the unified solve interface that combines
propagator networks with SMT solver acceleration.

Recommended usage::

    from propagator import Cell
    from propagator.guessing_machine import one_of, require_distinct
    from propagator.solver_export import solve, SolveMode, search_mode, SearchMode

    with search_mode(SearchMode.DEFER_TO_SMT):
        x, y, z = Cell(name="x"), Cell(name="y"), Cell(name="z")
        one_of({1, 2, 3}, x)
        one_of({1, 2, 3}, y)
        one_of({1, 2, 3}, z)
        require_distinct([x, y, z])

    result = solve([x, y, z], mode=SolveMode.SMT_ITERATIVE, verbose=True)
    print(result.solution)  # {x: 1, y: 2, z: 3}
"""

from .compiler import NetworkCompiler, SolverBackend, ConstraintType, SolverResult
from .backends import SolverBackendBase, DimacsBackend, SMTLib2Backend
from .solver_runner import SolverRunner, solve_dimacs, solve_smtlib2
from .auto_extract import extract_domains_only
from .from_roots import (
    TranslationMode, TranslationIssue, RootCompileReport,
    UnsupportedTranslationError, compile_from_roots, solve_from_roots,
)
from .solve import solve, SolveMode, SolveResult
from .search_mode import SearchMode, search_mode
from .true_hybrid import TrueHybridNetwork, SMTHypothesis, TrackedConstraint

__all__ = [
    # === Recommended API ===
    'solve', 'SolveMode', 'SolveResult',
    'SearchMode', 'search_mode',

    # === Compiler infrastructure ===
    'NetworkCompiler', 'SolverBackend', 'ConstraintType', 'SolverResult',

    # === Internal engine (advanced use) ===
    'compile_from_roots', 'solve_from_roots',
    'TranslationMode', 'TranslationIssue', 'RootCompileReport',
    'UnsupportedTranslationError',

    # === Building blocks ===
    'TrueHybridNetwork', 'SMTHypothesis', 'TrackedConstraint',
    'extract_domains_only',

    # === Backends and solver execution ===
    'SolverBackendBase', 'DimacsBackend', 'SMTLib2Backend',
    'SolverRunner', 'solve_dimacs', 'solve_smtlib2',
]
