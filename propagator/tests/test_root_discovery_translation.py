"""Tests for root-cell based network discovery and unified compile-from-roots APIs."""

import pytest

from propagator import Cell, absolute_value, adder, conditional, constant, initialize_scheduler, one_of
from propagator.network_discovery import discover_network
from propagator.solver_export import (
    SolverBackend,
    TranslationMode,
    UnsupportedTranslationError,
    compile_from_roots,
    extract_domains_only,
    solve_from_roots,
)


# Check if z3 is available
try:
    import z3  # noqa: F401

    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


def test_discover_network_from_roots_finds_reachable_cells_and_add_constraint():
    initialize_scheduler()

    a = Cell(name="a")
    b = Cell(name="b")
    c = Cell(name="c")

    adder(a, b, c)

    discovered = discover_network([a])

    assert a in discovered.cells
    assert b in discovered.cells
    assert c in discovered.cells

    add_constraints = [ct for ct in discovered.constraints if ct.kind == "add"]
    assert add_constraints, "Expected an inferred add constraint"


def test_extract_domains_only_scopes_to_roots():
    initialize_scheduler()

    x = Cell(name="x")
    y = Cell(name="y")

    one_of([1, 2], x)
    one_of([3, 4], y)

    compiler = extract_domains_only(root_cells=[x])

    assert x in compiler.variables
    assert y not in compiler.variables


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
def test_solve_from_roots_strict_solves_existing_network():
    initialize_scheduler()

    a = Cell(name="a")
    b = Cell(name="b")
    c = Cell(name="c")

    one_of([1, 2, 3, 4], a)
    one_of([1, 2, 3, 4], b)
    adder(a, b, c)
    constant(5, c)

    result, report = solve_from_roots(
        [a, b],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.STRICT,
    )

    assert result.satisfiable
    assert result.solution is not None
    assert result.solution[a] + result.solution[b] == 5
    assert report.skipped_constraint_count == 0


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
def test_compile_from_roots_strict_blocks_on_unsupported_operations():
    initialize_scheduler()

    a = Cell(name="a")
    b = Cell(name="b")
    c = Cell(name="c")
    out = Cell(name="out")

    # conditional/switch with unpinned control is not translatable
    # via _try_add_constraint — a genuinely unsupported discovered
    # constraint (unlike and/or/not which now have reification).
    one_of([True, False], a)
    one_of([True, False], b)
    conditional(a, b, c, out)

    with pytest.raises(UnsupportedTranslationError):
        compile_from_roots(
            [a, b, c, out],
            backend=SolverBackend.Z3_PYTHON,
            mode=TranslationMode.STRICT,
        )


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
def test_solve_from_roots_hybrid_oracle_skips_unsupported_but_solves_subset():
    initialize_scheduler()

    a = Cell(name="a")
    b = Cell(name="b")
    c = Cell(name="c")
    out = Cell(name="out")

    one_of([True, False], a)
    one_of([True, False], b)
    conditional(a, b, c, out)  # switch: untranslatable in hybrid mode

    result, report = solve_from_roots(
        [a, b, c, out],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
    )

    assert result.satisfiable
    assert report.skipped_constraint_count >= 1
    assert any(issue.kind == "switch" for issue in report.issues)


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
def test_compile_from_roots_strict_translates_absolute_value():
    initialize_scheduler()

    x = Cell(name="x")
    y = Cell(name="y")

    one_of([-3, -2, -1, 0, 1, 2, 3], x)
    one_of([0, 1, 2, 3], y)
    absolute_value(x, y)
    constant(2, y)

    report = compile_from_roots(
        [x, y],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.STRICT,
    )

    assert report.skipped_constraint_count == 0
    result = report.compiler.solve(backend=SolverBackend.Z3_PYTHON)
    assert result.satisfiable
    assert result.solution is not None
    assert abs(result.solution[x]) == result.solution[y] == 2


@pytest.mark.skipif(not HAS_Z3, reason="Z3 not available")
def test_solve_from_roots_default_include_nogoods_is_unsound_on_searched_network():
    """
    Regression test for a real false-UNSAT: a network built the ordinary way
    (one_of/require_distinct under default auto-run) runs real amb/DDB search
    *during construction*, and can learn nogoods that are only valid against
    that incomplete, in-progress network. solve_from_roots's own default of
    include_nogoods=True exports those as hard SMT constraints and turns this
    known-satisfiable puzzle into a false UNSAT; include_nogoods=False (the
    documented workaround, see solve_hybrid_from_existing_network's docstring
    in solver_export/true_hybrid.py) solves it correctly.

    This is exactly the "take an existing propagator network and solve from
    its roots" use case -- callers who don't know to override the default
    will get a wrong answer on ordinarily-built networks like this one.
    """
    from propagator.examples.puzzles.superintendent_puzzle import multiple_dwelling

    initialize_scheduler()
    cells = multiple_dwelling()

    result_default, _ = solve_from_roots(
        cells,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
    )
    assert not result_default.satisfiable, (
        "If this now passes, solve_from_roots's default include_nogoods=True "
        "handling has changed -- update/remove this regression test and the "
        "include_nogoods=False guidance that depends on it."
    )

    result_fixed, report_fixed = solve_from_roots(
        cells,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        include_nogoods=False,
    )
    assert result_fixed.satisfiable
    assert result_fixed.solution is not None
    baker, fletcher, smith, cooper, miller = cells
    solution = {c.name: result_fixed.solution[c] for c in cells if c in result_fixed.solution}
    assert len(set(solution.values())) == 5
    assert solution["miller"] > solution["cooper"]
    assert solution["baker"] != 5 and solution["cooper"] != 1
    assert solution["fletcher"] not in (1, 5)
    assert abs(solution["smith"] - solution["fletcher"]) != 1
    assert abs(solution["fletcher"] - solution["cooper"]) != 1
