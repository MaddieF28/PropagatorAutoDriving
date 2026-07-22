#!/usr/bin/env python3
"""
Demonstration: propagator network compilation to SAT/SMT solvers.

Shows the compiler pipeline at multiple levels:
  1. High-level: ``solve()`` with ``search_mode()`` (recommended)
  2. Mid-level: ``compile_from_roots()`` for inspecting the encoding
  3. Low-level: ``NetworkCompiler`` directly

Run with:
    python3 -m propagator.examples.performance.demo_solver_export
"""

from __future__ import annotations

import os
import tempfile
import time

from propagator import (
    Cell,
    abhor,
    absolute_value,
    adder,
    constant,
    eq,
    gt,
    gte,
    initialize_scheduler,
    lte,
    one_of,
    require,
    require_distinct,
    subtractor,
)
from propagator.solver_export import (
    SolverBackend,
    TranslationMode,           # internal — for compile_from_roots demos
    compile_from_roots,
    solve_from_roots,          # internal engine — prefer solve() below
    solve,
    SolveMode,
    search_mode,
    SearchMode,
)


def _print_preview(label: str, content: str, max_chars: int = 2000) -> None:
    """Print a bounded preview to keep the demo responsive in terminals."""
    print(f"\n{label} preview:")
    preview = content[:max_chars]
    print(preview)
    if len(content) > max_chars:
        print(f"... ({len(content) - max_chars} more characters)")


def _run_demo_step(title: str, fn) -> None:
    """Run one demo step with explicit timing/progress output."""
    print(f"\n>>> Starting: {title}")
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f">>> Finished: {title} in {elapsed:.3f}s")


def demo_basic_compilation():
    """Compile a simple roots-first network and export SAT/SMT encodings."""
    print("=" * 70)
    print("DEMO 1: Roots-First Compilation to SAT/SMT Formats")
    print("=" * 70)

    initialize_scheduler()

    x = Cell(name="x")
    y = Cell(name="y")
    z = Cell(name="z")

    one_of([1, 2, 3], x)
    one_of([1, 2, 3], y)
    one_of([2, 3, 4, 5, 6], z)
    adder(x, y, z)
    constant(4, z)

    report = compile_from_roots(
        [x, y, z],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.STRICT,
        name="simple_sum",
    )
    compiler = report.compiler

    print("\nProblem: x, y in {1, 2, 3}, z in {2..6}, and x + y = z = 4")
    print(f"Translated={report.translated_constraint_count}, skipped={report.skipped_constraint_count}")

    print("\n" + "-" * 40)
    print("DIMACS CNF Format:")
    print("-" * 40)
    dimacs_result = compiler.export(SolverBackend.DIMACS_CNF)
    _print_preview("DIMACS CNF", dimacs_result.content, max_chars=800)
    print(f"\nStats: {dimacs_result.var_count} variables, {dimacs_result.clause_count} clauses")

    print("\n" + "-" * 40)
    print("SMT-LIB2 Format:")
    print("-" * 40)
    smt_result = compiler.export(SolverBackend.SMT_LIB2)
    _print_preview("SMT-LIB2", smt_result.content, max_chars=800)

    return compiler


def demo_multiple_dwelling():
    """Solve/export Multiple Dwelling from a roots-first propagator model."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multiple Dwelling (Roots-First)")
    print("=" * 70)

    initialize_scheduler()

    baker = Cell(name="baker")
    cooper = Cell(name="cooper")
    fletcher = Cell(name="fletcher")
    miller = Cell(name="miller")
    smith = Cell(name="smith")

    people = [baker, cooper, fletcher, miller, smith]

    # Keep this demo snappy: model 1..5 bounds directly instead of one_of(),
    # which can trigger expensive AMB/TMS branching before export.
    one = Cell(name="one")
    five = Cell(name="five")
    constant(1, one)
    constant(5, five)
    for p in people:
        p_gte_1 = Cell(name=f"{p.name}_gte_1")
        p_lte_5 = Cell(name=f"{p.name}_lte_5")
        gte(p, one, p_gte_1)
        lte(p, five, p_lte_5)
        require(p_gte_1)
        require(p_lte_5)

    # Distinct floors: pairwise inequality via eq(...)->abhor(...)
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            same_floor = Cell(name=f"{people[i].name}_{people[j].name}_same")
            eq(people[i], people[j], same_floor)
            abhor(same_floor)

    # baker != 5, cooper != 1, fletcher != 1,5
    for cell, bad in [(baker, 5), (cooper, 1), (fletcher, 1), (fletcher, 5)]:
        bad_cell = Cell()
        bad_eq = Cell()
        constant(bad, bad_cell)
        eq(cell, bad_cell, bad_eq)
        abhor(bad_eq)

    # miller > cooper
    m_gt_c = Cell(name="m_gt_c")
    gt(miller, cooper, m_gt_c)
    require(m_gt_c)

    # abs(smith - fletcher) != 1
    s_f_diff = Cell(name="s_f_diff")
    s_f_abs = Cell(name="s_f_abs")
    one = Cell()
    s_f_adj = Cell(name="s_f_adj")
    subtractor(smith, fletcher, s_f_diff)
    absolute_value(s_f_diff, s_f_abs)
    constant(1, one)
    eq(s_f_abs, one, s_f_adj)
    abhor(s_f_adj)

    # abs(fletcher - cooper) != 1
    f_c_diff = Cell(name="f_c_diff")
    f_c_abs = Cell(name="f_c_abs")
    f_c_adj = Cell(name="f_c_adj")
    subtractor(fletcher, cooper, f_c_diff)
    absolute_value(f_c_diff, f_c_abs)
    eq(f_c_abs, one, f_c_adj)
    abhor(f_c_adj)

    result, report = solve_from_roots(
        people,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        name="multiple_dwelling",
        include_nogoods=False,
        timeout=5.0,
    )

    print("\nProblem constraints modeled with propagators; compiled from roots.")
    print(f"Translated={report.translated_constraint_count}, skipped={report.skipped_constraint_count}")
    if result.satisfiable and result.solution:
        print("Solution:")
        for person in people:
            print(f"  {person.name}: floor {result.solution.get(person)}")
    else:
        print(f"No solution: {result.error}")

    compiler = report.compiler
    smt_result = compiler.export(SolverBackend.SMT_LIB2)
    _print_preview("SMT-LIB2 encoding", smt_result.content, max_chars=800)

    return compiler


def demo_solving_with_z3():
    """Solve 4-Queens via roots-first compile/solve."""
    print("\n" + "=" * 70)
    print("DEMO 3: Solving 4-Queens with roots-first Z3 backend")
    print("=" * 70)

    try:
        import z3  # noqa: F401
    except ImportError:
        print("Z3 not installed. Install with: pip install z3-solver")
        print("Skipping Z3 demo.")
        return None

    initialize_scheduler()

    n = 4
    queens = [Cell(name=f"q{i}") for i in range(n)]
    for q in queens:
        one_of(list(range(n)), q)
    require_distinct(queens)

    # No same diagonal: q_i - q_j != ±(j-i)
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            diff = Cell(name=f"diff_{i}_{j}")
            subtractor(queens[i], queens[j], diff)
            for forbidden in (d, -d):
                d_cell = Cell()
                eq_cell = Cell()
                constant(forbidden, d_cell)
                eq(diff, d_cell, eq_cell)
                abhor(eq_cell)

    result, report = solve_from_roots(
        queens,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        name="4_queens",
        include_nogoods=False,
        timeout=5.0,
    )

    print(f"Translated={report.translated_constraint_count}, skipped={report.skipped_constraint_count}")
    if not (result.satisfiable and result.solution):
        print(f"No solution found: {result.error}")
        return result

    print("\nSolution found!")
    for i in range(n):
        col = result.solution.get(queens[i], -1)
        row = ["." if j != col else "Q" for j in range(n)]
        print("  " + " ".join(row))

    return result


def demo_convenience_functions():
    """Demonstrate compile_from_roots + solve_from_roots convenience flow."""
    print("\n" + "=" * 70)
    print("DEMO 4: Roots-First Convenience Functions")
    print("=" * 70)

    initialize_scheduler()

    a = Cell(name="a")
    b = Cell(name="b")
    c = Cell(name="c")

    one_of([1, 2, 3, 4, 5], a)
    one_of([1, 2, 3, 4, 5], b)
    adder(a, b, c)
    constant(7, c)

    compile_report = compile_from_roots(
        [a, b],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.STRICT,
        name="sum_to_seven",
    )
    compiler = compile_report.compiler

    print("\nProblem: a, b in {1..5}, a + b = 7")
    print(f"compile translated={compile_report.translated_constraint_count}, skipped={compile_report.skipped_constraint_count}")

    dimacs = compiler.export(SolverBackend.DIMACS_CNF)
    smt = compiler.export(SolverBackend.SMT_LIB2)
    print(f"DIMACS: {dimacs.var_count} vars, {dimacs.clause_count} clauses")
    print(f"SMT-LIB2: {smt.clause_count} assertions")

    result, solve_report = solve_from_roots(
        [a, b],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.STRICT,
        name="sum_to_seven_solve",
        include_nogoods=False,
    )
    if result.satisfiable and result.solution:
        print(f"Solution: a={result.solution[a]}, b={result.solution[b]}, c={result.solution[c]}")
    print(f"solve translated={solve_report.translated_constraint_count}, skipped={solve_report.skipped_constraint_count}")

    return compiler


def demo_write_files():
    """Write DIMACS/SMT files from a roots-first compiled network."""
    print("\n" + "=" * 70)
    print("DEMO 5: Writing Files for External Solvers")
    print("=" * 70)

    initialize_scheduler()

    x = Cell(name="x")
    y = Cell(name="y")

    one_of([1, 2, 3, 4, 5], x)
    one_of([1, 2, 3, 4, 5], y)
    x_gt_y = Cell(name="x_gt_y")
    gt(x, y, x_gt_y)
    require(x_gt_y)

    report = compile_from_roots(
        [x, y],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        name="write_files_example",
    )
    compiler = report.compiler

    with tempfile.TemporaryDirectory() as tmpdir:
        cnf_path = os.path.join(tmpdir, "example.cnf")
        smt_path = os.path.join(tmpdir, "example.smt2")

        compiler.write_to_file(cnf_path, SolverBackend.DIMACS_CNF)
        compiler.write_to_file(smt_path, SolverBackend.SMT_LIB2)

        print(f"\nWrote DIMACS CNF to: {cnf_path}")
        print(f"Wrote SMT-LIB2 to: {smt_path}")

        print("\nDIMACS content:")
        with open(cnf_path, encoding="utf-8") as f:
            _print_preview("DIMACS file", f.read(), max_chars=600)

        print("\nSMT-LIB2 content:")
        with open(smt_path, encoding="utf-8") as f:
            _print_preview("SMT-LIB2 file", f.read(), max_chars=600)

    return compiler


def demo_comparison_with_propagator(n: int = 6):
    """Quick timing demo using roots-first solve for an N-Queens problem."""
    print("\n" + "=" * 70)
    print(f"DEMO 6: Roots-First Timing Snapshot ({n}-Queens)")
    print("=" * 70)

    initialize_scheduler()

    # Defer all search to SMT during construction.
    with search_mode(SearchMode.DEFER_TO_SMT):
        queens = [Cell(name=f"q{i}") for i in range(n)]
        for q in queens:
            one_of(list(range(n)), q)
        require_distinct(queens)

        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                diff = Cell(name=f"diff_{i}_{j}")
                subtractor(queens[i], queens[j], diff)
                for forbidden in (d, -d):
                    d_cell = Cell()
                    eq_cell = Cell()
                    constant(forbidden, d_cell)
                    eq(diff, d_cell, eq_cell)
                    abhor(eq_cell)

    start = time.time()
    result, report = solve_from_roots(
        queens,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        name=f"{n}_queens",
        include_nogoods=False,
        timeout=5.0,
    )
    elapsed = time.time() - start

    print(f"Time: {elapsed:.4f}s")
    print(f"Translated={report.translated_constraint_count}, skipped={report.skipped_constraint_count}")
    print(f"SAT: {result.satisfiable}")
    if result.satisfiable and result.solution:
        solution = [result.solution[q] for q in queens]
        print(f"Solution: {solution}")


def main():
    """Run all roots-first solver-export demos."""
    print("\n" + "=" * 70)
    print("ROOTS-FIRST PROPAGATOR NETWORK -> SAT/SMT DEMO")
    print("=" * 70)
    print(
        """
This demonstrates compiling propagator networks from root cells to:
  1. DIMACS CNF format (for SAT solvers)
  2. SMT-LIB2 format (for SMT solvers)

All demos follow the same procedure:
  - model constraints with normal propagators
  - compile/solve via compile_from_roots/solve_from_roots
"""
    )

    _run_demo_step("Demo 1 - Basic Compilation", demo_basic_compilation)
    _run_demo_step("Demo 2 - Multiple Dwelling", demo_multiple_dwelling)
    _run_demo_step("Demo 3 - 4-Queens", demo_solving_with_z3)
    _run_demo_step("Demo 4 - Convenience Functions", demo_convenience_functions)
    _run_demo_step("Demo 5 - Write Files", demo_write_files)
    # Keep default run snappy; raise n manually for heavier stress tests.
    _run_demo_step("Demo 6 - Timing Snapshot", lambda: demo_comparison_with_propagator(n=6))

    print("\n" + "=" * 70)
    print("DEMOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
