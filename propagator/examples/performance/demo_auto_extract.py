#!/usr/bin/env python3
"""Demo: automatic roots-first solver export workflows.

This demo shows the current recommended flow:
1) Build a normal propagator network.
2) Compile/solve from root cells via solve_from_roots().
3) Inspect discovered domains via compile report metadata.
"""

from propagator import Cell, adder, constant, initialize_scheduler, one_of
from propagator.solver_export import (
    SolverBackend,
    TranslationMode,
    compile_from_roots,
    solve_from_roots,
)


def demo_sum_to_seven() -> None:
    print("=" * 60)
    print("DEMO 1: Roots-First Automatic Solve")
    print("=" * 60)

    initialize_scheduler()

    a = Cell(name="a")
    b = Cell(name="b")
    total = Cell(name="total")

    one_of([1, 2, 3, 4, 5], a)
    one_of([1, 2, 3, 4, 5], b)
    adder(a, b, total)
    constant(7, total)

    result, report = solve_from_roots(
        [a, b],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.STRICT,
        name="sum_to_seven",
    )

    print(f"translated={report.translated_constraint_count} skipped={report.skipped_constraint_count}")
    if result.satisfiable and result.solution:
        print(f"solution: a={result.solution[a]} b={result.solution[b]} total={result.solution[total]}")
    else:
        print("no solution")


def demo_domain_view() -> None:
    print("\n" + "=" * 60)
    print("DEMO 2: Domain Introspection from Compile Report")
    print("=" * 60)

    initialize_scheduler()

    x = Cell(name="x")
    y = Cell(name="y")
    one_of([1, 2, 3], x)
    one_of([10, 20, 30], y)

    report = compile_from_roots(
        [x],
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        name="domains_demo_compile",
    )
    compiler = report.compiler

    print(f"registered variables: {len(compiler.variables)}")
    for cell, var in compiler.variables.items():
        print(f"  {var.name}: {var.domain}")
    print(f"compile report: translated={report.translated_constraint_count} skipped={report.skipped_constraint_count}")


if __name__ == "__main__":
    demo_sum_to_seven()
    demo_domain_view()
