"""Side-by-side comparison demo for WHERE repair approaches.

This program compares three WHERE-repair strategies over a fixed catalog:
1) network_data  - reuse repair-network relation outputs
2) network_tms   - TMS worldview search over predicate positions
3) qr_hint_ast   - AST site/bounds/fix search (QR-Hint-style)

It prints:
- The exact interfaces each strategy consumes and returns
- Per-scenario diagnosis output
- Strategy viability, chosen repair sites, and costs
- Per-strategy overhead (median and mean runtime in ms)

Run with:
    python -m propagator.sql.demo_compare_repair_approaches
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import sqlglot
from sqlglot import exp

from propagator.sql.catalog import Catalog
from propagator.sql.relation_info import FullRelation
from propagator.sql.repair import (
    HintLevel,
    diagnose_from_reference,
    _compare_where_repair_approaches,
    _extract_predicates,
    _find_minimal_predicate_repairs_from_network,
    _find_minimal_predicate_repairs_qr_hint,
    _find_minimal_predicate_repairs_tms,
    _gather_from_rows,
    _run_query_network,
    build_repair_network,
)


@dataclass
class Scenario:
    """One working/reference query pair to evaluate."""

    name: str
    working_sql: str
    reference_sql: str
    notes: str


@dataclass
class WhereContext:
    """Inputs required to run all WHERE repair strategies."""

    working_preds: List[Any]
    reference_preds: List[Any]
    working_pred: Any
    reference_pred: Any
    all_rows: List[dict]
    working_filtered: Optional[FullRelation]
    reference_filtered: Optional[FullRelation]


def build_fixed_catalog() -> Catalog:
    """Build one fixed catalog used by every scenario in this demo."""
    return Catalog.from_dicts(
        {
            "employees": [
                {"id": 1, "name": "Alice", "dept": "eng", "salary": 100, "age": 30},
                {"id": 2, "name": "Bob", "dept": "eng", "salary": 120, "age": 27},
                {"id": 3, "name": "Carol", "dept": "sales", "salary": 90, "age": 36},
                {"id": 4, "name": "Dave", "dept": "sales", "salary": 110, "age": 34},
                {"id": 5, "name": "Eve", "dept": "hr", "salary": 95, "age": 29},
                {"id": 6, "name": "Frank", "dept": "eng", "salary": 80, "age": 41},
            ],
            "departments": [
                {"dept": "eng", "budget": 500, "region": "west"},
                {"dept": "sales", "budget": 300, "region": "east"},
                {"dept": "hr", "budget": 200, "region": "west"},
            ],
            "bonuses": [
                {"employee_id": 1, "bonus": 20},
                {"employee_id": 2, "bonus": 10},
                {"employee_id": 3, "bonus": 5},
                {"employee_id": 4, "bonus": 15},
                {"employee_id": 5, "bonus": 12},
            ],
        }
    )


def build_scenarios() -> List[Scenario]:
    """Edge-case-heavy scenario set over a fixed catalog."""
    return [
        Scenario(
            name="Equivalent WHERE",
            working_sql=(
                "SELECT name, salary FROM employees "
                "WHERE salary > 100"
            ),
            reference_sql=(
                "SELECT name, salary FROM employees "
                "WHERE salary > 100"
            ),
            notes="No repair required; approaches should be viable with empty sites.",
        ),
        Scenario(
            name="Missing WHERE",
            working_sql="SELECT name, salary FROM employees",
            reference_sql="SELECT name, salary FROM employees WHERE salary > 100",
            notes="Structural mismatch: working query missing WHERE clause.",
        ),
        Scenario(
            name="Extra WHERE",
            working_sql="SELECT name, salary FROM employees WHERE salary > 100",
            reference_sql="SELECT name, salary FROM employees",
            notes="Structural mismatch: working query has extra WHERE clause.",
        ),
        Scenario(
            name="Wrong Literal",
            working_sql="SELECT name FROM employees WHERE salary > 200",
            reference_sql="SELECT name FROM employees WHERE salary > 100",
            notes="Single predicate threshold error.",
        ),
        Scenario(
            name="Wrong Operator",
            working_sql="SELECT name FROM employees WHERE salary < 100",
            reference_sql="SELECT name FROM employees WHERE salary > 100",
            notes="Single predicate operator direction error.",
        ),
        Scenario(
            name="Two Predicates One Wrong",
            working_sql=(
                "SELECT name FROM employees "
                "WHERE salary > 100 AND dept = 'eng'"
            ),
            reference_sql=(
                "SELECT name FROM employees "
                "WHERE salary > 100 AND dept = 'sales'"
            ),
            notes="AND predicate vector with one mismatching site.",
        ),
        Scenario(
            name="Working Has Extra Predicate",
            working_sql=(
                "SELECT name FROM employees "
                "WHERE salary > 100 AND age > 35"
            ),
            reference_sql=(
                "SELECT name FROM employees "
                "WHERE salary > 100"
            ),
            notes="Working query is too restrictive due to extra conjunct.",
        ),
        Scenario(
            name="Working Missing Predicate",
            working_sql=(
                "SELECT name FROM employees "
                "WHERE salary > 100"
            ),
            reference_sql=(
                "SELECT name FROM employees "
                "WHERE salary > 100 AND dept = 'eng'"
            ),
            notes="Working query too permissive due to missing conjunct.",
        ),
        Scenario(
            name="Wrong Column",
            working_sql="SELECT name FROM employees WHERE age > 30",
            reference_sql="SELECT name FROM employees WHERE salary > 100",
            notes="Predicate refers to wrong column family.",
        ),
        Scenario(
            name="Join Predicate Mismatch",
            working_sql=(
                "SELECT e.name FROM employees e "
                "JOIN departments d ON e.dept = d.dept "
                "WHERE d.region = 'east'"
            ),
            reference_sql=(
                "SELECT e.name FROM employees e "
                "JOIN departments d ON e.dept = d.dept "
                "WHERE d.region = 'west'"
            ),
            notes="WHERE mismatch over joined relation aliases.",
        ),
    ]


def _format_sites(sites: List[Any]) -> str:
    if not sites:
        return "[]"
    return "[" + ", ".join(str(s) for s in sites) + "]"


def _benchmark(fn: Callable[[], Any], repeat: int = 25) -> Dict[str, float]:
    timings = []
    for _ in range(repeat):
        t0 = perf_counter()
        fn()
        t1 = perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return {
        "median_ms": median(timings),
        "mean_ms": mean(timings),
    }


def _prepare_where_context(scenario: Scenario, catalog: Catalog) -> Optional[WhereContext]:
    w_ast = sqlglot.parse_one(scenario.working_sql)
    r_ast = sqlglot.parse_one(scenario.reference_sql)

    w_where = w_ast.find(exp.Where)
    r_where = r_ast.find(exp.Where)
    if w_where is None or r_where is None:
        return None

    w_net, _ = _run_query_network(scenario.working_sql, catalog)
    r_net, _ = _run_query_network(scenario.reference_sql, catalog)
    repair_net = build_repair_network(w_net, r_net, w_ast, r_ast)

    where_stage = repair_net.stages.get("WHERE")
    if where_stage is None:
        return None

    all_rows = _gather_from_rows(w_ast, catalog)
    return WhereContext(
        working_preds=_extract_predicates(w_where.this),
        reference_preds=_extract_predicates(r_where.this),
        working_pred=w_where.this,
        reference_pred=r_where.this,
        all_rows=all_rows,
        working_filtered=(
            where_stage.working_cell.content
            if isinstance(where_stage.working_cell.content, FullRelation)
            else None
        ),
        reference_filtered=(
            where_stage.reference_cell.content
            if isinstance(where_stage.reference_cell.content, FullRelation)
            else None
        ),
    )


def _print_interface_contracts() -> None:
    print("=" * 90)
    print("  WHERE Repair Approach Comparison Demo")
    print("=" * 90)
    print("\nInterfaces this demo uses:")
    print("  1) High-level diagnosis interface")
    print("     diagnose_from_reference(working_sql, reference_sql, catalog, hint_level)")
    print("     Provides: stage_results, hints, selected approach, and per-approach summary.")
    print("\n  2) Strategy interface: network_data")
    print("     _find_minimal_predicate_repairs_from_network(")
    print("       working_preds, reference_preds, all_rows, working_filtered, reference_filtered")
    print("     Provides: viable flag, repair_sites, cost.")
    print("\n  3) Strategy interface: network_tms")
    print("     _find_minimal_predicate_repairs_tms(working_preds, reference_preds, all_rows)")
    print("     Provides: viable flag, repair_sites, cost.")
    print("\n  4) Strategy interface: qr_hint_ast")
    print("     _find_minimal_predicate_repairs_qr_hint(working_pred, reference_pred, all_rows)")
    print("     Provides: viable flag, repair_sites, cost, derived fixes.")
    print("\nWhat code is benchmarked for overhead:")
    print("  - ONLY the approach functions above (post-parse, post-row-gather).")
    print("  - Full diagnosis time is also reported separately for context.")


def run_demo() -> None:
    _print_interface_contracts()

    catalog = build_fixed_catalog()
    scenarios = build_scenarios()

    print("\nFixed catalog tables:")
    for table in catalog.table_names():
        full = catalog.get_table(table)
        rows = len(full.rows) if isinstance(full, FullRelation) else 0
        print(f"  - {table}: {rows} rows")

    print("\nScenario execution begins...\n")

    for i, scenario in enumerate(scenarios, start=1):
        print("-" * 90)
        print(f"Scenario {i}: {scenario.name}")
        print(f"Notes: {scenario.notes}")
        print(f"Working : {scenario.working_sql}")
        print(f"Reference: {scenario.reference_sql}")

        t0 = perf_counter()
        report = diagnose_from_reference(
            scenario.working_sql,
            scenario.reference_sql,
            catalog,
            hint_level=HintLevel.DIRECTION,
        )
        t1 = perf_counter()
        diagnosis_ms = (t1 - t0) * 1000.0

        print(f"\nDiagnosis overhead (end-to-end): {diagnosis_ms:.3f} ms")
        print(f"Equivalent: {report.equivalent}")
        print(f"Stage results: {report.stage_results}")

        where_cmp = report.approach_comparison.get("WHERE", {})
        if where_cmp:
            print("Selected approach from oracle:", where_cmp.get("selected"))
            for key in ("network_data", "network_tms", "qr_hint_ast"):
                item = where_cmp.get(key, {})
                print(
                    f"  {key:12s} viable={item.get('viable')} "
                    f"cost={item.get('cost')} sites={item.get('repair_sites')}"
                )
        else:
            print("Selected approach from oracle: N/A (WHERE structural mismatch or stage not reached)")

        ctx = _prepare_where_context(scenario, catalog)
        if ctx is None:
            print("\nPer-approach benchmarking: skipped (cannot build WHERE context).")
            if report.hints:
                print("Top hint:", report.hints[0].message)
            continue

        # Run each approach once for detailed result snapshots.
        r_network_data = _find_minimal_predicate_repairs_from_network(
            ctx.working_preds,
            ctx.reference_preds,
            ctx.all_rows,
            ctx.working_filtered,
            ctx.reference_filtered,
        )
        r_network_tms = _find_minimal_predicate_repairs_tms(
            ctx.working_preds,
            ctx.reference_preds,
            ctx.all_rows,
        )
        r_qr_hint_ast = _find_minimal_predicate_repairs_qr_hint(
            ctx.working_pred,
            ctx.reference_pred,
            ctx.all_rows,
        )

        print("\nApproach result snapshots:")
        print(
            "  network_data : "
            f"viable={r_network_data.viable} cost={r_network_data.cost:.6f} "
            f"sites={_format_sites(r_network_data.repair_sites)}"
        )
        print(
            "  network_tms  : "
            f"viable={r_network_tms.viable} cost={r_network_tms.cost:.6f} "
            f"sites={_format_sites(r_network_tms.repair_sites)}"
        )
        print(
            "  qr_hint_ast  : "
            f"viable={r_qr_hint_ast.viable} cost={r_qr_hint_ast.cost:.6f} "
            f"sites={_format_sites(r_qr_hint_ast.repair_sites)}"
        )

        # Benchmark only approach kernels to compare overhead.
        b_network_data = _benchmark(
            lambda: _find_minimal_predicate_repairs_from_network(
                ctx.working_preds,
                ctx.reference_preds,
                ctx.all_rows,
                ctx.working_filtered,
                ctx.reference_filtered,
            )
        )
        b_network_tms = _benchmark(
            lambda: _find_minimal_predicate_repairs_tms(
                ctx.working_preds,
                ctx.reference_preds,
                ctx.all_rows,
            )
        )
        b_qr_hint_ast = _benchmark(
            lambda: _find_minimal_predicate_repairs_qr_hint(
                ctx.working_pred,
                ctx.reference_pred,
                ctx.all_rows,
            )
        )

        print("\nApproach overhead (kernel-only, 25 runs):")
        print(
            "  network_data : "
            f"median={b_network_data['median_ms']:.4f} ms "
            f"mean={b_network_data['mean_ms']:.4f} ms"
        )
        print(
            "  network_tms  : "
            f"median={b_network_tms['median_ms']:.4f} ms "
            f"mean={b_network_tms['mean_ms']:.4f} ms"
        )
        print(
            "  qr_hint_ast  : "
            f"median={b_qr_hint_ast['median_ms']:.4f} ms "
            f"mean={b_qr_hint_ast['mean_ms']:.4f} ms"
        )

        if report.hints:
            print("\nTop hint:", report.hints[0].message)

    print("\n" + "=" * 90)
    print("Demo complete.")
    print("=" * 90)


if __name__ == "__main__":
    run_demo()
