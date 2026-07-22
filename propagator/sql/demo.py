"""End-to-end demo of SQL query modeling with propagators.

Demonstrates:
1. Parsing real SQL into propagator networks via sqlglot
2. Loading schema and data from .sql files
3. Stepping through lattice levels (schema -> estimate -> full)
4. Cardinality diagnostics (estimate vs actual discrepancies)
5. Row-level provenance tracking
6. Query repair diagnosis
"""

import os

from propagator import Cell, initialize_scheduler, run
from propagator.sql import (
    Catalog, ColumnDef, SchemaInfo, EstimateInfo, FullRelation,
    QueryNetwork, QueryStepper,
    parse_query, parse_sql_file, parse_sql_statements,
    scan, filter_prop, join_prop, aggregate_prop, project_prop,
    filter_constraint, join_constraint,
    cardinality_surprise, diagnose_network,
    explain_row, why_missing,
    diagnose_from_reference, print_repair_report,
    HintLevel,
    register_sql_merge,
)
from propagator.sql.visualize import (
    print_network, print_diagnostics, print_provenance,
)


def load_sample_catalog():
    """Load catalog from the sample_data.sql file."""
    sql_file = os.path.join(os.path.dirname(__file__), "sample_data.sql")
    catalog, _ = Catalog.from_sql_file(sql_file)
    return catalog


def demo_basic_query():
    """Demo 1: Parse real SQL into a propagator network.

    Query: SELECT name, SUM(amount) as total
           FROM orders JOIN customers ON cust_id = id
           WHERE amount > 100
           GROUP BY name
    """
    print("\n" + "="*70)
    print(" DEMO 1: Real SQL -> propagator network")
    print("="*70)

    initialize_scheduler()
    catalog = load_sample_catalog()

    sql = """
        SELECT name, SUM(amount) AS total
        FROM orders
        JOIN customers ON cust_id = id
        WHERE amount > 100
        GROUP BY name
    """

    print(f"\n  SQL: {sql.strip()}")
    net = parse_query(sql, catalog)

    # Run propagation
    run()

    # Visualize
    print_network(net, show_rows=True)

    # Diagnostics
    issues = diagnose_network(net)
    print_diagnostics(issues)

    return net


def demo_sql_file_loading():
    """Demo 2: Load schema + data from a .sql file, then query."""
    print("\n" + "="*70)
    print(" DEMO 2: Load .sql file -> build catalog -> query")
    print("="*70)

    initialize_scheduler()

    # Load from file
    sql_file = os.path.join(os.path.dirname(__file__), "sample_data.sql")
    catalog, _ = Catalog.from_sql_file(sql_file)

    print(f"\n  Loaded tables: {catalog.table_names()}")
    for t in catalog.table_names():
        info = catalog.get_table(t)
        print(f"    {t}: {info}")

    # Run a query on the loaded data
    sql = "SELECT city, COUNT(*) AS num_customers FROM customers GROUP BY city"
    print(f"\n  Query: {sql}")

    net = parse_query(sql, catalog)
    run()
    print_network(net, show_rows=True)

    return net


def demo_inline_sql():
    """Demo 3: Build catalog from inline SQL statements (no file needed)."""
    print("\n" + "="*70)
    print(" DEMO 3: Inline SQL DDL+DML -> catalog -> query")
    print("="*70)

    initialize_scheduler()

    ddl_dml = """
        CREATE TABLE employees (
            id INT PRIMARY KEY,
            name TEXT,
            dept TEXT,
            salary FLOAT
        );

        INSERT INTO employees VALUES
            (1, 'Alice', 'Engineering', 120000),
            (2, 'Bob', 'Engineering', 105000),
            (3, 'Carol', 'Marketing', 95000),
            (4, 'Dave', 'Marketing', 88000),
            (5, 'Eve', 'Engineering', 130000),
            (6, 'Frank', 'Sales', 78000);
    """

    catalog, _ = Catalog.from_sql(ddl_dml)
    print(f"\n  Created tables: {catalog.table_names()}")

    # Query: average salary by department
    sql = """
        SELECT dept, AVG(salary) AS avg_salary, COUNT(*) AS headcount
        FROM employees
        GROUP BY dept
        ORDER BY avg_salary DESC
    """
    print(f"  Query: {sql.strip()}")

    net = parse_query(sql, catalog)
    run()
    print_network(net, show_rows=True)

    return net


def demo_provenance():
    """Demo 4: Row-level provenance tracking with parsed SQL."""
    print("\n" + "="*70)
    print(" DEMO 4: Provenance tracking")
    print(" Trace output rows back through parsed SQL operators")
    print("="*70)

    initialize_scheduler()
    catalog = load_sample_catalog()

    sql = """
        SELECT name, amount
        FROM orders
        JOIN customers ON cust_id = id
        WHERE amount > 100
    """
    print(f"\n  SQL: {sql.strip()}")

    net = parse_query(sql, catalog)
    run()

    # Pick an output row and trace it
    output = net.output_cell.content
    if output and hasattr(output, 'rows') and output.rows:
        target_row = output.rows[0]
        print(f"\n  Tracing row: {target_row}")
        chain = explain_row(target_row, net)
        print_provenance(chain, target_row)

        # Show why a row that doesn't exist might be missing
        fake_row = {"name": "Dave", "amount": 50}
        print(f"\n  Why is this row missing? {fake_row}")
        missing_trace = why_missing(fake_row, net)
        for cell_name, op_type, status, detail in missing_trace:
            icon = "+" if status == "present" else "x" if status == "missing" else "!"
            print(f"    [{icon}] [{cell_name}] {detail}")

    return net


def demo_repair():
    """Demo 5: Query repair — reference query comparison (QR-Hint style).

    Given a correct reference query and a wrong working query,
    generate clause-level hints about what to fix.
    """
    print("\n" + "="*70)
    print(" DEMO 5: Hint Oracle (QR-Hint style, non-leaking)")
    print(" Graduated hints that identify errors WITHOUT revealing the answer")
    print("="*70)

    catalog = load_sample_catalog()

    # Reference (correct) query — kept secret by the oracle
    reference_sql = """
        SELECT name, amount
        FROM orders
        JOIN customers ON cust_id = id
        WHERE amount > 100
    """

    # Working (wrong) query — wrong threshold in WHERE
    working_sql = """
        SELECT name, amount
        FROM orders
        JOIN customers ON cust_id = id
        WHERE amount > 200
    """

    print(f"\n  Student's query: {working_sql.strip()}")
    print(f"  (Reference query is hidden from student)")

    # --- Level 1: Which clause? ---
    print("\n" + "-"*50)
    print(" Level 1 hints (clause identification only):")
    print("-"*50)
    report_l1 = diagnose_from_reference(working_sql, reference_sql, catalog,
                                         hint_level=HintLevel.CLAUSE)
    print_repair_report(report_l1)

    # --- Level 2: What kind of error? ---
    print("\n" + "-"*50)
    print(" Level 2 hints (error characterization):")
    print("-"*50)
    report_l2 = diagnose_from_reference(working_sql, reference_sql, catalog,
                                         hint_level=HintLevel.CHARACTER)
    print_repair_report(report_l2)

    # --- Level 3: Narrowed direction ---
    print("\n" + "-"*50)
    print(" Level 3 hints (narrowed direction):")
    print("-"*50)
    report_l3 = diagnose_from_reference(working_sql, reference_sql, catalog,
                                         hint_level=HintLevel.DIRECTION)
    print_repair_report(report_l3)

    # --- Second example: missing GROUP BY, wrong SELECT ---
    print("\n" + "-"*70)
    print(" Repair Example 2: Structural differences (Level 3)")
    print("-"*70)

    reference_sql2 = """
        SELECT name, SUM(amount) AS total
        FROM orders
        JOIN customers ON cust_id = id
        GROUP BY name
    """

    working_sql2 = """
        SELECT name, amount
        FROM orders
        JOIN customers ON cust_id = id
    """

    print(f"\n  Student's query: {working_sql2.strip()}")
    print(f"  (Reference query is hidden from student)")

    report2 = diagnose_from_reference(working_sql2, reference_sql2, catalog,
                                       hint_level=HintLevel.DIRECTION)
    print_repair_report(report2)

    return report_l3


def demo_repair_complexity():
    """Demo 6: Repair complexity progression.

    Shows how hints scale from trivial single-predicate errors to
    multi-clause, multi-site repairs. Each example shows BOTH queries
    (for demo purposes) and the graduated hints the oracle produces.
    """
    print("\n" + "="*70)
    print(" DEMO 6: Repair Complexity Progression")
    print(" How hints grow from simple → complex errors")
    print("="*70)

    catalog = load_sample_catalog()

    examples = [
        # ── Complexity 1: Single literal value wrong ──
        {
            "title": "1. Single Literal (trivial)",
            "desc": "Wrong constant in WHERE — simplest possible error",
            "reference": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount > 100
            """,
            "working": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount > 200
            """,
        },

        # ── Complexity 2: Wrong comparison operator ──
        {
            "title": "2. Wrong Operator",
            "desc": "Correct value, wrong comparison direction",
            "reference": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount > 100
            """,
            "working": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount < 100
            """,
        },

        # ── Complexity 3: Missing WHERE clause ──
        {
            "title": "3. Missing WHERE",
            "desc": "Student forgot the filter entirely",
            "reference": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount > 100
            """,
            "working": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
            """,
        },

        # ── Complexity 4: Wrong column in SELECT ──
        {
            "title": "4. Wrong SELECT Column",
            "desc": "Selected wrong column — amount instead of city",
            "reference": """
                SELECT name, city FROM customers
                WHERE city = 'NYC'
            """,
            "working": """
                SELECT name, id FROM customers
                WHERE city = 'NYC'
            """,
        },

        # ── Complexity 5: Missing GROUP BY + aggregate ──
        {
            "title": "5. Missing Aggregation",
            "desc": "Forgot GROUP BY and SUM — structural gap",
            "reference": """
                SELECT name, SUM(amount) AS total
                FROM orders JOIN customers ON cust_id = id
                GROUP BY name
            """,
            "working": """
                SELECT name, amount
                FROM orders JOIN customers ON cust_id = id
            """,
        },

        # ── Complexity 6: Wrong table (FROM error) ──
        {
            "title": "6. Wrong Table in FROM",
            "desc": "Querying the wrong table entirely",
            "reference": """
                SELECT pname, price FROM products
                WHERE price > 20
            """,
            "working": """
                SELECT name, amount FROM orders
                WHERE amount > 20
            """,
        },

        # ── Complexity 7: Multiple errors across clauses ──
        {
            "title": "7. Multi-Clause Errors",
            "desc": "Wrong WHERE + wrong SELECT + missing GROUP BY",
            "reference": """
                SELECT name, SUM(amount) AS total
                FROM orders JOIN customers ON cust_id = id
                WHERE amount > 50
                GROUP BY name
            """,
            "working": """
                SELECT name, amount
                FROM orders JOIN customers ON cust_id = id
                WHERE amount > 200
            """,
        },

        # ── Complexity 8: Correct query (equivalence check) ──
        {
            "title": "8. Equivalent Query (no error)",
            "desc": "Both queries are semantically identical",
            "reference": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount > 100
            """,
            "working": """
                SELECT name, amount FROM orders
                JOIN customers ON cust_id = id
                WHERE amount > 100
            """,
        },
    ]

    for i, ex in enumerate(examples):
        print(f"\n{'─'*70}")
        print(f"  {ex['title']}")
        print(f"  {ex['desc']}")
        print(f"{'─'*70}")

        ref = ex['reference']
        wrk = ex['working']

        print(f"\n  Reference (correct):  {' '.join(ref.split())}")
        print(f"  Working   (student):  {' '.join(wrk.split())}")

        # Show all three hint levels for the first 3, then just Level 3
        if i < 3:
            for level in [HintLevel.CLAUSE, HintLevel.CHARACTER, HintLevel.DIRECTION]:
                report = diagnose_from_reference(wrk, ref, catalog, hint_level=level)
                print(f"\n  ── Level {level.value} ({level.name}) ──")
                _print_compact_report(report)
        else:
            report = diagnose_from_reference(wrk, ref, catalog, hint_level=HintLevel.DIRECTION)
            _print_compact_report(report)

    print(f"\n{'═'*70}")
    print(f"  Complexity demo complete: {len(examples)} examples")
    print(f"{'═'*70}")


def demo_lattice_levels():
    """Demo 7: Schema-only vs Estimate vs Full analysis.

    Demonstrates which analyses work WITHOUT row data, with estimates,
    and with full data. This is the foundation for GUI modes.
    """
    print("\n" + "="*70)
    print(" DEMO 7: Lattice-Level Analysis (Schema / Estimate / Full)")
    print(" Which analyses work WITHOUT data?")
    print("="*70)

    from propagator.sql.relation_info import LatticeLevel

    # ── Schema-Only Catalog (no data loaded) ──
    print("\n" + "-"*50)
    print(" Mode 1: Schema-Only (no data)")
    print("-"*50)

    schema_catalog = Catalog()
    schema_catalog.register_table("employees", [
        ColumnDef("id", "int", nullable=False, is_pk=True),
        ColumnDef("name", "text", nullable=False),
        ColumnDef("dept", "text"),
        ColumnDef("salary", "float"),
    ])
    schema_catalog.register_table("departments", [
        ColumnDef("dept_id", "int", nullable=False, is_pk=True),
        ColumnDef("dept_name", "text", nullable=False),
        ColumnDef("budget", "float"),
    ])

    sql = """
        SELECT name, salary FROM employees
        WHERE salary > 50000
    """
    print(f"\n  SQL: {' '.join(sql.split())}")
    print(f"  Data: NONE (schema only)")

    initialize_scheduler()
    net_schema = parse_query(sql, schema_catalog)
    run()

    output = net_schema.output_cell.content if net_schema.output_cell else None
    print(f"  Output type: {type(output).__name__}")
    if hasattr(output, 'columns'):
        print(f"  Columns: {[c.name for c in output.columns]}")
    if hasattr(output, 'rows'):
        print(f"  Rows: {len(output.rows)} (full data)")
    elif hasattr(output, 'row_count'):
        print(f"  Est. rows: {output.row_count:.0f}")
    else:
        print(f"  Rows: N/A (schema only)")

    # Structural repair works with schema-only
    working_sql = "SELECT name, dept FROM employees WHERE salary > 50000"
    reference_sql = "SELECT name, salary FROM employees WHERE salary > 50000"
    report_schema = diagnose_from_reference(working_sql, reference_sql, schema_catalog,
                                             hint_level=HintLevel.DIRECTION)
    print(f"\n  Schema-only repair hints:")
    for h in report_schema.hints:
        if h.severity != 'info' or not report_schema.equivalent:
            print(f"    [{h.severity[0].upper()}] {h.clause}: {h.message}")
    print(f"  → Structural analysis (FROM, SELECT, GROUP BY) works WITHOUT data")
    print(f"  → WHERE equivalence requires data (predicate evaluation)")

    # ── Estimate Catalog (statistics, no rows) ──
    print("\n" + "-"*50)
    print(" Mode 2: Estimate (statistics, no rows)")
    print("-"*50)

    est_catalog = Catalog()
    est_catalog.register_table("employees", [
        ColumnDef("id", "int"), ColumnDef("name", "text"),
        ColumnDef("dept", "text"), ColumnDef("salary", "float"),
    ], row_count=1000, distinct_counts={"id": 1000, "name": 950, "dept": 10, "salary": 500})

    initialize_scheduler()
    net_est = parse_query(sql, est_catalog)
    run()

    output_est = net_est.output_cell.content if net_est.output_cell else None
    print(f"\n  SQL: {' '.join(sql.split())}")
    print(f"  Data: statistics only (1000 rows, no actual data)")
    print(f"  Output type: {type(output_est).__name__}")
    if hasattr(output_est, 'row_count'):
        print(f"  Est. rows after filter: {output_est.row_count:.0f}")
    if hasattr(output_est, 'selectivity'):
        print(f"  Selectivity chain: {output_est.selectivity:.4f}")
    print(f"  → Cardinality estimation works (for optimizer hints)")
    print(f"  → WHERE predicate evaluation NOT available (no rows)")

    # ── Full Catalog (actual data) ──
    print("\n" + "-"*50)
    print(" Mode 3: Full (actual data)")
    print("-"*50)

    full_catalog = load_sample_catalog()
    sql_full = """
        SELECT name, amount FROM orders
        JOIN customers ON cust_id = id
        WHERE amount > 100
    """

    initialize_scheduler()
    net_full = parse_query(sql_full, full_catalog)
    run()

    output_full = net_full.output_cell.content if net_full.output_cell else None
    print(f"\n  SQL: {' '.join(sql_full.split())}")
    print(f"  Data: full (actual rows)")
    print(f"  Output type: {type(output_full).__name__}")
    if hasattr(output_full, 'rows'):
        print(f"  Actual rows: {len(output_full.rows)}")
        for row in output_full.rows[:3]:
            print(f"    {row}")
        if len(output_full.rows) > 3:
            print(f"    ... ({len(output_full.rows) - 3} more)")

    # ── Summary: API capabilities by level ──
    # print(f"\n{'─'*70}")
    # print("  API Capabilities by Lattice Level:")
    # print(f"{'─'*70}")
    # capabilities = [
    #     ("Schema Only", "✓ Column types/names", "✓ FROM/SELECT/GROUP BY structural hints",
    #      "✗ WHERE predicate eval", "✗ Row counts", "✓ Parse + network build"),
    #     ("Estimate",     "✓ Cardinality estimates", "✓ Selectivity chains",
    #      "✗ Actual predicate eval", "✓ Row count estimates", "✓ Optimizer-style analysis"),
    #     ("Full Data",    "✓ Exact row results", "✓ Complete predicate eval",
    #      "✓ Exact row counts", "✓ Provenance tracking", "✓ Full QR-Hint repair"),
    # ]
    # for mode, *caps in capabilities:
    #     print(f"\n  {mode}:")
    #     for c in caps:
    #         print(f"    {c}")

    return report_schema


def _print_compact_report(report):
    """Print a compact single-line-per-hint report."""
    if report.equivalent:
        print(f"    ✓ Correct!")
        return
    count_info = ""
    if report.working_row_count is not None and report.reference_row_count is not None:
        count_info = f" [{report.working_row_count} rows vs {report.reference_row_count} expected]"
    # Show stage results inline
    stages = " ".join(
        f"{'✓' if v else '✗'}{k}"
        for k, v in report.stage_results.items()
    )
    if stages:
        print(f"    Stages: {stages}{count_info}")
    for h in report.hints:
        cost_str = f" [cost={h.cost:.2f}]" if h.cost is not None else ""
        icon = {"error": "!", "warning": "?", "info": "i"}[h.severity]
        print(f"    [{icon}] {h.clause}: {h.message}{cost_str}")


if __name__ == "__main__":
    net1 = demo_basic_query()
    net2 = demo_sql_file_loading()
    net3 = demo_inline_sql()
    net4 = demo_provenance()
    net5 = demo_repair()
    demo_repair_complexity()
    demo_lattice_levels()

    print("\n" + "="*70)
    print(" All demos complete.")
    print("="*70)
