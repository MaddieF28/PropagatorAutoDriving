"""Systematic tests for the SQL propagator framework.

Run with:
    python3 -m propagator.sql.test_sql
"""

import math
from propagator import Cell, scheduler
from propagator.nothing import nothing_p
from propagator.sql.relation_info import (
    ColumnDef, SchemaInfo, EstimateInfo, FullRelation, LatticeLevel,
)
from propagator.sql.sql_merge import register_sql_merge
from propagator.sql.catalog import Catalog
from propagator.sql.operators import (
    scan, filter_prop, join_prop, aggregate_prop, project_prop,
    sort_prop, limit_prop, union_prop,
)
from propagator.sql.constraints import filter_constraint, join_constraint
from propagator.sql.parser import parse_query, parse_sql_statements, _compile_expr, _extract_aggregates
from propagator.sql.network import build_network_manual, QueryNetwork
from propagator.sql.stepper import QueryStepper
from propagator.sql.diagnostics import cardinality_surprise, join_explosion, diagnose_network
from propagator.sql.provenance import explain_row, why_missing
from propagator.sql.repair import (
    diagnose_from_reference, HintLevel, print_repair_report,
)

import sqlglot
from sqlglot import exp

register_sql_merge()

# ── Helpers ─────────────────────────────────────────────────────────────

def _make_catalog():
    """Build a small test catalog."""
    return Catalog.from_dicts({
        "employees": [
            {"id": 1, "name": "Alice", "dept": "eng", "salary": 100},
            {"id": 2, "name": "Bob", "dept": "eng", "salary": 120},
            {"id": 3, "name": "Carol", "dept": "sales", "salary": 90},
            {"id": 4, "name": "Dave", "dept": "sales", "salary": 110},
            {"id": 5, "name": "Eve", "dept": "hr", "salary": 95},
        ],
        "departments": [
            {"dept": "eng", "budget": 500},
            {"dept": "sales", "budget": 300},
            {"dept": "hr", "budget": 200},
        ],
    })

passed = 0
failed = 0

def check(description, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {description}")


# ── 1. Lattice types ───────────────────────────────────────────────────

def test_lattice_levels():
    print("Testing lattice levels...")
    cols = [ColumnDef("a", "INT"), ColumnDef("b", "TEXT")]
    s = SchemaInfo(cols)
    e = EstimateInfo(cols, row_count=100, distinct_counts={"a": 50})
    f = FullRelation(cols, rows=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])

    check("SchemaInfo level", s.level == LatticeLevel.SCHEMA)
    check("EstimateInfo level", e.level == LatticeLevel.ESTIMATE)
    check("FullRelation level", f.level == LatticeLevel.FULL)
    check("column_names from SchemaInfo", s.column_names == ["a", "b"])
    check("FullRelation row_count", f.row_count == 2)
    check("FullRelation equality", f == FullRelation(cols, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]))
    check("FullRelation inequality", f != FullRelation(cols, [{"a": 1, "b": "x"}]))


# ── 2. Merge ────────────────────────────────────────────────────────────

def test_merge():
    print("Testing merge operations...")
    from propagator import Cell
    from propagator.scheduler import run as run_scheduler

    cols = [ColumnDef("x", "INT")]
    s = SchemaInfo(cols)
    e = EstimateInfo(cols, row_count=10, distinct_counts={})
    f = FullRelation(cols, [{"x": 1}])

    # Higher level subsumes lower
    c = Cell()
    c.add_content(s)
    c.add_content(e)
    check("Estimate subsumes Schema", isinstance(c.content, EstimateInfo))

    c2 = Cell()
    c2.add_content(e)
    c2.add_content(f)
    check("Full subsumes Estimate", isinstance(c2.content, FullRelation))

    # Same-type: same value keeps it
    c3 = Cell()
    c3.add_content(f)
    c3.add_content(FullRelation(cols, [{"x": 1}]))
    check("Same FullRelation merges", c3.content == f)

    # Same-type contradiction
    try:
        c4 = Cell()
        c4.add_content(FullRelation(cols, [{"x": 1}]))
        c4.add_content(FullRelation(cols, [{"x": 2}]))
        check("Contradiction on mismatched FullRelation", False)
    except Exception:
        check("Contradiction on mismatched FullRelation", True)


# ── 3. Operators at FullRelation level ──────────────────────────────────

def test_operators():
    print("Testing operators...")
    from propagator.scheduler import run as run_scheduler

    cat = _make_catalog()
    full = cat.get_table("employees", LatticeLevel.FULL)

    # scan
    out = Cell()
    scan("employees", cat, out)
    run_scheduler()
    check("scan propagates", not nothing_p(out.content))
    check("scan preserves rows", out.content.row_count == 5)

    # filter
    inp2 = Cell()
    out2 = Cell()
    filter_prop(inp2, lambda r: r["dept"] == "eng", out2)
    inp2.add_content(full)
    run_scheduler()
    check("filter result count", out2.content.row_count == 2)
    check("filter correct rows", all(r["dept"] == "eng" for r in out2.content.rows))

    # project
    inp3 = Cell()
    out3 = Cell()
    project_prop(inp3, ["name"], out3)
    inp3.add_content(full)
    run_scheduler()
    check("project column count", len(out3.content.columns) == 1)
    check("project has name col", out3.content.column_names == ["name"])

    # aggregate
    inp4 = Cell()
    out4 = Cell()
    aggregate_prop(inp4, ["dept"],
                   [("total_sal", lambda rows: sum(r["salary"] for r in rows)),
                    ("cnt", lambda rows: len(rows))],
                   out4)
    inp4.add_content(full)
    run_scheduler()
    check("aggregate groups", out4.content.row_count == 3)
    eng_row = [r for r in out4.content.rows if r["dept"] == "eng"][0]
    check("aggregate sum", eng_row["total_sal"] == 220)
    check("aggregate count", eng_row["cnt"] == 2)

    # sort
    inp5 = Cell()
    out5 = Cell()
    sort_prop(inp5, [("salary", "desc")], out5)
    inp5.add_content(full)
    run_scheduler()
    salaries = [r["salary"] for r in out5.content.rows]
    check("sort descending", salaries == sorted(salaries, reverse=True))

    # limit
    inp6 = Cell()
    out6 = Cell()
    limit_prop(inp6, 3, out6)
    inp6.add_content(out5.content)
    run_scheduler()
    check("limit count", out6.content.row_count == 3)

    # join — use renamed dept key to avoid collision in combined dict
    dept_full = cat.get_table("departments", LatticeLevel.FULL)
    dept_full_renamed = FullRelation(
        [ColumnDef("d_dept"), ColumnDef("budget")],
        [{"d_dept": r["dept"], "budget": r["budget"]} for r in dept_full.rows]
    )
    left_cell = Cell()
    right_cell = Cell()
    join_out = Cell()
    join_prop(left_cell, right_cell,
              lambda r: r["dept"] == r["d_dept"],
              join_out)
    left_cell.add_content(full)
    right_cell.add_content(dept_full_renamed)
    run_scheduler()
    check("join result count", join_out.content.row_count == 5)
    check("join has budget col", "budget" in join_out.content.column_names)

    # union
    u_left = Cell()
    u_right = Cell()
    u_out = Cell()
    small1 = FullRelation([ColumnDef("x", "INT")], [{"x": 1}, {"x": 2}])
    small2 = FullRelation([ColumnDef("x", "INT")], [{"x": 3}])
    union_prop(u_left, u_right, u_out)
    u_left.add_content(small1)
    u_right.add_content(small2)
    run_scheduler()
    check("union count", u_out.content.row_count == 3)


# ── 4. Expression compiler ──────────────────────────────────────────────

def test_compile_expr():
    print("Testing expression compiler...")

    def compile_sql_expr(sql_fragment):
        """Parse a SQL expression and compile it."""
        # Wrap in SELECT to make it parseable
        tree = sqlglot.parse_one(f"SELECT * FROM t WHERE {sql_fragment}")
        where_node = tree.find(exp.Where).this
        return _compile_expr(where_node)

    row = {"a": 10, "b": 5, "name": "Alice", "status": None}

    # Comparisons
    fn = compile_sql_expr("a > 7")
    check("GT true", fn(row) == True)
    fn = compile_sql_expr("a < 7")
    check("GT false", fn(row) == False)
    fn = compile_sql_expr("a = 10")
    check("EQ true", fn(row) == True)
    fn = compile_sql_expr("a != 10")
    check("NEQ", fn(row) == False)
    fn = compile_sql_expr("a >= 10")
    check("GTE", fn(row) == True)
    fn = compile_sql_expr("a <= 10")
    check("LTE", fn(row) == True)

    # AND / OR / NOT
    fn = compile_sql_expr("a > 5 AND b < 10")
    check("AND true", fn(row) == True)
    fn = compile_sql_expr("a > 5 AND b > 10")
    check("AND false", fn(row) == False)
    fn = compile_sql_expr("a > 100 OR b < 10")
    check("OR true", fn(row) == True)
    fn = compile_sql_expr("NOT a > 100")
    check("NOT", fn(row) == True)

    # IN
    fn = compile_sql_expr("a IN (10, 20, 30)")
    check("IN true", fn(row) == True)
    fn = compile_sql_expr("a IN (1, 2, 3)")
    check("IN false", fn(row) == False)

    # BETWEEN
    fn = compile_sql_expr("a BETWEEN 5 AND 15")
    check("BETWEEN true", fn(row) == True)
    fn = compile_sql_expr("a BETWEEN 11 AND 15")
    check("BETWEEN false", fn(row) == False)

    # IS NULL / IS NOT NULL
    fn = compile_sql_expr("status IS NULL")
    check("IS NULL true", fn(row) == True)
    fn = compile_sql_expr("name IS NULL")
    check("IS NULL false (non-null)", fn(row) == False)

    # LIKE
    fn = compile_sql_expr("name LIKE 'Ali%'")
    check("LIKE prefix", fn(row) == True)
    fn = compile_sql_expr("name LIKE '%Bob%'")
    check("LIKE no match", fn(row) == False)

    # Arithmetic in expressions
    fn = compile_sql_expr("a + b > 14")
    check("arithmetic add", fn(row) == True)
    fn = compile_sql_expr("a * b = 50")
    check("arithmetic mul", fn(row) == True)
    fn = compile_sql_expr("a - b = 5")
    check("arithmetic sub", fn(row) == True)


# ── 5. Aggregate extraction ─────────────────────────────────────────────

def test_extract_aggregates():
    print("Testing aggregate extraction...")

    tree = sqlglot.parse_one("SELECT dept, COUNT(id), SUM(salary) AS total FROM t GROUP BY dept")
    select_exprs = tree.find(exp.Select).expressions
    group_cols, agg_exprs, proj_cols, has_aggs = _extract_aggregates(select_exprs)

    check("has aggregates", has_aggs == True)
    check("extracted 2 aggregates", len(agg_exprs) == 2)
    check("group col is dept", "dept" in group_cols)
    check("project cols count", len(proj_cols) == 3)  # dept, count, total


# ── 6. Parser integration ───────────────────────────────────────────────

def test_parser():
    print("Testing parser integration...")
    cat = _make_catalog()

    # Simple SELECT
    net = parse_query("SELECT name, salary FROM employees WHERE dept = 'eng'", cat)
    check("network has output", net.output_cell is not None)
    check("network has cells", len(net.cells) > 0)
    val = net.output_cell.content
    check("output has rows", val is not None and hasattr(val, 'rows'))
    check("filter applied", val.row_count == 2)
    check("project applied", set(val.column_names) == {"name", "salary"})

    # JOIN query
    net2 = parse_query(
        "SELECT e.name, d.budget FROM employees e JOIN departments d ON e.dept = d.dept",
        cat
    )
    val2 = net2.output_cell.content
    check("join query rows", val2.row_count == 5)
    check("join query cols", "budget" in val2.column_names)

    # GROUP BY + aggregate
    net3 = parse_query(
        "SELECT dept, COUNT(id) AS cnt FROM employees GROUP BY dept",
        cat
    )
    val3 = net3.output_cell.content
    check("group by rows", val3.row_count == 3)
    check("group by has cnt", "cnt" in val3.column_names)

    # ORDER BY + LIMIT
    net4 = parse_query(
        "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 2",
        cat
    )
    val4 = net4.output_cell.content
    check("limit rows", val4.row_count == 2)
    salaries = [r["salary"] for r in val4.rows]
    check("order desc", salaries[0] >= salaries[1])


# ── 7. SQL file / statements parsing ────────────────────────────────────

def test_parse_sql_statements():
    print("Testing parse_sql_statements...")
    sql = """
    CREATE TABLE items (id INTEGER, name TEXT, price REAL);
    INSERT INTO items VALUES (1, 'Widget', 9.99);
    INSERT INTO items VALUES (2, 'Gadget', 19.99);
    INSERT INTO items VALUES (3, 'Doohickey', 4.99);
    SELECT name, price FROM items WHERE price > 5;
    """
    cat, networks = parse_sql_statements(sql)
    check("statements: network created", len(networks) > 0)
    net = networks[0]
    check("statements: catalog has items", cat.get_table("items") is not None)
    val = net.output_cell.content
    check("statements: filter applied", val.row_count == 2)


# ── 8. Stepper ──────────────────────────────────────────────────────────

def test_stepper():
    print("Testing stepper...")
    cat = _make_catalog()
    net = parse_query("SELECT name FROM employees WHERE dept = 'eng'", cat)
    stepper = QueryStepper(net, cat)

    level = stepper.current_level()
    check("stepper level is FULL", level == LatticeLevel.FULL)
    snap = stepper.snapshot()
    check("snapshot is dict", isinstance(snap, dict))


# ── 9. Diagnostics ─────────────────────────────────────────────────────

def test_diagnostics():
    print("Testing diagnostics...")

    # cardinality_surprise returns None for FullRelation
    c = Cell()
    cols = [ColumnDef("x", "INT")]
    c.add_content(FullRelation(cols, [{"x": 1}, {"x": 2}]))
    check("no surprise for FullRelation", cardinality_surprise(c) is None)

    # cardinality_surprise returns None for EstimateInfo (no actual rows)
    c2 = Cell()
    c2.add_content(EstimateInfo(cols, 100))
    check("no surprise for EstimateInfo", cardinality_surprise(c2) is None)

    # diagnose_network on a simple network
    cat = _make_catalog()
    net = parse_query("SELECT * FROM employees", cat)
    issues = diagnose_network(net)
    check("diagnose_network returns list", isinstance(issues, list))


# ── 10. Provenance ─────────────────────────────────────────────────────

def test_provenance():
    print("Testing provenance...")
    cat = _make_catalog()
    net = parse_query("SELECT name, salary FROM employees WHERE salary > 100", cat)
    val = net.output_cell.content

    # explain_row for a row that exists in output
    if val.row_count > 0:
        row = val.rows[0]
        lineage = explain_row(row, net)
        check("explain_row returns list", isinstance(lineage, list))
        check("explain_row non-empty", len(lineage) > 0)

    # why_missing for a row filtered out
    missing = {"name": "Carol", "salary": 90}
    reasons = why_missing(missing, net)
    check("why_missing returns list", isinstance(reasons, list))


# ── 11. Repair / Hint Oracle ───────────────────────────────────────────

def test_repair():
    print("Testing repair hint oracle...")
    cat = _make_catalog()

    working = "SELECT name, salary FROM employees WHERE salary > 200"
    reference = "SELECT name, salary FROM employees WHERE salary > 100"

    # All three hint levels
    for level in [HintLevel.CLAUSE, HintLevel.CHARACTER, HintLevel.DIRECTION]:
        report = diagnose_from_reference(working, reference, cat, hint_level=level)
        check(f"repair level={level.name} has hints", len(report.hints) > 0)
        check(f"repair level={level.name} not equivalent", not report.equivalent)

    # Equivalent queries
    report_eq = diagnose_from_reference(
        "SELECT name, salary FROM employees WHERE salary > 100",
        "SELECT name, salary FROM employees WHERE salary > 100",
        cat,
    )
    check("equivalent queries detected", report_eq.equivalent)

    # DIRECTION level should have message content
    report_dir = diagnose_from_reference(working, reference, cat, hint_level=HintLevel.DIRECTION)
    for hint in report_dir.hints:
        check(f"hint has message", hint.message is not None and len(hint.message) > 0)
        check(f"hint has clause", hint.clause is not None)

    where_cmp = report_dir.approach_comparison.get("WHERE", {})
    check("where comparison includes selected approach", where_cmp.get("selected") is not None)
    check("where comparison includes network_data", "network_data" in where_cmp)
    check("where comparison includes network_tms", "network_tms" in where_cmp)
    check("where comparison includes qr_hint_ast", "qr_hint_ast" in where_cmp)


# ── 11b. Propagator-based repair internals ─────────────────────────────

def test_repair_propagator_internals():
    """Test that repair functions actually use propagator merge / TMS."""
    print("Testing propagator-based repair internals...")
    from propagator.merge import merge, contradictory_p
    from propagator.sql.repair import (
        _outputs_equivalent, _test_predicate_equivalence,
        _find_minimal_predicate_repairs, _canonicalize_relation,
        _safe_compile, _safe_eval,
        _find_minimal_predicate_repairs_qr_hint,
    )

    cols = [ColumnDef("x"), ColumnDef("y")]

    # ── _outputs_equivalent uses merge ──
    r1 = FullRelation(cols, [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    r2 = FullRelation(cols, [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    check("outputs_equivalent: same relations", _outputs_equivalent(r1, r2))

    r3 = FullRelation(cols, [{"x": 1, "y": 2}])
    check("outputs_equivalent: different row count", not _outputs_equivalent(r1, r3))

    r4 = FullRelation(cols, [{"x": 1, "y": 99}, {"x": 3, "y": 4}])
    check("outputs_equivalent: different values", not _outputs_equivalent(r1, r4))

    # ── _canonicalize_relation strips aliases and sorts ──
    r_aliased = FullRelation(cols, [
        {"x": 3, "y": 4, "t.x": 3, "t.y": 4},
        {"x": 1, "y": 2, "t.x": 1, "t.y": 2},
    ])
    canon = _canonicalize_relation(r_aliased)
    check("canonicalize strips alias keys",
          all("." not in k for row in canon.rows for k in row))
    # Rows should be sorted deterministically
    check("canonicalize sorts rows", canon.rows[0]["x"] <= canon.rows[-1]["x"])

    # ── _outputs_equivalent ignores alias key differences ──
    r_no_alias = FullRelation(cols, [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    check("outputs_equivalent: alias-agnostic", _outputs_equivalent(r_aliased, r_no_alias))

    # ── _test_predicate_equivalence uses merge per-row ──
    fn_gt100 = lambda r: r["salary"] > 100
    fn_gt200 = lambda r: r["salary"] > 200
    rows = [{"salary": 50}, {"salary": 150}, {"salary": 250}]
    check("pred_equiv: same predicate", _test_predicate_equivalence(fn_gt100, fn_gt100, rows))
    check("pred_equiv: different predicates", not _test_predicate_equivalence(fn_gt100, fn_gt200, rows))

    # Empty rows → vacuously equivalent
    check("pred_equiv: empty rows", _test_predicate_equivalence(fn_gt100, fn_gt200, []))

    # ── _find_minimal_predicate_repairs uses TMS ──
    # Single mismatched predicate — extract from full WHERE clauses
    w_ast = sqlglot.parse_one("SELECT * FROM t WHERE salary > 200")
    r_ast = sqlglot.parse_one("SELECT * FROM t WHERE salary > 100")
    p1 = w_ast.find(exp.Where).this  # salary > 200
    p2 = r_ast.find(exp.Where).this  # salary > 100
    test_rows = [{"salary": 50}, {"salary": 150}, {"salary": 250}]

    sites, cost = _find_minimal_predicate_repairs([p1], [p2], test_rows)
    check("tms_repair: finds single site", sites == [0])
    check("tms_repair: cost > 0", cost > 0)

    # Identical predicates → no repair needed
    sites2, cost2 = _find_minimal_predicate_repairs([p2], [p2], test_rows)
    check("tms_repair: identical → empty sites", sites2 == [])
    check("tms_repair: identical → zero cost", cost2 == 0.0)

    # Two predicates, one wrong
    w2_ast = sqlglot.parse_one("SELECT * FROM t WHERE name = 'Alice'")
    r2_ast = sqlglot.parse_one("SELECT * FROM t WHERE name = 'Bob'")
    p3 = w2_ast.find(exp.Where).this  # name = 'Alice'
    p4 = r2_ast.find(exp.Where).this  # name = 'Bob'
    multi_rows = [
        {"salary": 50, "name": "Alice"},
        {"salary": 150, "name": "Bob"},
        {"salary": 250, "name": "Carol"},
    ]
    sites3, cost3 = _find_minimal_predicate_repairs(
        [p2, p3],  # working: salary > 100 AND name = 'Alice'
        [p2, p4],  # reference: salary > 100 AND name = 'Bob'
        multi_rows,
    )
    check("tms_repair: finds correct mismatch index", sites3 == [1])

    # Empty rows → no repair needed
    sites4, cost4 = _find_minimal_predicate_repairs([p1], [p2], [])
    check("tms_repair: empty rows → no repair", sites4 == [])

    qr_result = _find_minimal_predicate_repairs_qr_hint(p1, p2, test_rows)
    check("qr_hint_ast: viable on simple threshold repair", qr_result.viable)
    check("qr_hint_ast: finds at least one site", len(qr_result.repair_sites) >= 1)

    # ── _stage_from uses merge (integration via diagnose_from_reference) ──
    cat = _make_catalog()
    report = diagnose_from_reference(
        "SELECT name FROM employees",
        "SELECT name FROM departments",
        cat,
        hint_level=HintLevel.DIRECTION,
    )
    check("stage_from: merge detects table mismatch", report.stage_results.get('FROM') == False)
    check("stage_from: generates FROM hints",
          any(h.clause == "FROM" for h in report.hints))

    # Same tables, different WHERE → FROM passes, WHERE fails
    report2 = diagnose_from_reference(
        "SELECT name, salary FROM employees WHERE salary > 200",
        "SELECT name, salary FROM employees WHERE salary > 100",
        cat,
    )
    check("stage_from: merge accepts same tables", report2.stage_results.get('FROM') == True)

    # ── _stage_group_by uses merge on column sets ──
    # Same grouping but different SELECT → GROUP BY passes
    report3 = diagnose_from_reference(
        "SELECT dept, COUNT(*) FROM employees GROUP BY dept",
        "SELECT dept, SUM(salary) FROM employees GROUP BY dept",
        cat,
    )
    check("stage_group_by: merge accepts same columns",
          report3.stage_results.get('GROUP BY') == True)

    report4 = diagnose_from_reference(
        "SELECT dept, COUNT(*) FROM employees GROUP BY dept",
        "SELECT name, COUNT(*) FROM employees GROUP BY name",
        cat,
    )
    check("stage_group_by: merge detects column mismatch",
          report4.stage_results.get('GROUP BY') == False)


# ── 12. Catalog from_dicts / from_sql ──────────────────────────────────

def test_catalog():
    print("Testing catalog...")
    cat = Catalog()
    cat.register_table("t", ["a", "b"], rows=[{"a": 1, "b": "x"}])

    schema = cat.get_table("t", LatticeLevel.SCHEMA)
    check("get schema level", isinstance(schema, SchemaInfo))
    est = cat.get_table("t", LatticeLevel.ESTIMATE)
    check("get estimate level", isinstance(est, EstimateInfo))
    full = cat.get_table("t", LatticeLevel.FULL)
    check("get full level", isinstance(full, FullRelation))
    check("full has rows", full.row_count == 1)

    # from_sql
    cat2, _ = Catalog.from_sql("""
        CREATE TABLE items (id INTEGER, name TEXT);
        INSERT INTO items VALUES (1, 'A');
        INSERT INTO items VALUES (2, 'B');
    """)
    check("from_sql creates table", cat2.get_table("items") is not None)
    full2 = cat2.get_table("items", LatticeLevel.FULL)
    check("from_sql 2 rows", full2.row_count == 2)


# ── 12. Constraints (bidirectional) ────────────────────────────────────

def test_constraints():
    print("Testing constraints...")

    # --- filter_constraint backward propagation ---
    scheduler.initialize_scheduler()

    cols = [ColumnDef("id"), ColumnDef("name"), ColumnDef("dept")]
    input_cell = Cell()
    output_cell = Cell()
    input_rows = [
        {"id": 1, "name": "Alice", "dept": "eng"},
        {"id": 2, "name": "Bob", "dept": "sales"},
        {"id": 3, "name": "Carol", "dept": "eng"},
    ]
    input_cell.add_content(FullRelation(cols, input_rows))

    filter_constraint(input_cell, lambda r: r["dept"] == "eng", output_cell, selectivity=0.5)
    scheduler.run()

    # Forward: output should have filtered rows
    out = output_cell.content
    check("filter_constraint forward works", isinstance(out, FullRelation))
    check("filter_constraint forward count", out.row_count == 2)
    check("filter_constraint forward correct", all(r["dept"] == "eng" for r in out.rows))

    # --- filter_constraint backward: estimate propagation ---
    scheduler.initialize_scheduler()
    in2 = Cell()
    out2 = Cell()
    filter_constraint(in2, lambda r: True, out2, selectivity=0.5)
    out2.add_content(EstimateInfo(cols, 50))
    scheduler.run()

    in_val = in2.content
    check("filter_constraint backward estimate", isinstance(in_val, EstimateInfo))
    check("filter_constraint backward row_count >= output",
          in_val.row_count >= 50)

    # --- join_constraint forward ---
    scheduler.initialize_scheduler()

    left_cols = [ColumnDef("id"), ColumnDef("name")]
    right_cols = [ColumnDef("dept_id"), ColumnDef("dept")]

    # Test forward only with schema level (no backward contradiction)
    l_schema = Cell()
    r_schema = Cell()
    j_out = Cell()
    l_schema.add_content(SchemaInfo(left_cols))
    r_schema.add_content(SchemaInfo(right_cols))
    join_constraint(l_schema, r_schema, lambda r: True, j_out)
    scheduler.run()

    jout = j_out.content
    check("join_constraint forward schema", isinstance(jout, SchemaInfo))
    check("join_constraint forward schema cols",
          len(jout.columns) == len(left_cols) + len(right_cols))

    # --- join_constraint backward schema propagation ---
    scheduler.initialize_scheduler()
    l = Cell()
    r = Cell()
    o = Cell()
    l.add_content(SchemaInfo(left_cols))
    r.add_content(SchemaInfo(right_cols))
    join_constraint(l, r, lambda row: True, o)
    o.add_content(SchemaInfo(left_cols + right_cols))
    scheduler.run()

    check("join_constraint backward schema left",
          not nothing_p(l.content) and isinstance(l.content, SchemaInfo))
    check("join_constraint backward schema right",
          not nothing_p(r.content) and isinstance(r.content, SchemaInfo))


# ── 13. Stepper step_to ────────────────────────────────────────────────

def test_stepper_step_to():
    print("Testing stepper step_to...")

    cat = _make_catalog()

    # Build network but only at schema level initially
    scheduler.initialize_scheduler()
    net = parse_query("SELECT * FROM employees WHERE salary > 100", cat)
    scheduler.run()

    stepper = QueryStepper(net, catalog=cat)

    # current_level should already be FULL (catalog has full data)
    lvl = stepper.current_level()
    check("stepper current_level not None", lvl is not None)

    # cell_levels returns dict
    lvls = stepper.cell_levels()
    check("cell_levels returns dict", isinstance(lvls, dict))
    check("cell_levels has entries", len(lvls) > 0)

    # snapshot returns dict with (level, summary) tuples
    snap = stepper.snapshot()
    check("snapshot returns dict", isinstance(snap, dict))
    check("snapshot has entries", len(snap) > 0)
    for name, (level, summary) in snap.items():
        check(f"snapshot {name} has summary string", isinstance(summary, str))

    # print_state doesn't crash
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        stepper.print_state(header="Test")
        stepper.print_state()  # without header
    finally:
        sys.stdout = old_stdout
    check("print_state runs without error", True)

    # Test step_to with a fresh schema-only network
    scheduler.initialize_scheduler()
    from propagator.sql.catalog import Catalog
    schema_cat = Catalog()
    schema_cat.register_table(
        "t", [ColumnDef("x", "INT")],
        row_count=10, distinct_counts={"x": 5},
    )
    net2 = QueryNetwork()
    from propagator.sql.operators import scan
    c = net2.add_cell("scan_t")
    scan("t", schema_cat, c)
    net2.output_cell = c
    scheduler.run()

    stepper2 = QueryStepper(net2, catalog=schema_cat)
    check("schema-only stepper level", stepper2.current_level() == LatticeLevel.ESTIMATE)

    snap2 = stepper2.step_to(LatticeLevel.SCHEMA)
    check("step_to SCHEMA returns snapshot", isinstance(snap2, dict))


# ── Run all tests ───────────────────────────────────────────────────────

if __name__ == "__main__":
    test_lattice_levels()
    test_merge()
    test_operators()
    test_compile_expr()
    test_extract_aggregates()
    test_parser()
    test_parse_sql_statements()
    test_stepper()
    test_diagnostics()
    test_provenance()
    test_repair()
    test_repair_propagator_internals()
    test_catalog()
    test_constraints()
    test_stepper_step_to()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)
