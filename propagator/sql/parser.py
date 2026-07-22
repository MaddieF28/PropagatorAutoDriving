"""SQL-to-propagator-network parser using sqlglot.

Translates real SQL queries into propagator networks by walking the
sqlglot AST and emitting the appropriate relational algebra operators.

Usage:
    from propagator.sql.parser import parse_query
    net = parse_query("SELECT * FROM orders WHERE amount > 100", catalog)

Supports:
    - SELECT with column lists, aliases, *, aggregate functions
    - FROM with table aliases
    - JOIN (INNER, LEFT, CROSS) with ON conditions
    - WHERE with comparison, AND/OR, IN, BETWEEN, IS NULL, LIKE, NOT
    - GROUP BY with COUNT, SUM, AVG, MIN, MAX (including DISTINCT)
    - ORDER BY (ASC/DESC)
    - LIMIT
"""

import operator as op_mod
import re
from typing import Any, Callable, Optional

import sqlglot
from sqlglot import exp

from .network import QueryNetwork, build_network_manual


# Default-catalog construction is injected by catalog.py at import time,
# so parser.py never needs to import catalog.py (which itself depends on
# parser.py for its from_sql/from_sql_file convenience methods) -- same
# hook pattern as register_tms_initializer/register_cdcl_handlers.
_default_catalog_factory: Optional[Callable[[], Any]] = None


def register_default_catalog_factory(factory: Callable[[], Any]) -> None:
    """Register the factory used to create a Catalog when none is passed."""
    global _default_catalog_factory
    _default_catalog_factory = factory


# ============================================================================
# Expression compilation: sqlglot AST node -> Python callable
# ============================================================================

def _compile_expr(node):
    """Compile a sqlglot expression node into a Python callable(row_dict) -> value.

    For predicates (WHERE, ON conditions), returns callable(row_dict) -> bool.
    For value expressions (SELECT list), returns callable(row_dict) -> value.
    """
    if isinstance(node, exp.Column):
        col_name = node.name
        table = node.table
        # Return a function that looks up the column in the row dict.
        # Try qualified name first (e.g. "o.amount"), then bare name.
        if table:
            def col_fn(row, _c=col_name, _t=table):
                key = f"{_t}.{_c}"
                if key in row:
                    return row[key]
                return row.get(_c)
            return col_fn
        else:
            return lambda row, _c=col_name: row.get(_c)

    if isinstance(node, exp.Literal):
        if node.is_string:
            val = node.this
            return lambda row, _v=val: _v
        else:
            val = _parse_number(node.this)
            return lambda row, _v=val: _v

    if isinstance(node, exp.Boolean):
        val = node.this
        return lambda row, _v=val: _v

    if isinstance(node, exp.Null):
        return lambda row: None

    if isinstance(node, exp.Star):
        return None  # special: means "all columns"

    # Comparison operators
    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        op_map = {
            exp.EQ: op_mod.eq,
            exp.NEQ: op_mod.ne,
            exp.GT: op_mod.gt,
            exp.GTE: op_mod.ge,
            exp.LT: op_mod.lt,
            exp.LTE: op_mod.le,
        }
        op_fn = op_map[type(node)]
        return lambda row, _l=left_fn, _r=right_fn, _op=op_fn: _op(_l(row), _r(row))

    # Logical operators
    if isinstance(node, exp.And):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        return lambda row, _l=left_fn, _r=right_fn: _l(row) and _r(row)

    if isinstance(node, exp.Or):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        return lambda row, _l=left_fn, _r=right_fn: _l(row) or _r(row)

    if isinstance(node, exp.Not):
        inner_fn = _compile_expr(node.this)
        return lambda row, _f=inner_fn: not _f(row)

    # IN expression
    if isinstance(node, exp.In):
        left_fn = _compile_expr(node.this)
        values = []
        for child in node.expressions:
            values.append(_compile_expr(child))
        return lambda row, _l=left_fn, _vs=values: _l(row) in [v(row) for v in _vs]

    # BETWEEN
    if isinstance(node, exp.Between):
        expr_fn = _compile_expr(node.this)
        low_fn = _compile_expr(node.args["low"])
        high_fn = _compile_expr(node.args["high"])
        return lambda row, _e=expr_fn, _lo=low_fn, _hi=high_fn: _lo(row) <= _e(row) <= _hi(row)

    # IS (NULL)
    if isinstance(node, exp.Is):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        return lambda row, _l=left_fn, _r=right_fn: _l(row) is _r(row)

    # LIKE
    if isinstance(node, exp.Like):
        left_fn = _compile_expr(node.this)
        pattern_fn = _compile_expr(node.expression)
        def like_fn(row, _l=left_fn, _p=pattern_fn):
            val = _l(row)
            pat = _p(row)
            if val is None or pat is None:
                return False
            # Convert SQL LIKE to regex: % -> .*, _ -> .
            # First replace LIKE wildcards with placeholders, escape rest, restore
            s = str(pat)
            parts = []
            for ch in s:
                if ch == '%':
                    parts.append('.*')
                elif ch == '_':
                    parts.append('.')
                else:
                    parts.append(re.escape(ch))
            regex = "^" + "".join(parts) + "$"
            return bool(re.match(regex, str(val), re.IGNORECASE))
        return like_fn

    # Arithmetic
    if isinstance(node, exp.Add):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        return lambda row, _l=left_fn, _r=right_fn: (_l(row) or 0) + (_r(row) or 0)

    if isinstance(node, exp.Sub):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        return lambda row, _l=left_fn, _r=right_fn: (_l(row) or 0) - (_r(row) or 0)

    if isinstance(node, exp.Mul):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        return lambda row, _l=left_fn, _r=right_fn: (_l(row) or 0) * (_r(row) or 0)

    if isinstance(node, exp.Div):
        left_fn = _compile_expr(node.this)
        right_fn = _compile_expr(node.expression)
        def div_fn(row, _l=left_fn, _r=right_fn):
            r = _r(row)
            return (_l(row) or 0) / r if r else None
        return div_fn

    if isinstance(node, exp.Neg):
        inner_fn = _compile_expr(node.this)
        return lambda row, _f=inner_fn: -(_f(row) or 0)

    if isinstance(node, exp.Paren):
        return _compile_expr(node.this)

    # Alias — compile the inner expression
    if isinstance(node, exp.Alias):
        return _compile_expr(node.this)

    # Aggregate functions — these are handled specially in _extract_aggregates
    # but can appear in expressions too; return a placeholder
    if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
        return _compile_aggregate_value_fn(node)

    # Fallback: try to evaluate as a string representation
    raise ValueError(f"Cannot compile expression: {type(node).__name__}: {node}")


def _parse_number(s):
    """Parse a numeric string to int or float."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def _compile_aggregate_value_fn(node):
    """Compile an aggregate function node for use in value contexts."""
    if isinstance(node, exp.Count):
        if node.this and isinstance(node.this, exp.Star):
            return lambda row: 1  # placeholder
        col_fn = _compile_expr(node.this) if node.this else None
        return lambda row, _c=col_fn: 1 if (_c is None or _c(row) is not None) else 0

    inner_fn = _compile_expr(node.this) if node.this else None
    return lambda row, _f=inner_fn: _f(row) if _f else None


# ============================================================================
# Aggregate extraction: pull agg functions from SELECT list
# ============================================================================

def _extract_aggregates(select_expressions):
    """Extract aggregate expressions and plain columns from SELECT list.

    Returns:
        group_columns: list of column names that are plain columns (not aggregates)
        agg_exprs: list of (output_name, agg_fn) for aggregate_prop
        project_columns: list of output column names in SELECT order
        has_aggregates: bool
    """
    group_columns = []
    agg_exprs = []
    project_columns = []
    has_aggregates = False

    for expr in select_expressions:
        alias_name = None
        inner = expr

        if isinstance(expr, exp.Alias):
            alias_name = expr.alias
            inner = expr.this

        if isinstance(inner, exp.Count):
            has_aggregates = True
            name = alias_name or "count"
            distinct = inner.args.get("distinct", False)
            col_node = inner.this
            if col_node and not isinstance(col_node, exp.Star):
                col_fn = _compile_expr(col_node)
                if distinct:
                    agg_fn = lambda rows, _f=col_fn: len(set(_f(r) for r in rows if _f(r) is not None))
                else:
                    agg_fn = lambda rows, _f=col_fn: sum(1 for r in rows if _f(r) is not None)
            else:
                agg_fn = lambda rows: len(rows)
            agg_exprs.append((name, agg_fn))
            project_columns.append(name)

        elif isinstance(inner, exp.Sum):
            has_aggregates = True
            name = alias_name or "sum"
            col_fn = _compile_expr(inner.this)
            agg_fn = lambda rows, _f=col_fn: sum(_f(r) or 0 for r in rows)
            agg_exprs.append((name, agg_fn))
            project_columns.append(name)

        elif isinstance(inner, exp.Avg):
            has_aggregates = True
            name = alias_name or "avg"
            col_fn = _compile_expr(inner.this)
            def avg_fn(rows, _f=col_fn):
                vals = [_f(r) for r in rows if _f(r) is not None]
                return sum(vals) / len(vals) if vals else None
            agg_exprs.append((name, avg_fn))
            project_columns.append(name)

        elif isinstance(inner, exp.Min):
            has_aggregates = True
            name = alias_name or "min"
            col_fn = _compile_expr(inner.this)
            agg_fn = lambda rows, _f=col_fn: min((_f(r) for r in rows if _f(r) is not None), default=None)
            agg_exprs.append((name, agg_fn))
            project_columns.append(name)

        elif isinstance(inner, exp.Max):
            has_aggregates = True
            name = alias_name or "max"
            col_fn = _compile_expr(inner.this)
            agg_fn = lambda rows, _f=col_fn: max((_f(r) for r in rows if _f(r) is not None), default=None)
            agg_exprs.append((name, agg_fn))
            project_columns.append(name)

        elif isinstance(inner, exp.Column):
            col_name = inner.name
            if alias_name:
                project_columns.append(alias_name)
            else:
                project_columns.append(col_name)
            group_columns.append(col_name)

        elif isinstance(inner, exp.Star):
            project_columns.append("*")

        else:
            # Some computed expression
            name = alias_name or str(inner)
            project_columns.append(name)

    return group_columns, agg_exprs, project_columns, has_aggregates


# ============================================================================
# Main parser: SQL string -> propagator network plan -> build_network_manual
# ============================================================================

def parse_query(sql, catalog, use_constraints=False):
    """Parse a SQL SELECT statement and build a propagator network.

    Args:
        sql: SQL SELECT query string
        catalog: Catalog with registered tables
        use_constraints: If True, use bidirectional constraints

    Returns:
        QueryNetwork
    """
    tree = sqlglot.parse_one(sql)

    if not isinstance(tree, exp.Select):
        raise ValueError(f"Expected SELECT statement, got {type(tree).__name__}")

    plan = []

    # --- FROM / JOIN ---
    from_clause = tree.find(exp.From)
    if from_clause is None:
        raise ValueError("Query must have a FROM clause")

    # Primary table
    primary_table = from_clause.this
    primary_name = primary_table.name
    primary_alias = primary_table.alias or primary_name
    plan.append({'op': 'scan', 'table': primary_name, 'alias': primary_alias})

    # Joins
    join_count = 0
    for join_node in tree.find_all(exp.Join):
        join_count += 1
        join_table = join_node.this
        join_table_name = join_table.name
        join_alias = join_table.alias or join_table_name

        plan.append({'op': 'scan', 'table': join_table_name, 'alias': join_alias})

        # ON condition
        on_expr = join_node.args.get("on")
        if on_expr:
            condition_fn = _compile_expr(on_expr.this if isinstance(on_expr, exp.Connector) else on_expr)
        else:
            condition_fn = lambda row: True  # CROSS JOIN

        # Determine left cell name
        if join_count == 1:
            left_cell = f"scan_{primary_alias}"
        else:
            left_cell = f"join_{join_count - 1}"

        plan.append({
            'op': 'join',
            'left': left_cell,
            'right': f"scan_{join_alias}",
            'condition': condition_fn,
            'selectivity': 0.2,
            'name': f"join_{join_count}",
        })

    # --- WHERE ---
    where_clause = tree.find(exp.Where)
    if where_clause is not None:
        predicate_fn = _compile_expr(where_clause.this)
        filter_name = "filter"
        plan.append({
            'op': 'filter',
            'predicate': predicate_fn,
            'selectivity': 0.33,
            'name': filter_name,
        })

    # --- GROUP BY + aggregates ---
    group_node = tree.find(exp.Group)
    select_expressions = tree.expressions

    _, agg_exprs, project_columns, has_aggregates = _extract_aggregates(select_expressions)

    if group_node is not None:
        group_keys = [col.name for col in group_node.expressions]
        plan.append({
            'op': 'aggregate',
            'group_by': group_keys,
            'agg_exprs': agg_exprs,
            'name': 'aggregate',
        })
    elif has_aggregates:
        # Aggregate without GROUP BY (e.g. SELECT COUNT(*) FROM t)
        plan.append({
            'op': 'aggregate',
            'group_by': [],
            'agg_exprs': agg_exprs,
            'name': 'aggregate',
        })

    # --- SELECT projection ---
    if project_columns and "*" not in project_columns:
        plan.append({
            'op': 'project',
            'columns': project_columns,
            'name': 'project',
        })

    # --- ORDER BY ---
    order_node = tree.find(exp.Order)
    if order_node is not None:
        order_by = []
        for ordered in order_node.expressions:
            col = ordered.this
            col_name = col.name if isinstance(col, exp.Column) else str(col)
            desc = ordered.args.get("desc", False)
            order_by.append((col_name, "DESC" if desc else "ASC"))
        plan.append({
            'op': 'sort',
            'order_by': order_by,
            'name': 'sort',
        })

    # --- LIMIT ---
    limit_node = tree.find(exp.Limit)
    if limit_node is not None:
        n = int(limit_node.expression.this)
        plan.append({
            'op': 'limit',
            'n': n,
            'name': 'limit',
        })

    # Build the network
    net = build_network_manual(catalog, plan, use_constraints=use_constraints)
    net.sql = sql
    return net


# ============================================================================
# Multi-statement SQL file loading
# ============================================================================

def parse_sql_file(filepath, catalog=None):
    """Parse a SQL file containing DDL and DML statements.

    Processes CREATE TABLE and INSERT INTO statements to build a Catalog.
    Then processes any SELECT statements and returns networks for them.

    Args:
        filepath: Path to .sql file
        catalog: Optional existing Catalog to extend (creates new if None)

    Returns:
        (catalog, networks) where networks is a list of QueryNetwork
    """
    with open(filepath, 'r') as f:
        sql_text = f.read()

    return parse_sql_statements(sql_text, catalog)


def parse_sql_statements(sql_text, catalog=None):
    """Parse multiple SQL statements from a string.

    Handles CREATE TABLE, INSERT INTO, and SELECT.

    Args:
        sql_text: String containing one or more SQL statements
        catalog: Optional existing Catalog to extend

    Returns:
        (catalog, networks) where networks is a list of QueryNetwork
    """
    from .relation_info import ColumnDef

    if catalog is None:
        if _default_catalog_factory is None:
            raise RuntimeError(
                "No catalog provided and no default catalog factory registered "
                "(propagator.sql.catalog should have registered one on import)"
            )
        catalog = _default_catalog_factory()

    statements = sqlglot.parse(sql_text)
    networks = []

    for stmt in statements:
        if stmt is None:
            continue

        if isinstance(stmt, exp.Create):
            _process_create(stmt, catalog)

        elif isinstance(stmt, exp.Insert):
            _process_insert(stmt, catalog)

        elif isinstance(stmt, exp.Select):
            net = parse_query(stmt.sql(), catalog)
            networks.append(net)

    return catalog, networks


def _process_create(stmt, catalog):
    """Process a CREATE TABLE statement to register schema in catalog."""
    from .relation_info import ColumnDef

    schema = stmt.this  # exp.Schema node
    table_name = schema.this.name if isinstance(schema.this, exp.Table) else str(schema.this)

    columns = []
    for col_expr in schema.expressions:
        if isinstance(col_expr, exp.ColumnDef):
            col_name = col_expr.name
            dtype_node = col_expr.args.get("kind")
            dtype = str(dtype_node) if dtype_node else "any"
            # Clean up dtype
            dtype = dtype.strip().upper()
            is_pk = any(
                isinstance(c, exp.PrimaryKeyColumnConstraint)
                for c in (col_expr.args.get("constraints") or [])
                if hasattr(c, 'kind') or isinstance(c, exp.ColumnConstraint)
            )
            nullable = not any(
                isinstance(c, exp.NotNullColumnConstraint)
                for c in (col_expr.args.get("constraints") or [])
                if hasattr(c, 'kind') or isinstance(c, exp.ColumnConstraint)
            )
            # Check constraint expressions
            for constraint in (col_expr.args.get("constraints") or []):
                if isinstance(constraint, exp.ColumnConstraint):
                    kind = constraint.args.get("kind")
                    if isinstance(kind, exp.PrimaryKeyColumnConstraint):
                        is_pk = True
                    if isinstance(kind, exp.NotNullColumnConstraint):
                        nullable = False

            columns.append(ColumnDef(col_name, dtype=dtype.lower(), nullable=nullable, is_pk=is_pk))

    catalog.register_table(table_name, columns, rows=[])


def _process_insert(stmt, catalog):
    """Process an INSERT INTO statement to add rows to catalog."""
    table_name = stmt.this.name if isinstance(stmt.this, exp.Table) else str(stmt.this)

    entry = catalog._tables.get(table_name)
    if entry is None:
        raise ValueError(f"INSERT into unknown table '{table_name}'. CREATE TABLE first.")

    col_names = [c.name for c in entry['columns']]

    # Check if INSERT specifies columns
    schema = stmt.this
    if isinstance(schema, exp.Schema):
        # INSERT INTO t (col1, col2) VALUES ...
        table_name = schema.this.name
        insert_cols = [col.name for col in schema.expressions if isinstance(col, exp.Column)]
        if insert_cols:
            col_names = insert_cols
        entry = catalog._tables.get(table_name)
        if entry is None:
            raise ValueError(f"INSERT into unknown table '{table_name}'.")

    # Extract VALUES
    values_expr = stmt.expression
    if values_expr is None:
        return

    rows_to_add = []
    if isinstance(values_expr, exp.Values):
        for tuple_node in values_expr.expressions:
            if isinstance(tuple_node, exp.Tuple):
                row = {}
                for i, val_node in enumerate(tuple_node.expressions):
                    if i < len(col_names):
                        row[col_names[i]] = _eval_literal(val_node)
                rows_to_add.append(row)
    elif isinstance(values_expr, exp.Tuple):
        # Single row
        row = {}
        for i, val_node in enumerate(values_expr.expressions):
            if i < len(col_names):
                row[col_names[i]] = _eval_literal(val_node)
        rows_to_add.append(row)

    if rows_to_add:
        existing = entry.get('rows') or []
        existing.extend(rows_to_add)
        entry['rows'] = existing
        entry['row_count'] = len(existing)
        # Update distinct counts
        all_cols = [c.name for c in entry['columns']]
        entry['distinct_counts'] = {
            c: len(set(r.get(c) for r in existing)) for c in all_cols
        }


def _eval_literal(node):
    """Evaluate a literal value node."""
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        return _parse_number(node.this)
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Boolean):
        return node.this
    if isinstance(node, exp.Neg):
        inner = _eval_literal(node.this)
        return -inner if inner is not None else None
    # Fallback
    return str(node)
