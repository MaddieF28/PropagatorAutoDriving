"""Unidirectional relational algebra propagators.

Each propagator operates at whatever lattice level its input cells are at.
Convention: output cell is LAST argument (follows propagator framework).
"""

from collections import defaultdict

from ..cell import Cell, propagator, compound_propagator, function_to_propagator_constructor
from ..primitives import constant
from ..nothing import nothing_p
from .relation_info import (
    ColumnDef, SchemaInfo, EstimateInfo, FullRelation,
    LatticeLevel, is_relation_info,
)


# ============================================================================
# Scan: seed a cell from catalog data
# ============================================================================

def scan(table_name, catalog, output: Cell):
    """Scan propagator — reads table from catalog into output cell.

    The catalog provides schema and optionally stats + rows.
    Seeds at the highest available level.
    """
    info = catalog.get_table(table_name)
    if info is not None:
        output.add_content(info)


# ============================================================================
# Filter (WHERE): narrow rows by predicate
# ============================================================================

def _apply_filter(input_info, predicate_fn, selectivity=0.33):
    """Apply filter at whatever lattice level the input is at."""
    if not is_relation_info(input_info):
        return None

    if isinstance(input_info, FullRelation):
        filtered = [r for r in input_info.rows if predicate_fn(r)]
        return FullRelation(input_info.columns, filtered, input_info.ordering)

    if isinstance(input_info, EstimateInfo):
        new_count = input_info.row_count * selectivity
        return EstimateInfo(input_info.columns, new_count,
                            {k: v * selectivity for k, v in input_info.distinct_counts.items()},
                            selectivity=input_info.selectivity * selectivity,
                            ordering=input_info.ordering)

    if isinstance(input_info, SchemaInfo):
        return input_info  # filter doesn't change schema

    return None


def filter_prop(input_cell: Cell, predicate_fn, output: Cell, selectivity=0.33):
    """Filter propagator: output = σ_{predicate}(input).

    Args:
        input_cell: Input relation cell
        predicate_fn: callable(row_dict) -> bool
        output: Output relation cell
        selectivity: estimated fraction of rows passing (for estimate level)
    """
    def to_do():
        val = input_cell.content
        if nothing_p(val):
            return
        result = _apply_filter(val, predicate_fn, selectivity)
        if result is not None:
            output.add_content(result)

    propagator([input_cell], to_do)


# ============================================================================
# Join: combine two relations
# ============================================================================

def _apply_join(left_info, right_info, condition_fn, join_selectivity=0.1):
    """Apply join at whatever lattice level both inputs are at."""
    if not is_relation_info(left_info) or not is_relation_info(right_info):
        return None

    # Combined schema (prefix-qualified if needed)
    combined_cols = list(left_info.columns) + list(right_info.columns)

    # Use the minimum level of the two inputs
    left_level = left_info.level
    right_level = right_info.level
    target_level = min(left_level, right_level)

    if target_level >= LatticeLevel.FULL and isinstance(left_info, FullRelation) and isinstance(right_info, FullRelation):
        result_rows = []
        for lr in left_info.rows:
            for rr in right_info.rows:
                combined = {**lr, **rr}
                if condition_fn(combined):
                    result_rows.append(combined)
        return FullRelation(combined_cols, result_rows)

    if target_level >= LatticeLevel.ESTIMATE:
        left_count = getattr(left_info, 'row_count', 0)
        right_count = getattr(right_info, 'row_count', 0)
        est_count = left_count * right_count * join_selectivity
        return EstimateInfo(combined_cols, est_count)

    # Schema level
    return SchemaInfo(combined_cols)


def join_prop(left: Cell, right: Cell, condition_fn, output: Cell, selectivity=0.1):
    """Join propagator: output = left ⋈_{condition} right.

    Args:
        left, right: Input relation cells
        condition_fn: callable(combined_row_dict) -> bool
        output: Output relation cell
        selectivity: estimated join selectivity for estimates
    """
    def to_do():
        lv = left.content
        rv = right.content
        if nothing_p(lv) or nothing_p(rv):
            return
        result = _apply_join(lv, rv, condition_fn, selectivity)
        if result is not None:
            output.add_content(result)

    propagator([left, right], to_do)


# ============================================================================
# Aggregate (GROUP BY)
# ============================================================================

def _apply_aggregate(input_info, group_keys, agg_exprs):
    """Apply aggregation.

    agg_exprs: list of (output_name, agg_fn) where agg_fn(group_rows) -> value
    """
    if not is_relation_info(input_info):
        return None

    out_cols = [ColumnDef(k) for k in group_keys] + [ColumnDef(name) for name, _ in agg_exprs]

    if isinstance(input_info, (FullRelation,)):
        rows = input_info.rows
        groups = defaultdict(list)
        for row in rows:
            key = tuple(row.get(k) for k in group_keys)
            groups[key].append(row)

        result_rows = []
        for key, group_rows in groups.items():
            out_row = dict(zip(group_keys, key))
            for name, agg_fn in agg_exprs:
                out_row[name] = agg_fn(group_rows)
            result_rows.append(out_row)

        return FullRelation(out_cols, result_rows)

    if isinstance(input_info, EstimateInfo):
        # Estimate distinct groups
        est_groups = 1
        for k in group_keys:
            est_groups = max(est_groups, input_info.distinct_counts.get(k, input_info.row_count * 0.1))
        return EstimateInfo(out_cols, est_groups)

    if isinstance(input_info, SchemaInfo):
        return SchemaInfo(out_cols)

    return None


def aggregate_prop(input_cell: Cell, group_keys, agg_exprs, output: Cell):
    """Aggregate propagator: output = γ_{group_keys, agg_exprs}(input).

    Args:
        input_cell: Input relation cell
        group_keys: list of column names to group by
        agg_exprs: list of (output_col_name, agg_function(group_rows) -> value)
        output: Output relation cell
    """
    def to_do():
        val = input_cell.content
        if nothing_p(val):
            return
        result = _apply_aggregate(val, group_keys, agg_exprs)
        if result is not None:
            output.add_content(result)

    propagator([input_cell], to_do)


# ============================================================================
# Project (SELECT columns)
# ============================================================================

def _apply_project(input_info, columns):
    if not is_relation_info(input_info):
        return None

    # Find matching column defs
    col_map = {c.name: c for c in input_info.columns}
    out_cols = [col_map.get(c, ColumnDef(c)) for c in columns]

    if isinstance(input_info, FullRelation):
        projected = [{c: row.get(c) for c in columns} for row in input_info.rows]
        return FullRelation(out_cols, projected, input_info.ordering)

    if isinstance(input_info, EstimateInfo):
        dc = {k: v for k, v in input_info.distinct_counts.items() if k in columns}
        return EstimateInfo(out_cols, input_info.row_count, dc, input_info.selectivity,
                            ordering=input_info.ordering)

    if isinstance(input_info, SchemaInfo):
        return SchemaInfo(out_cols, input_info.ordering)

    return None


def project_prop(input_cell: Cell, columns, output: Cell):
    """Project propagator: output = π_{columns}(input)."""
    def to_do():
        val = input_cell.content
        if nothing_p(val):
            return
        result = _apply_project(val, columns)
        if result is not None:
            output.add_content(result)

    propagator([input_cell], to_do)


# ============================================================================
# Sort (ORDER BY)
# ============================================================================

def sort_prop(input_cell: Cell, order_keys, output: Cell):
    """Sort propagator: output = τ_{order_keys}(input).

    Preserves cardinality, adds ordering annotation.
    order_keys: list of (col_name, 'asc'|'desc')
    """
    def to_do():
        val = input_cell.content
        if nothing_p(val):
            return

        if isinstance(val, FullRelation):
            sorted_rows = list(val.rows)
            for col, direction in reversed(order_keys):
                reverse = direction.lower() == 'desc'
                sorted_rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=reverse)
            output.add_content(FullRelation(val.columns, sorted_rows, ordering=order_keys))
        elif isinstance(val, EstimateInfo):
            output.add_content(EstimateInfo(val.columns, val.row_count,
                                            val.distinct_counts, val.selectivity,
                                            ordering=order_keys))
        elif isinstance(val, SchemaInfo):
            output.add_content(SchemaInfo(val.columns, ordering=order_keys))

    propagator([input_cell], to_do)


# ============================================================================
# Limit
# ============================================================================

def limit_prop(input_cell: Cell, n: int, output: Cell):
    """Limit propagator: output = first n rows of input."""
    def to_do():
        val = input_cell.content
        if nothing_p(val):
            return

        if isinstance(val, FullRelation):
            output.add_content(FullRelation(val.columns, val.rows[:n], val.ordering))
        elif isinstance(val, EstimateInfo):
            est = min(n, val.row_count)
            output.add_content(EstimateInfo(val.columns, est, val.distinct_counts,
                                            ordering=val.ordering))
        elif isinstance(val, SchemaInfo):
            output.add_content(val)

    propagator([input_cell], to_do)


# ============================================================================
# Union
# ============================================================================

def union_prop(left: Cell, right: Cell, output: Cell, all_rows=True):
    """Union propagator: output = left ∪ right.

    Args:
        all_rows: If True, UNION ALL; if False, UNION (deduplicate).
    """
    def to_do():
        lv = left.content
        rv = right.content
        if nothing_p(lv) or nothing_p(rv):
            return

        if not is_relation_info(lv) or not is_relation_info(rv):
            return

        # Use left's columns (assume compatible schema)
        cols = lv.columns

        if isinstance(lv, FullRelation) and isinstance(rv, FullRelation):
            combined = lv.rows + rv.rows
            if not all_rows:
                seen = set()
                deduped = []
                for r in combined:
                    key = tuple(sorted(r.items()))
                    if key not in seen:
                        seen.add(key)
                        deduped.append(r)
                combined = deduped
            output.add_content(FullRelation(cols, combined))
        elif hasattr(lv, 'row_count') and hasattr(rv, 'row_count'):
            est = lv.row_count + rv.row_count
            if not all_rows:
                est *= 0.8  # rough dedup estimate
            output.add_content(EstimateInfo(cols, est))
        else:
            output.add_content(SchemaInfo(cols))

    propagator([left, right], to_do)
