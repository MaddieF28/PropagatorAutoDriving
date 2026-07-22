"""Bidirectional relational algebra constraints.

Like product() composes multiplier+divider, these compose forward and
backward propagators to enable constraint propagation in both directions.
"""

from ..cell import Cell, propagator, compound_propagator
from ..nothing import nothing_p
from .relation_info import (
    ColumnDef, SchemaInfo, EstimateInfo, FullRelation,
    is_relation_info,
)
from .operators import filter_prop, join_prop


def filter_constraint(input_cell: Cell, predicate_fn, output: Cell, selectivity=0.33):
    """Bidirectional filter constraint.

    Forward: output = σ_{pred}(input)
    Backward: given output rows, infer that input must contain at least those rows
              (plus any others — the filter only removes, never adds).
    """
    # Forward direction
    filter_prop(input_cell, predicate_fn, output, selectivity)

    # Backward direction: output rows must be a subset of input rows
    def backward():
        out_val = output.content
        if nothing_p(out_val):
            return
        if not is_relation_info(out_val):
            return

        # If output has rows, input must contain at least those rows
        if isinstance(out_val, FullRelation):
            # Input must contain at least these rows (schema propagation backward)
            input_cell.add_content(SchemaInfo(out_val.columns))
        elif isinstance(out_val, EstimateInfo):
            # Input must have at least as many rows as output (selectivity <= 1)
            input_cell.add_content(EstimateInfo(
                out_val.columns,
                out_val.row_count / max(selectivity, 0.01),
            ))
        elif isinstance(out_val, SchemaInfo):
            input_cell.add_content(out_val)

    propagator([output], backward)


def join_constraint(left: Cell, right: Cell, condition_fn, output: Cell, selectivity=0.1):
    """Bidirectional join constraint.

    Forward: output = left ⋈ right
    Backward: given output rows, infer which left and right rows contributed.
    """
    # Forward
    join_prop(left, right, condition_fn, output, selectivity)

    # Backward: project output back to left/right column sets
    def backward_left():
        out_val = output.content
        lv = left.content
        if nothing_p(out_val):
            return
        if not is_relation_info(out_val):
            return

        if is_relation_info(lv):
            left_col_names = [c.name for c in lv.columns]
        else:
            return

        if isinstance(out_val, FullRelation):
            left_rows = []
            seen = set()
            for row in out_val.rows:
                projected = {k: v for k, v in row.items() if k in left_col_names}
                key = tuple(sorted(projected.items()))
                if key not in seen:
                    seen.add(key)
                    left_rows.append(projected)
            left_cols = [c for c in out_val.columns if c.name in left_col_names]
            left.add_content(FullRelation(left_cols or lv.columns, left_rows))
        elif isinstance(out_val, SchemaInfo):
            left.add_content(SchemaInfo([c for c in out_val.columns if c.name in left_col_names] or lv.columns))

    def backward_right():
        out_val = output.content
        rv = right.content
        if nothing_p(out_val):
            return
        if not is_relation_info(out_val):
            return

        if is_relation_info(rv):
            right_col_names = [c.name for c in rv.columns]
        else:
            return

        if isinstance(out_val, FullRelation):
            right_rows = []
            seen = set()
            for row in out_val.rows:
                projected = {k: v for k, v in row.items() if k in right_col_names}
                key = tuple(sorted(projected.items()))
                if key not in seen:
                    seen.add(key)
                    right_rows.append(projected)
            right_cols = [c for c in out_val.columns if c.name in right_col_names]
            right.add_content(FullRelation(right_cols or rv.columns, right_rows))
        elif isinstance(out_val, SchemaInfo):
            right.add_content(SchemaInfo([c for c in out_val.columns if c.name in right_col_names] or rv.columns))

    propagator([output, left], backward_left)
    propagator([output, right], backward_right)
