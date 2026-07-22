"""Register SQL relation merge operations with the propagator framework.

Merge follows the lattice: higher level subsumes lower.
Same-level merges refine information (e.g., intersect estimates).
Conflicting FullRelations produce the_contradiction.
"""

from ..merge import assign_merge_operation, the_contradiction, assign_equivalent_operation
from .relation_info import (
    SchemaInfo, EstimateInfo, FullRelation,
    is_relation_info,
)


def _schemas_compatible(a_cols, b_cols):
    """Check that two column lists have the same names (order-sensitive)."""
    return [c.name for c in a_cols] == [c.name for c in b_cols]


# --- Same-type merges ---

def _merge_schema_schema(a: SchemaInfo, b: SchemaInfo):
    if _schemas_compatible(a.columns, b.columns):
        return a
    return the_contradiction


def _merge_estimate_estimate(a: EstimateInfo, b: EstimateInfo):
    if not _schemas_compatible(a.columns, b.columns):
        return the_contradiction
    # Tighter estimate: take the minimum row count (more informative)
    row_count = min(a.row_count, b.row_count)
    merged_distinct = dict(a.distinct_counts)
    for k, v in b.distinct_counts.items():
        if k in merged_distinct:
            merged_distinct[k] = min(merged_distinct[k], v)
        else:
            merged_distinct[k] = v
    return EstimateInfo(a.columns, row_count, merged_distinct,
                        selectivity=min(a.selectivity, b.selectivity),
                        ordering=a.ordering or b.ordering)


def _merge_full_full(a: FullRelation, b: FullRelation):
    if a == b:
        return a
    return the_contradiction


# --- Cross-level merges: higher subsumes lower ---

def _merge_higher_lower(higher, lower):
    """Higher lattice level subsumes lower — keep higher."""
    if not _schemas_compatible(higher.columns, lower.columns):
        return the_contradiction
    return higher


def _merge_lower_higher(lower, higher):
    return _merge_higher_lower(higher, lower)


# --- Equivalence ---

def _equiv_relation_info(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, FullRelation):
        return a == b
    if isinstance(a, (SchemaInfo, EstimateInfo)):
        return a.column_names == b.column_names and getattr(a, 'row_count', None) == getattr(b, 'row_count', None)
    return False


# --- Predicates for dispatch ---

def _is_schema(x):
    return isinstance(x, SchemaInfo)

def _is_estimate(x):
    return isinstance(x, EstimateInfo)

def _is_full(x):
    return isinstance(x, FullRelation)


_registered = False

def register_sql_merge():
    """Register all SQL relation merge handlers. Safe to call multiple times."""
    global _registered
    if _registered:
        return
    _registered = True

    # Same-type merges
    assign_merge_operation(_merge_schema_schema, _is_schema, _is_schema)
    assign_merge_operation(_merge_estimate_estimate, _is_estimate, _is_estimate)
    assign_merge_operation(_merge_full_full, _is_full, _is_full)

    # Cross-level merges (all 12 ordered pairs for 4 types)
    for higher_pred, lower_pred in [
        (_is_estimate, _is_schema),
        (_is_full, _is_schema),
        (_is_full, _is_estimate),
    ]:
        assign_merge_operation(_merge_higher_lower, higher_pred, lower_pred)
        assign_merge_operation(_merge_lower_higher, lower_pred, higher_pred)

    # Equivalence
    assign_equivalent_operation(_equiv_relation_info, is_relation_info, is_relation_info)
