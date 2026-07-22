"""Relation information lattice for SQL propagator cells.

The lattice forms a partial order of increasing information:
  Nothing < SchemaInfo < EstimateInfo < FullRelation

Each level subsumes the previous — an EstimateInfo contains schema
information plus cardinality estimates.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class LatticeLevel(IntEnum):
    SCHEMA = 1
    ESTIMATE = 2
    FULL = 3


@dataclass(frozen=True)
class ColumnDef:
    name: str
    dtype: str = "any"  # e.g. "int", "text", "float", "any"
    nullable: bool = True
    is_pk: bool = False

    def __repr__(self):
        parts = [self.name, self.dtype]
        if self.is_pk:
            parts.append("PK")
        if not self.nullable:
            parts.append("NOT NULL")
        return f"ColumnDef({', '.join(parts)})"


@dataclass
class SchemaInfo:
    """Column definitions only — what the parser/binder resolves."""
    columns: list  # list[ColumnDef]
    ordering: Optional[list] = None  # ORDER BY annotation

    @property
    def level(self):
        return LatticeLevel.SCHEMA

    @property
    def column_names(self):
        return [c.name for c in self.columns]

    def __repr__(self):
        cols = ", ".join(c.name for c in self.columns)
        return f"Schema([{cols}])"


@dataclass
class EstimateInfo:
    """Schema + cardinality estimates — what the optimizer uses."""
    columns: list  # list[ColumnDef]
    row_count: float  # estimated rows
    distinct_counts: dict = field(default_factory=dict)  # col_name -> estimated distinct
    selectivity: float = 1.0
    ordering: Optional[list] = None

    @property
    def level(self):
        return LatticeLevel.ESTIMATE

    @property
    def column_names(self):
        return [c.name for c in self.columns]

    @property
    def schema(self):
        return SchemaInfo(self.columns, self.ordering)

    def __repr__(self):
        cols = ", ".join(c.name for c in self.columns)
        return f"Estimate([{cols}], ~{self.row_count:.0f} rows)"


@dataclass
class FullRelation:
    """Fully materialized relation — all rows."""
    columns: list  # list[ColumnDef]
    rows: list  # list[dict]
    ordering: Optional[list] = None

    @property
    def level(self):
        return LatticeLevel.FULL

    @property
    def column_names(self):
        return [c.name for c in self.columns]

    @property
    def schema(self):
        return SchemaInfo(self.columns, self.ordering)

    @property
    def row_count(self):
        return len(self.rows)

    def __repr__(self):
        cols = ", ".join(c.name for c in self.columns)
        return f"FullRelation([{cols}], {len(self.rows)} rows)"

    def __eq__(self, other):
        if not isinstance(other, FullRelation):
            return False
        return self.column_names == other.column_names and self.rows == other.rows


# Union type for dispatch
RelationInfo = (SchemaInfo, EstimateInfo, FullRelation)


def is_relation_info(x) -> bool:
    return isinstance(x, RelationInfo)


def lattice_level(x) -> Optional[LatticeLevel]:
    if is_relation_info(x):
        return x.level
    return None
