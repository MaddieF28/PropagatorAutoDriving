"""SQL query modeling with propagators.

Maps relational algebra operators to propagators, with cells carrying
a lattice of partial information: schema -> estimates -> full relation.
"""

from .relation_info import (
    LatticeLevel,
    ColumnDef,
    SchemaInfo,
    EstimateInfo,
    FullRelation,
    RelationInfo,
    is_relation_info,
    lattice_level,
)

from .sql_merge import register_sql_merge

from .operators import (
    scan,
    filter_prop,
    join_prop,
    aggregate_prop,
    project_prop,
    sort_prop,
    limit_prop,
    union_prop,
)

from .constraints import (
    filter_constraint,
    join_constraint,
)

from .catalog import Catalog

from .network import QueryNetwork

from .parser import parse_query, parse_sql_file, parse_sql_statements

from .stepper import QueryStepper

from .diagnostics import (
    cardinality_surprise,
    join_explosion,
    diagnose_network,
)

from .provenance import explain_row, why_missing

from .repair import (
    diagnose_from_reference, RepairHint, RepairReport,
    print_repair_report, HintLevel,
    compute_table_mapping, repair_cost,
)

from .visualize import print_network

# Register SQL merge operations on import
register_sql_merge()
