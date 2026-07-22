"""Table catalog for SQL propagator networks.

Stores schema, statistics, and optionally full data for tables.
Provides table info at various lattice levels.
"""

from .relation_info import (
    ColumnDef, SchemaInfo, EstimateInfo, FullRelation,
    LatticeLevel,
)
from .parser import parse_sql_statements, parse_sql_file, register_default_catalog_factory


class Catalog:
    """Registry of table schemas, statistics, and data."""

    def __init__(self):
        self._tables = {}  # name -> dict with 'columns', 'rows', 'stats'

    def register_table(self, name, columns, rows=None, row_count=None, distinct_counts=None):
        """Register a table.

        Args:
            name: Table name
            columns: list of ColumnDef or list of str (auto-converted)
            rows: optional list[dict] of all rows
            row_count: estimated row count (defaults to len(rows) if rows given)
            distinct_counts: dict col_name -> distinct count estimate
        """
        if columns and isinstance(columns[0], str):
            columns = [ColumnDef(c) for c in columns]
        entry = {
            'columns': columns,
            'rows': rows,
            'row_count': row_count if row_count is not None else (len(rows) if rows else 0),
            'distinct_counts': distinct_counts or {},
        }
        self._tables[name] = entry

    def get_table(self, name, level=LatticeLevel.FULL):
        """Get table info at the requested lattice level.

        Returns the highest available level up to `level`.
        """
        entry = self._tables.get(name)
        if entry is None:
            return None

        cols = entry['columns']
        rows = entry.get('rows')
        rc = entry.get('row_count', 0)
        dc = entry.get('distinct_counts', {})

        if level >= LatticeLevel.FULL and rows is not None:
            return FullRelation(cols, rows)
        if level >= LatticeLevel.ESTIMATE and rc > 0:
            return EstimateInfo(cols, rc, dc)
        return SchemaInfo(cols)

    def table_names(self):
        return list(self._tables.keys())

    @classmethod
    def from_dicts(cls, tables):
        """Create catalog from dict of {table_name: list[dict]}.

        Column defs are inferred from the first row's keys.
        """
        cat = cls()
        for name, rows in tables.items():
            if rows:
                col_names = list(rows[0].keys())
                columns = [ColumnDef(c) for c in col_names]
                distinct = {c: len(set(r.get(c) for r in rows)) for c in col_names}
                cat.register_table(name, columns, rows=rows, distinct_counts=distinct)
            else:
                cat.register_table(name, [], rows=[])
        return cat

    @classmethod
    def from_sql(cls, sql_text, catalog=None):
        """Create/extend a catalog from SQL DDL+DML statements.

        Processes CREATE TABLE and INSERT INTO statements.

        Args:
            sql_text: String containing SQL statements
            catalog: Optional existing catalog to extend

        Returns:
            (catalog, networks) — catalog with tables, list of QueryNetwork
                                  for any SELECT statements found
        """
        return parse_sql_statements(sql_text, catalog)

    @classmethod
    def from_sql_file(cls, filepath, catalog=None):
        """Create/extend a catalog from a .sql file.

        The file can contain CREATE TABLE, INSERT INTO, and SELECT statements.
        DDL/DML populate the catalog; SELECTs are converted to propagator networks.

        Args:
            filepath: Path to .sql file
            catalog: Optional existing catalog to extend

        Returns:
            (catalog, networks)
        """
        return parse_sql_file(filepath, catalog)


# Register with parser.py so it can construct a default Catalog when none is
# passed, without ever importing this module -- see register_default_catalog_factory
# in parser.py for the full rationale (mirrors register_tms_initializer/
# register_cdcl_handlers for the same kind of cycle).
register_default_catalog_factory(Catalog)
