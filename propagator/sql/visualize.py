"""Text-based visualization for SQL propagator networks."""

from ..nothing import nothing_p
from .relation_info import is_relation_info, FullRelation, EstimateInfo, SchemaInfo
from .network import QueryNetwork


def print_network(network: QueryNetwork, show_rows=False, max_rows=5):
    """Print the DAG of cells with current lattice level and key stats."""
    print(f"\n{'='*70}")
    print(f"  SQL Propagator Network" + (f": {network.sql}" if network.sql else ""))
    print(f"  {len(network.cells)} cells, {len(network.operators)} operators")
    print(f"{'='*70}")

    for name, cell in network.cells.items():
        val = cell.content

        # Marker for output cell
        marker = " ◀ OUTPUT" if cell is network.output_cell else ""

        if nothing_p(val):
            print(f"\n  ┌─ {name}{marker}")
            print(f"  │  [EMPTY]")
            print(f"  └─")
            continue

        if not is_relation_info(val):
            print(f"\n  ┌─ {name}{marker}")
            print(f"  │  [NON-SQL: {type(val).__name__}] {val}")
            print(f"  └─")
            continue

        level = val.level.name
        cols = ", ".join(c.name for c in val.columns)

        print(f"\n  ┌─ {name}{marker}")
        print(f"  │  Level: {level}")
        print(f"  │  Columns: [{cols}]")

        if hasattr(val, 'row_count'):
            if isinstance(val, FullRelation):
                print(f"  │  Rows: {val.row_count} (exact)")
            else:
                print(f"  │  Rows: ~{val.row_count:.0f} (estimated)")

        if hasattr(val, 'distinct_counts') and val.distinct_counts:
            dc = ", ".join(f"{k}={v}" for k, v in val.distinct_counts.items())
            print(f"  │  Distinct: {dc}")

        if hasattr(val, 'ordering') and val.ordering:
            ord_str = ", ".join(f"{c} {d}" for c, d in val.ordering)
            print(f"  │  Ordering: {ord_str}")

        if show_rows and hasattr(val, 'rows') and val.rows:
            n = min(max_rows, len(val.rows))
            print(f"  │  Data ({n} of {len(val.rows)}):")
            for i, row in enumerate(val.rows[:n]):
                print(f"  │    {row}")
            if len(val.rows) > n:
                print(f"  │    ... ({len(val.rows) - n} more)")

        print(f"  └─")

    # Print operator connections
    if network.operators:
        print(f"\n  Operator chain:")
        for op_type, op_info in network.operators:
            parts = []
            for key in ('left', 'right', 'input'):
                c = op_info.get(key)
                if c:
                    for n, cc in network.cells.items():
                        if cc is c:
                            parts.append(f"{key}={n}")
                            break
            out_c = op_info.get('output')
            if out_c:
                for n, cc in network.cells.items():
                    if cc is out_c:
                        parts.append(f"→ {n}")
                        break
            print(f"    {op_type:12s} {', '.join(parts)}")

    print()


def print_diagnostics(issues, header="Diagnostics"):
    """Pretty-print diagnostic results."""
    if not issues:
        print(f"\n  {header}: No issues detected ✓")
        return

    print(f"\n  {header}: {len(issues)} issue(s) found")
    print(f"  {'─'*50}")
    for name, diag_type, detail in issues:
        icon = "⚠" if "surprise" in diag_type else "🔥" if "explosion" in diag_type else "•"
        print(f"  {icon} [{name}] {diag_type}: {detail}")


def print_provenance(chain, row=None):
    """Pretty-print a provenance chain."""
    if row:
        print(f"\n  Provenance for row: {row}")
    else:
        print(f"\n  Provenance chain:")
    print(f"  {'─'*50}")
    for i, (cell_name, matching_rows) in enumerate(chain):
        indent = "  " + "  " * i
        print(f"{indent}↳ {cell_name}: {len(matching_rows)} matching row(s)")
        for mr in matching_rows[:3]:
            print(f"{indent}    {mr}")
        if len(matching_rows) > 3:
            print(f"{indent}    ... ({len(matching_rows) - 3} more)")
