"""Lattice-level stepper for controlled query execution.

Allows advancing all cells through the information lattice incrementally,
surfacing intermediate state at each level.
"""

from ..nothing import nothing_p
from .relation_info import LatticeLevel, is_relation_info
from .network import QueryNetwork


class QueryStepper:
    """Step through a query network's lattice levels incrementally."""

    def __init__(self, network: QueryNetwork, catalog=None):
        self.network = network
        self.catalog = catalog
        self._current_level = None

    def current_level(self):
        """Return the minimum lattice level across all non-empty cells."""
        levels = []
        for name, cell in self.network.cells.items():
            if not nothing_p(cell.content) and is_relation_info(cell.content):
                levels.append(cell.content.level)
        return min(levels) if levels else None

    def cell_levels(self):
        """Return dict of cell_name -> current lattice level."""
        result = {}
        for name, cell in self.network.cells.items():
            if not nothing_p(cell.content) and is_relation_info(cell.content):
                result[name] = cell.content.level
            else:
                result[name] = None
        return result

    def snapshot(self):
        """Return a dict of cell_name -> (level, summary) for the current state."""
        snap = {}
        for name, cell in self.network.cells.items():
            val = cell.content
            if nothing_p(val):
                snap[name] = (None, "empty")
            elif is_relation_info(val):
                level = val.level
                if hasattr(val, 'rows') and val.rows is not None:
                    summary = f"{len(val.rows)} rows"
                elif hasattr(val, 'row_count'):
                    summary = f"~{val.row_count:.0f} rows (est)"
                else:
                    cols = ", ".join(c.name for c in val.columns)
                    summary = f"schema: [{cols}]"
                snap[name] = (level, summary)
            else:
                snap[name] = (None, str(val))
        return snap

    def step_to(self, level: LatticeLevel):
        """Advance scan cells to the specified level, then let propagation settle.

        This re-seeds scan cells at the requested level and runs the scheduler
        so downstream propagators pick up the new information.
        """
        from ..scheduler import run as scheduler_run
        from .operators import scan

        if self.catalog is not None:
            for name, cell in self.network.cells.items():
                if name.startswith("scan_"):
                    table_alias = name[5:]  # strip "scan_"
                    info = self.catalog.get_table(table_alias, level=level)
                    if info is None:
                        # Try without alias — might be stored under actual table name
                        for tname in self.catalog.table_names():
                            if tname == table_alias or tname.startswith(table_alias):
                                info = self.catalog.get_table(tname, level=level)
                                break
                    if info is not None:
                        cell.add_content(info)

        # Let propagation settle
        scheduler_run()
        self._current_level = level
        return self.snapshot()

    def print_state(self, header=None):
        """Print current state of all cells."""
        if header:
            print(f"\n{'='*60}")
            print(f"  {header}")
            print(f"{'='*60}")

        snap = self.snapshot()
        for name, (level, summary) in snap.items():
            level_str = level.name if level else "EMPTY"
            print(f"  {name:20s} [{level_str:8s}] {summary}")
