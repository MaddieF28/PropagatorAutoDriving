"""
Automatic extraction of constraints from existing propagator networks.

This module provides tools to automatically compile propagator networks to
SAT/SMT without modifying the original code. The preferred workflow is
roots-first compilation via `compile_from_roots`/`solve_from_roots`.

This module remains useful for compatibility and specialized introspection
workflows. It works by:

1. Introspecting the TMS hypotheticals to discover cells and domains
2. Extracting learned nogoods from the TMS
3. Optionally recording selected guessing-machine calls (legacy path)

Usage (zero-modification approach):
    from propagator import initialize_scheduler
    from propagator.solver_export.auto_extract import extract_from_hypotheticals
    
    # Run your existing propagator setup
    initialize_scheduler()
    setup_my_problem()  # Your existing code, unmodified
    
    # Extract and compile
    compiler = extract_from_hypotheticals()
    result = compiler.solve(SolverBackend.Z3_PYTHON)

Preferred API for new code:
    from propagator.solver_export import solve_from_roots, SolverBackend, TranslationMode

    result, report = solve_from_roots(
        root_cells,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict
import weakref

if TYPE_CHECKING:
    from ..cell import Cell

from .compiler import NetworkCompiler


# =============================================================================
# Hypothetical Introspection
# =============================================================================

def extract_from_hypotheticals(
    name: str = "extracted_network",
    add_all_distinct: bool = False,
) -> NetworkCompiler:
    """
    Extract constraints from an already-built propagator network by
    introspecting the TMS hypotheticals.
    
    This is the ZERO-MODIFICATION approach: run your existing propagator
    setup code, then call this to extract the constraint structure.
    
    Works by:
    1. Finding all Hypothetical objects in the TMS
    2. Grouping them by output_cell to reconstruct domains
    3. Extracting nogoods as learned clauses
    
    Args:
        name: Name for the compiled network
        add_all_distinct: If True, adds an all_distinct constraint on all
                         extracted cells. Useful for puzzles where all 
                         variables must have different values (N-Queens row
                         constraint, Multiple Dwelling floors, etc.)
    
    Returns:
        NetworkCompiler ready for export/solve
        
    Example:
        from propagator import initialize_scheduler, Cell
        from propagator.guessing_machine import one_of, require_distinct
        from propagator.solver_export import extract_from_hypotheticals, SolverBackend
        
        # Your existing code, UNCHANGED
        initialize_scheduler()
        x = Cell(name="x")
        y = Cell(name="y")
        one_of([1, 2, 3], x)
        one_of([1, 2, 3], y)
        require_distinct([x, y])
        
        # Extract and solve with external solver
        # Use add_all_distinct=True since require_distinct was called
        compiler = extract_from_hypotheticals(add_all_distinct=True)
        result = compiler.solve(SolverBackend.Z3_PYTHON)
        print(result.solution)  # {x: 1, y: 2}
    """
    from ..tms import (
        hypothetical_p,
        get_all_hypotheticals,
        get_all_nogoods,
    )

    compiler = NetworkCompiler(name)

    allowed_cells = None

    cell_values, cell_names = _collect_domains_from_hypotheticals(
        get_all_hypotheticals(),
        allowed_cells=allowed_cells,
    )

    all_cells = _register_domains(compiler, cell_values, cell_names)

    if add_all_distinct and len(all_cells) >= 2:
        compiler.add_all_distinct(all_cells)

    # Extract nogoods
    for nogood in get_all_nogoods():
        assignments = []
        for premise in nogood:
            if hypothetical_p(premise):
                if hasattr(premise, 'output_cell') and hasattr(premise, 'value_if_chosen'):
                    cell = premise.output_cell
                    value = premise.value_if_chosen
                    if cell is not None and value is not None:
                        if allowed_cells is not None and cell not in allowed_cells:
                            continue
                        if not (isinstance(value, str) and value.startswith("one of")):
                            assignments.append((cell, value))

        if len(assignments) >= 2:
            compiler.add_nogood(assignments)

    return compiler


def _collect_domains_from_hypotheticals(
    hypotheticals,
    allowed_cells: Optional[Set[Any]] = None,
) -> tuple:
    """
    Iterate over TMS hypotheticals and group leaf values by output_cell.

    Returns:
        (cell_values, cell_names) where cell_values is a defaultdict(set)
        and cell_names maps each cell to its resolved name string.
    """
    cell_values: Dict[Any, Set[Any]] = defaultdict(set)
    cell_names: Dict[Any, str] = {}

    for hyp in hypotheticals:
        if hasattr(hyp, 'output_cell') and hasattr(hyp, 'value_if_chosen'):
            cell = hyp.output_cell
            value = hyp.value_if_chosen
            if cell is not None and value is not None:
                if allowed_cells is not None and cell not in allowed_cells:
                    continue
                # Skip composite "one of [...]" placeholders
                if not (isinstance(value, str) and value.startswith("one of")):
                    cell_values[cell].add(value)
                    if cell not in cell_names:
                        if hasattr(cell, 'name') and cell.name:
                            cell_names[cell] = cell.name
                        else:
                            cell_names[cell] = f"cell_{len(cell_names)}"

    return cell_values, cell_names


def _register_domains(
    compiler: NetworkCompiler,
    cell_values: Dict,
    cell_names: Dict,
) -> list:
    """Register extracted domains into a NetworkCompiler. Returns the list of cells."""
    all_cells = []
    for cell, values in cell_values.items():
        try:
            domain = sorted(list(values))
        except TypeError:
            domain = list(values)
        compiler.add_domain(cell, domain, name=cell_names.get(cell))
        all_cells.append(cell)
    return all_cells


def extract_domains_only(
    name: str = "domains_only",
    root_cells: Optional[List['Cell']] = None,
) -> NetworkCompiler:
    """
    Extract only the domain structure (cells and their possible values).

    This is useful when you want to add your own constraints programmatically
    but use the domains discovered from the propagator network.
    
    Returns:
        NetworkCompiler with domains registered but no constraints
    """
    from ..tms import get_all_hypotheticals

    compiler = NetworkCompiler(name)
    from ..network_discovery import discover_network

    discovered = discover_network(root_cells or []) if root_cells else None
    allowed_cells = set(discovered.cells) if discovered else None

    cell_values, cell_names = _collect_domains_from_hypotheticals(
        get_all_hypotheticals(),
        allowed_cells=allowed_cells,
    )

    if discovered is not None:
        for cell, values in discovered.domains.items():
            if cell not in cell_values:
                cell_values[cell] = set(values)
            else:
                cell_values[cell].update(values)
            if cell not in cell_names:
                cell_names[cell] = discovered.cell_names.get(cell, getattr(cell, 'name', None))

    _register_domains(compiler, cell_values, cell_names)
    return compiler

