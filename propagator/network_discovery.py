"""
Utilities to discover an existing propagator network starting from root cells.

The discovery is intentionally conservative:
- It finds reachable cells by traversing cell.neighbors and propagator closures.
- It infers primitive structural constraints from function_to_propagator_constructor
  closures when possible.
- It extracts one_of-style finite domains from TMS hypotheticals when available.
"""

from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class DiscoveredConstraint:
    """A structural constraint inferred from a propagator closure."""

    kind: str
    cells: List[Any]


@dataclass
class DiscoveredNetwork:
    """Result of discovering a network from root cells."""

    roots: List[Any]
    cells: List[Any]
    propagators: List[Callable]
    constraints: List[DiscoveredConstraint] = field(default_factory=list)
    domains: Dict[Any, Set[Any]] = field(default_factory=dict)
    fixed_values: Dict[Any, Any] = field(default_factory=dict)
    cell_names: Dict[Any, str] = field(default_factory=dict)


def discover_network(root_cells: Iterable[Any]) -> DiscoveredNetwork:
    """
    Discover reachable cells/propagators from the supplied root cells.

    Args:
        root_cells: Starting cells for network traversal.

    Returns:
        DiscoveredNetwork containing reachable cells, inferred structural
        constraints, finite domains from hypotheticals, and fixed values.
    """
    roots = [c for c in root_cells if c is not None]

    seen_cells: Set[Any] = set()
    seen_props: Set[Callable] = set()
    queue = deque(roots)

    while queue:
        cell = queue.popleft()
        if cell in seen_cells or not _is_cell_like(cell):
            continue
        seen_cells.add(cell)

        for neighbor in getattr(cell, "neighbors", []) or []:
            if callable(neighbor) and neighbor not in seen_props:
                seen_props.add(neighbor)

            for linked in _extract_cells_from_callable(neighbor):
                if linked not in seen_cells:
                    queue.append(linked)

    cells = list(seen_cells)
    cell_set = set(cells)
    propagators = list(seen_props)

    constraints = _infer_constraints(propagators)
    constraints = [
        c for c in constraints if c.cells and all(cell in cell_set for cell in c.cells)
    ]

    domains, names = _domains_from_hypotheticals(cell_set)
    fixed_values = _extract_fixed_values(cells)

    # Ensure all discovered cells have stable names.
    cell_names: Dict[Any, str] = dict(names)
    for idx, cell in enumerate(cells):
        if cell not in cell_names:
            name = getattr(cell, "name", None)
            cell_names[cell] = name if name else f"cell_{idx}"

    return DiscoveredNetwork(
        roots=roots,
        cells=cells,
        propagators=propagators,
        constraints=constraints,
        domains=domains,
        fixed_values=fixed_values,
        cell_names=cell_names,
    )


def _is_cell_like(obj: Any) -> bool:
    return hasattr(obj, "neighbors") and hasattr(obj, "add_content")


def _extract_cells_from_callable(fn: Any) -> Set[Any]:
    if not callable(fn):
        return set()

    cells: Set[Any] = set()
    closure = getattr(fn, "__closure__", None)
    if not closure:
        return cells

    for cell_obj in closure:
        try:
            value = cell_obj.cell_contents
        except ValueError:
            continue
        cells.update(_extract_cells(value))

    return cells


def _extract_cells(value: Any) -> Set[Any]:
    out: Set[Any] = set()

    if _is_cell_like(value):
        out.add(value)
        return out

    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            out.update(_extract_cells(item))
        return out

    if isinstance(value, dict):
        for item in value.values():
            out.update(_extract_cells(item))
        return out

    return out


def _closure_vars(fn: Callable) -> Dict[str, Any]:
    names = getattr(fn, "__code__", None).co_freevars if hasattr(fn, "__code__") else ()
    closure = getattr(fn, "__closure__", None) or ()
    out: Dict[str, Any] = {}
    for name, cell_obj in zip(names, closure):
        try:
            out[name] = cell_obj.cell_contents
        except ValueError:
            continue
    return out


def _operator_name_from_lifted(lifted_fn: Any) -> Optional[str]:
    if not callable(lifted_fn):
        return None

    mapping = _closure_vars(lifted_fn)
    op = mapping.get("f")

    # GenericOperator carries a stable symbolic name ('+', '-', '*', ...)
    if op is not None and hasattr(op, "name"):
        return getattr(op, "name")

    return getattr(op, "__name__", None) if op is not None else None


def _infer_constraints(propagators: Iterable[Callable]) -> List[DiscoveredConstraint]:
    constraints: List[DiscoveredConstraint] = []
    seen_signatures: Set[Tuple[str, Tuple[int, ...]]] = set()

    for fn in propagators:
        vars_map = _closure_vars(fn)
        inputs = vars_map.get("inputs")
        output = vars_map.get("output")
        lifted = vars_map.get("lifted_f")

        if not isinstance(inputs, (list, tuple)) or output is None:
            continue

        input_cells = [c for c in inputs if _is_cell_like(c)]
        if not input_cells or not _is_cell_like(output):
            continue

        op_name = _operator_name_from_lifted(lifted)
        if op_name is None:
            continue

        mapped_kind = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "abs": "abs",
            "square": "square",
            "sqrt": "sqrt",
            "<": "lt",
            ">": "gt",
            "<=": "lte",
            ">=": "gte",
            "=": "eq",
            "not": "not",
            "and": "and",
            "or": "or",
            "switch": "switch",
        }.get(op_name)

        if mapped_kind is None:
            continue

        cells = input_cells + [output]
        signature = (mapped_kind, tuple(sorted(id(c) for c in cells)))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        constraints.append(DiscoveredConstraint(kind=mapped_kind, cells=cells))

    return constraints


def _domains_from_hypotheticals(allowed_cells: Set[Any]) -> Tuple[Dict[Any, Set[Any]], Dict[Any, str]]:
    domains: Dict[Any, Set[Any]] = defaultdict(set)
    names: Dict[Any, str] = {}

    try:
        from .tms import get_all_hypotheticals
    except Exception:
        return {}, {}

    for hyp in get_all_hypotheticals():
        cell = getattr(hyp, "output_cell", None)
        value = getattr(hyp, "value_if_chosen", None)
        if cell not in allowed_cells or value is None:
            continue

        # Skip composite placeholders from one_of recursion.
        if isinstance(value, str) and value.startswith("one of"):
            continue

        domains[cell].add(value)
        if cell not in names:
            name = getattr(cell, "name", None)
            names[cell] = name if name else f"cell_{len(names)}"

    return dict(domains), names


def _extract_fixed_values(cells: Iterable[Any]) -> Dict[Any, Any]:
    from .nothing import nothing_p

    fixed: Dict[Any, Any] = {}
    for cell in cells:
        value = _extract_deterministic_cell_value(cell)
        if not nothing_p(value):
            fixed[cell] = value
    return fixed


def _extract_deterministic_cell_value(cell: Any) -> Any:
    """
    Returns the nothing sentinel (not None) when the cell isn't determined,
    so a cell genuinely fixed to the value None is distinguishable from one
    that simply has no content yet.
    """
    from .nothing import nothing, nothing_p

    content = getattr(cell, "content", nothing)
    if nothing_p(content):
        return nothing

    try:
        from .supported_values import get_support_premises, supported_p
        from .tms import hypothetical_p, tms_p, tms_query

        if supported_p(content):
            premises = get_support_premises(content)
            if any(hypothetical_p(p) for p in premises):
                return nothing
            return content.value

        if tms_p(content):
            queried = tms_query(content)
            if nothing_p(queried):
                return nothing
            if supported_p(queried):
                premises = get_support_premises(queried)
                if any(hypothetical_p(p) for p in premises):
                    return nothing
                return queried.value
            return queried
    except Exception:
        pass

    return content
