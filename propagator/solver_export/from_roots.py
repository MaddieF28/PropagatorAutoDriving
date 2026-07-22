"""Unified automatic compile/solve APIs from root cells.

This module provides a single backend-agnostic entry point for discovering an
existing propagator network and compiling it for an external solver backend.

Policy:
- Strict mode blocks on unsupported expressions.
- Hybrid oracle mode compiles the supported subset and reports what was skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..network_discovery import DiscoveredConstraint, discover_network
from .compiler import NetworkCompiler, SolverBackend, SolverResult


class TranslationMode(Enum):
    """Translation behavior when unsupported expressions are discovered."""

    STRICT = "strict"
    HYBRID_ORACLE = "hybrid_oracle"


@dataclass
class TranslationIssue:
    """Single translation issue surfaced during compile-from-roots."""

    kind: str
    cell_names: List[str]
    reason: str


@dataclass
class RootCompileReport:
    """Detailed report of root-based compilation."""

    compiler: NetworkCompiler
    backend: SolverBackend
    mode: TranslationMode
    discovered_cell_count: int = 0
    translated_constraint_count: int = 0
    skipped_constraint_count: int = 0
    issues: List[TranslationIssue] = field(default_factory=list)


class UnsupportedTranslationError(ValueError):
    """Raised when strict translation detects unsupported expressions."""


def compile_from_roots(
    root_cells: List[Any],
    *,
    backend: SolverBackend = SolverBackend.Z3_PYTHON,
    name: str = "compiled_from_roots",
    mode: TranslationMode = TranslationMode.STRICT,
    include_nogoods: bool = True,
) -> RootCompileReport:
    """
    Discover and compile an existing propagator network from root cells.

    Args:
        root_cells: Cells used as discovery roots.
        backend: Target backend.
        name: Compiler name.
        mode: STRICT blocks on unsupported constraints, HYBRID_ORACLE allows
            partial translation and reports skipped expressions.
        include_nogoods: Include learned TMS nogoods when available.

    Returns:
        RootCompileReport with compiler plus translation diagnostics.
    """
    discovered = discover_network(root_cells)
    compiler = NetworkCompiler(name)

    report = RootCompileReport(
        compiler=compiler,
        backend=backend,
        mode=mode,
        discovered_cell_count=len(discovered.cells),
    )

    sat_like = backend == SolverBackend.DIMACS_CNF
    sat_domain_ready = set()

    # Register variables first.
    for cell in discovered.cells:
        domain_values = discovered.domains.get(cell)
        fixed_value = discovered.fixed_values.get(cell)
        cell_name = discovered.cell_names.get(cell, getattr(cell, "name", None))

        if domain_values:
            domain_list = _sorted_values(domain_values)
            if not _domain_supported_by_backend(domain_list, backend):
                report.issues.append(
                    TranslationIssue(
                        kind="unsupported_domain_values",
                        cell_names=[_cell_name(cell, discovered.cell_names)],
                        reason=f"Domain values are not directly representable for backend {backend.name}.",
                    )
                )
                continue
            compiler.add_domain(cell, domain_list, name=cell_name)
            sat_domain_ready.add(cell)
            continue

        if fixed_value is not None:
            if not _value_supported_by_backend(fixed_value, backend):
                report.issues.append(
                    TranslationIssue(
                        kind="unsupported_fixed_value",
                        cell_names=[_cell_name(cell, discovered.cell_names)],
                        reason=f"Fixed value is not directly representable for backend {backend.name}.",
                    )
                )
                continue
            compiler.add_domain(cell, [fixed_value], name=cell_name)
            compiler.add_fixed_value(cell, fixed_value)
            sat_domain_ready.add(cell)
            continue

        if sat_like:
            report.issues.append(
                TranslationIssue(
                    kind="missing_domain",
                    cell_names=[_cell_name(cell, discovered.cell_names)],
                    reason="SAT backend requires finite domains for discovered cells.",
                )
            )
            continue

        compiler.add_integer(cell, name=cell_name)

    # Add fixed-value constraints for non-singleton domains.
    for cell, value in discovered.fixed_values.items():
        if cell in compiler.variables and (cell not in discovered.domains):
            continue
        if cell in compiler.variables:
            compiler.add_fixed_value(cell, value)

    # Structural constraints.
    for constraint in discovered.constraints:
        if sat_like:
            missing = [c for c in constraint.cells if c not in sat_domain_ready]
            if missing:
                report.skipped_constraint_count += 1
                report.issues.append(
                    TranslationIssue(
                        kind=constraint.kind,
                        cell_names=[_cell_name(c, discovered.cell_names) for c in missing],
                        reason="Constraint skipped for SAT backend because at least one cell lacks finite domain.",
                    )
                )
                continue

        if _try_add_constraint(compiler, constraint, discovered.fixed_values):
            report.translated_constraint_count += 1
            continue

        report.skipped_constraint_count += 1
        report.issues.append(
            TranslationIssue(
                kind=constraint.kind,
                cell_names=[_cell_name(c, discovered.cell_names) for c in constraint.cells],
                reason="Unsupported or ill-formed expression for current compiler mapping.",
            )
        )

    if include_nogoods:
        _add_tms_nogoods(compiler, discovered.cells)

    if mode == TranslationMode.STRICT and report.issues:
        raise UnsupportedTranslationError(_format_issues(report.issues))

    return report


def solve_from_roots(
    root_cells: List[Any],
    *,
    backend: SolverBackend = SolverBackend.Z3_PYTHON,
    mode: TranslationMode = TranslationMode.STRICT,
    name: str = "solve_from_roots",
    solver_path: Optional[str] = None,
    timeout: Optional[float] = None,
    include_nogoods: bool = True,
) -> Tuple[SolverResult, RootCompileReport]:
    """
    Compile and solve from root cells using backend-specific translation policy.

    Returns:
        (solver_result, compile_report)
    """
    report = compile_from_roots(
        root_cells,
        backend=backend,
        name=name,
        mode=mode,
        include_nogoods=include_nogoods,
    )
    result = report.compiler.solve(backend=backend, solver_path=solver_path, timeout=timeout)
    return result, report


def _try_add_constraint(
    compiler: NetworkCompiler,
    constraint: DiscoveredConstraint,
    fixed_values: Dict[Any, Any],
) -> bool:
    kind = constraint.kind
    cells = constraint.cells

    if kind == "add" and len(cells) == 3:
        compiler.add_sum_equals(cells, total=0, coefficients=[1, 1, -1])
        return True
    if kind == "sub" and len(cells) == 3:
        compiler.add_sum_equals(cells, total=0, coefficients=[1, -1, -1])
        return True
    if kind == "mul" and len(cells) == 3:
        compiler.add_product(cells[0], cells[1], cells[2])
        return True
    if kind == "abs" and len(cells) == 2:
        compiler.add_absolute_value(cells[0], cells[1])
        return True
    if kind == "square" and len(cells) == 2:
        # out = a * a
        compiler.add_product(cells[0], cells[0], cells[1])
        return True
    if kind == "sqrt" and len(cells) == 2:
        # sqrter(a, out) means out = sqrt(a), i.e. a = out * out.
        compiler.add_product(cells[1], cells[1], cells[0])
        return True
    if kind == "div" and len(cells) == 3:
        # a / b = c  <=>  a = c * b (assumes b != 0)
        compiler.add_product(cells[2], cells[1], cells[0])
        return True
    if kind == "eq" and len(cells) == 2:
        compiler.add_equality(cells[0], cells[1])
        return True
    if kind == "lt" and len(cells) == 2:
        compiler.add_less_than(cells[0], cells[1])
        return True
    if kind == "gt" and len(cells) == 2:
        compiler.add_greater_than(cells[0], cells[1])
        return True
    if kind == "lte" and len(cells) == 2:
        compiler.add_less_equal(cells[0], cells[1])
        return True
    if kind == "gte" and len(cells) == 2:
        compiler.add_greater_equal(cells[0], cells[1])
        return True

    # Boolean logic propagators discovered as functional constraints:
    #   not(a, output)           => output = NOT a
    #   and(a, b, ..., output)   => output = AND(a, b, ...)
    #   or(a, b, ..., output)    => output = OR(a, b, ...)
    # These are only encodable when cells are boolean (the backends declare
    # them as Bool, not Int, so inequality genuinely means complement).
    if kind == "not" and len(cells) == 2:
        # output = NOT a  <=>  a != output  (for boolean values)
        compiler.add_inequality(cells[0], cells[1])
        return True

    if kind == "and" and len(cells) >= 3:
        # output = AND(inputs...)
        inputs = list(cells[:-1])
        output = cells[-1]
        # output ⇒ each input
        for inp in inputs:
            compiler.add_implies(output, inp)
        # all inputs ⇒ output  i.e.  ¬inp₁ ∨ ¬inp₂ ∨ ... ∨ output
        compiler.add_raw_clause(
            [(inp, True, False) for inp in inputs]
            + [(output, True, True)]
        )
        return True

    if kind == "or" and len(cells) >= 3:
        # output = OR(inputs...)
        inputs = list(cells[:-1])
        output = cells[-1]
        # each input ⇒ output
        for inp in inputs:
            compiler.add_implies(inp, output)
        # inputs ∨ ¬output  i.e.  inp₁ ∨ inp₂ ∨ ... ∨ ¬output
        compiler.add_raw_clause(
            [(inp, True, True) for inp in inputs]
            + [(output, True, False)]
        )
        return True

    if kind == "switch" and len(cells) == 3:
        # switch(control, input, output) => if control then output = input
        control, inp, outp = cells
        control_val = fixed_values.get(control)
        if isinstance(control_val, bool):
            if control_val:
                compiler.add_equality(inp, outp)
                return True
            else:
                # control=False: output is free (no constraint)
                return True
        # Unpinned switch: can't express without ITE — fall through to skip

    # Comparison/equality propagators are discovered as ternary predicates:
    #   op(a, b, p) where p is a boolean output cell.
    # If p is fixed by require()/abhor(), lower to a direct binary relation.
    # Otherwise, use reified comparison: (p == (a <op> b)).
    if kind in {"eq", "lt", "gt", "lte", "gte"} and len(cells) == 3:
        lhs, rhs, predicate_cell = cells
        pred_value = fixed_values.get(predicate_cell)
        if isinstance(pred_value, bool):
            return _add_lowered_predicate_constraint(compiler, kind, lhs, rhs, pred_value)
        else:
            # Unpinned boolean output: use reified comparison
            _op_map = {'eq': 'eq', 'lt': 'lt', 'gt': 'gt', 'lte': 'lte', 'gte': 'gte'}
            compiler.add_reified_comparison(predicate_cell, lhs, rhs, _op_map[kind])
            return True

    return False


def _add_lowered_predicate_constraint(
    compiler: NetworkCompiler,
    kind: str,
    lhs: Any,
    rhs: Any,
    predicate_value: bool,
) -> bool:
    if kind == "eq":
        if predicate_value:
            compiler.add_equality(lhs, rhs)
        else:
            compiler.add_inequality(lhs, rhs)
        return True

    if kind == "lt":
        if predicate_value:
            compiler.add_less_than(lhs, rhs)
        else:
            compiler.add_greater_equal(lhs, rhs)
        return True

    if kind == "gt":
        if predicate_value:
            compiler.add_greater_than(lhs, rhs)
        else:
            compiler.add_less_equal(lhs, rhs)
        return True

    if kind == "lte":
        if predicate_value:
            compiler.add_less_equal(lhs, rhs)
        else:
            compiler.add_greater_than(lhs, rhs)
        return True

    if kind == "gte":
        if predicate_value:
            compiler.add_greater_equal(lhs, rhs)
        else:
            compiler.add_less_than(lhs, rhs)
        return True

    return False


def _add_tms_nogoods(compiler: NetworkCompiler, allowed_cells: List[Any]) -> None:
    try:
        from ..tms import get_all_nogoods, hypothetical_p
    except Exception:
        return

    allowed = set(allowed_cells)
    for nogood in get_all_nogoods():
        assignments = []
        for premise in nogood:
            if not hypothetical_p(premise):
                continue
            cell = getattr(premise, "output_cell", None)
            value = getattr(premise, "value_if_chosen", None)
            if isinstance(value, str) and value.startswith("one of"):
                continue
            if cell in allowed and value is not None:
                assignments.append((cell, value))
        if len(assignments) >= 2:
            compiler.add_nogood(assignments)


def _cell_name(cell: Any, known_names: Dict[Any, str]) -> str:
    return known_names.get(cell, getattr(cell, "name", None) or f"cell_{id(cell)}")


def _sorted_values(values) -> List[Any]:
    try:
        return sorted(list(values))
    except TypeError:
        return list(values)


def _domain_supported_by_backend(domain_values: List[Any], backend: SolverBackend) -> bool:
    return all(_value_supported_by_backend(v, backend) for v in domain_values)


def _value_supported_by_backend(value: Any, backend: SolverBackend) -> bool:
    if backend == SolverBackend.Z3_PYTHON:
        return isinstance(value, (int, bool))
    return True


def _format_issues(issues: List[TranslationIssue]) -> str:
    header = "compile_from_roots failed due to unsupported translation elements:"
    lines = [header]
    for issue in issues:
        cells = ", ".join(issue.cell_names) if issue.cell_names else "<none>"
        lines.append(f"- {issue.kind} on [{cells}]: {issue.reason}")
    return "\n".join(lines)
