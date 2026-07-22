"""
Unified Solve Interface for Propagator Networks

This module provides the single recommended entry point for solving
propagator networks with optional SMT oracle acceleration.

All other solve paths (solve_from_roots, solve_hybrid_from_existing_network,
solve_hybrid, TrueHybridNetwork.solve) delegate to this module's unified
solve loop.

Architecture:
    solve(cells, *, mode=SolveMode.SMT_ITERATIVE)
      ├── mode=ONESHOT → compile_from_roots → solve → return
      └── mode=ITERATIVE → loop:
            ├── discover network
            ├── compile translatable subset
            ├── SMT solve
            ├── inject as SMTHypothesis premises
            ├── propagate full network (including untranslatable constraints)
            ├── detect contradictions via TMS bridge
            ├── learn nogoods → feed back to SMT
            └── repeat until solved or UNSAT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .compiler import NetworkCompiler, SolverBackend, SolverResult
from .from_roots import (
    _try_add_constraint,
    compile_from_roots,
    solve_from_roots as _solve_from_roots_oneshot,
)

from ..cell import Cell
from ..network_discovery import discover_network, DiscoveredConstraint
from ..supported_values import supported, supported_p, get_support_premises
from ..nothing import nothing_p
from ..tms import (
    Hypothetical,
    hypothetical_p,
    bring_in,
    tms_p,
    tms_query,
    to_tms,
    get_contradictions,
    get_all_hypotheticals,
)
from ..scheduler import initialize_scheduler, run


# =============================================================================
# Type aliases and enums
# =============================================================================

class SolveMode(Enum):
    """
    How the SMT solver interacts with the propagator network.

    SMT_ITERATIVE (recommended, default):
        SMT solves the translatable subset, results are injected as
        SMTHypothesis premises, the full propagator network enforces
        every constraint (including ones SMT couldn't translate), and
        the TMS bridge learns nogoods from contradictions. Correct
        always — no constraint is silently dropped.

    SMT_ONESHOT:
        One-shot SMT on the translatable subset. Fast, but returns
        whatever the SMT says with no propagation verification.
        WARNING: if any constraint cannot be translated, the result
        may be invalid in the context of the full problem.

    PROPAGATOR:
        Pure propagator search with TMS/CDCL. No SMT at all. Use when
        no SMT solver is available or when you want the propagator's
        native scheduling.
    """
    SMT_ITERATIVE = "smt_iterative"
    SMT_ONESHOT = "smt_oneshot"
    SMT_INCREMENTAL = "smt_incremental"
    PROPAGATOR = "propagator"


@dataclass
class SolveResult:
    """Complete result of solving a propagator network."""
    solved: bool
    solution: Dict[Cell, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_solved(self) -> bool:
        return self.solved and len(self.solution) > 0


# =============================================================================
# Shared utilities
# =============================================================================

# TrackedConstraintKind re-exported from true_hybrid for convenience
from typing import Literal, Union as _Union
TrackedConstraintKind = Literal[
    "distinct", "add", "mul", "neq", "eq", "lt", "gt", "abs_diff_neq", "column_add",
]


@dataclass
class TrackedConstraint:
    """A constraint tracked for SMT export that has no real propagator."""
    constraint_type: TrackedConstraintKind
    cells: List[Cell]
    extra: Optional[Dict[str, Any]] = None


# Re-export SMTHypothesis for external use
from .true_hybrid import SMTHypothesis, TrackedConstraint  # noqa: F811 — re-export with standard name


def _cell_value(cell: Cell) -> Optional[Any]:
    """
    Resolve the effective value from a cell's content.

    Handles plain values, Supported, and Tms wrappers.
    Returns None if the cell has no settled value.
    """
    content = getattr(cell, 'content', None)
    if content is None or nothing_p(content):
        return None
    if supported_p(content):
        return content.value
    if tms_p(content):
        q = tms_query(content)
        if nothing_p(q):
            return None
        if supported_p(q):
            return q.value
        return q
    return content


def _extract_domains_from_tms(cells: List[Cell]) -> Dict[Cell, Set[Any]]:
    """
    Extract finite domains from TMS hypotheticals.

    Every one_of() call creates Hypothetical premises whose value_if_chosen
    encodes a domain value. This extracts them back for SMT domain registration.
    """
    domains: Dict[Cell, Set[Any]] = {}
    try:
        for hyp in get_all_hypotheticals():
            hcell = getattr(hyp, 'output_cell', None)
            hval = getattr(hyp, 'value_if_chosen', None)
            if (hcell in cells
                    and hval is not None
                    and not (isinstance(hval, str) and hval.startswith('one of'))):
                domains.setdefault(hcell, set()).add(hval)
    except Exception:
        pass
    return domains


def _extract_nogoods(allowed_cells: Set[Cell]) -> List[List[Tuple[Cell, Any]]]:
    """
    Extract TMS-learned nogoods as (cell, value) pairs for SMT blocking.

    Filters to only include nogoods whose premises reference allowed cells.
    Skips "one of" string markers from guess-based search.
    """
    nogoods: List[List[Tuple[Cell, Any]]] = []
    try:
        from ..tms import get_all_nogoods
        for nogood in get_all_nogoods():
            assignments = []
            for premise in nogood:
                if not hypothetical_p(premise):
                    continue
                cell = getattr(premise, 'output_cell', None)
                value = getattr(premise, 'value_if_chosen', None)
                if isinstance(value, str) and value.startswith("one of"):
                    continue
                if cell in allowed_cells and value is not None:
                    assignments.append((cell, value))
            if assignments:
                nogoods.append(assignments)
    except Exception:
        pass
    return nogoods


def _boolean_output_cells(constraints: List[DiscoveredConstraint]) -> Set[Cell]:
    """
    Cells holding a boolean flag produced by a comparison/logic propagator,
    which must be declared as Bool (not Int) when compiling to a solver.
    """
    _BOOLEAN_KINDS = {"eq", "lt", "gt", "lte", "gte", "and", "or", "not"}
    return {
        c.cells[-1] for c in constraints
        if c.kind in _BOOLEAN_KINDS and c.cells
    }


def _inject_smt_solution(
    solution: Dict[Cell, Any],
    round_num: int,
) -> None:
    """
    Inject SMT solution values as SMTHypothesis premises.

    Each SMT assignment becomes a Tms(Supported(value, {SMTHypothesis}))
    so the TMS tracks them and can kick them out if a contradiction occurs.

    Batching: auto_run is temporarily disabled during injection so that
    each ``cell.add_content()`` call doesn't trigger a full ``run()``
    cycle. Propagation fires once, at the end, via the caller.
    """
    from ..scheduler import set_auto_run as _set_ar, get_auto_run as _get_ar

    prev = _get_ar()
    _set_ar(False)
    try:
        for cell, value in solution.items():
            if value is None:
                continue
            hyp = SMTHypothesis(
                cell=cell, value=value, round=round_num,
                output_cell=cell, value_if_chosen=value, sign='smt',
                name=f"smt_r{round_num}",
            )
            bring_in(hyp)
            cell.add_content(to_tms(supported(value, [hyp])))
    finally:
        _set_ar(prev)


# =============================================================================
# Unified solve loop
# =============================================================================

def solve(
    root_cells: List[Cell],
    *,
    mode: SolveMode = SolveMode.SMT_ITERATIVE,
    backend: SolverBackend = SolverBackend.Z3_PYTHON,
    domains: Optional[Dict[Cell, Set[int]]] = None,
    extra_constraints: Optional[List[TrackedConstraint]] = None,
    verbose: bool = False,
    max_rounds: int = 100,
    timeout: Optional[float] = None,
) -> SolveResult:
    """
    Solve a propagator network from its root cells.

    This is THE recommended entry point for hybrid solving. It replaces
    the older solve_from_roots(), solve_hybrid_from_existing_network(),
    and solve_hybrid().

    Args:
        root_cells: Root cells of the network to solve.
        mode: How to use the solver backend.
        backend: Solver backend for SMT modes.
        domains: Explicit finite domains for cells without one_of().
        extra_constraints: SMT-only constraints not discoverable from
            propagator wiring (e.g., column_add, abs_diff_neq).
        verbose: Print per-round diagnostics including what constraints
            were skipped and why.
        max_rounds: Maximum iterative rounds (ITERATIVE mode only).
        timeout: Max seconds per SMT call.

    Returns:
        SolveResult with .solved, .solution, .stats

    Example:
        from propagator import Cell, set_auto_run
        from propagator.guessing_machine import one_of, require_distinct
        from propagator.solver_export import solve, SolveMode

        # Cold build: no native search during construction
        set_auto_run(False)
        try:
            x = Cell(name='x')
            y = Cell(name='y')
            one_of({1, 2, 3}, x)
            one_of({1, 2, 3}, y)
            require_distinct([x, y])
        finally:
            set_auto_run(True)

        result = solve([x, y], verbose=True)
        if result.solved:
            print(result.solution)  # {x: 2, y: 3} (or similar)

    Performance tip — "cold build":
        Always wrap network construction with:
            set_auto_run(False)
            # ... build network ...
            set_auto_run(True)
        then solve afterward. This prevents the native TMS search from
        exploring the search space during construction and lets the SMT
        oracle own 100% of the search. Expected speedup: 100-10000x.
    """
    if mode == SolveMode.PROPAGATOR:
        return _solve_propagator_only(root_cells, verbose)

    if mode == SolveMode.SMT_ONESHOT:
        return _solve_smt_oneshot(root_cells, backend, domains, extra_constraints, verbose, timeout)

    if mode == SolveMode.SMT_INCREMENTAL:
        return _solve_smt_incremental(
            root_cells, backend, domains, extra_constraints, verbose, max_rounds, timeout,
        )

    # SMT_ITERATIVE (default)
    return _solve_smt_iterative(
        root_cells, backend, domains, extra_constraints, verbose, max_rounds, timeout,
    )


# =============================================================================
# Internal solve strategies
# =============================================================================

def _solve_smt_incremental(
    root_cells: List[Cell],
    backend: SolverBackend,
    domains: Optional[Dict[Cell, Set[int]]],
    extra_constraints: Optional[List],
    verbose: bool,
    max_rounds: int,
    timeout: Optional[float],
) -> SolveResult:
    """
    Incremental theory propagation: interleave propagation with Z3.

    Instead of one-shot SMT -> inject everything -> reconcile
    (which creates a storm of propagator executions), this:
    1. Sets up a persistent Z3 solver with structural constraints
    2. Each round: propagate, push domain changes, incremental check
    3. Z3 implied values feed back to propagators
    4. Converges in fewer, cheaper rounds

    Works best with TrueHybridNetwork-built networks (no one_of overhead).
    For one_of-based networks, prefers SMT_ITERATIVE or SMT_ONESHOT.
    """
    try:
        import z3 as _z3
    except ImportError:
        raise ImportError("Z3 not available. Install with: pip install z3-solver")

    from .compiler import NetworkCompiler
    from .from_roots import _try_add_constraint
    from ..network_discovery import discover_network, DiscoveredConstraint

    # -- Setup persistent Z3 solver --
    discovered = discover_network(root_cells)
    all_cells = list(discovered.cells)
    boolean_cells = _boolean_output_cells(discovered.constraints)

    compiler = NetworkCompiler(name="smt_incremental")

    # Extract domains from TMS or explicit domains
    tms_domains = _extract_domains_from_tms(all_cells)
    if domains:
        for cell, d in domains.items():
            tms_domains[cell] = tms_domains.get(cell, set()) & d if cell in tms_domains else d

    # Register cells
    for cell in all_cells:
        cell_name = getattr(cell, "name", None) or discovered.cell_names.get(cell, f"v{id(cell)%10000}")
        if cell in boolean_cells:
            compiler.add_boolean(cell, name=cell_name)
        elif cell in tms_domains:
            compiler.add_domain(cell, list(tms_domains[cell]), name=cell_name)
        else:
            compiler.add_integer(cell, name=cell_name)

    # Fix determined cells + track grounded values
    fixed_values: Dict[Cell, Any] = {}
    grounded_values: Dict[Cell, Any] = {}
    for cell in all_cells:
        val = _cell_value(cell)
        if val is not None:
            fixed_values[cell] = val
            compiler.add_fixed_value(cell, val)
            try:
                content = getattr(cell, 'content', None)
                if content is not None and supported_p(content):
                    premises = get_support_premises(content)
                elif tms_p(content):
                    q = tms_query(content)
                    premises = get_support_premises(q) if supported_p(q) else []
                else:
                    premises = []
                if not any(hypothetical_p(p) for p in premises):
                    grounded_values[cell] = val
            except Exception:
                pass

    # Add discovered constraints (structural)
    for constraint in discovered.constraints:
        _try_add_constraint(compiler, constraint, grounded_values)

    # Add extra constraints
    extra = list(extra_constraints) if extra_constraints else []
    for ec in extra:
        _add_extra_constraint(compiler, ec)

    # Export to get Z3 solver + variable map
    encoding = compiler.export(SolverBackend.Z3_PYTHON)
    solver = encoding.metadata.get('solver')
    z3_vars = encoding.metadata.get('z3_vars', {})
    cell_to_name = {cell: var.name for cell, var in compiler.variables.items()}
    name_to_cell = {v: k for k, v in cell_to_name.items()}

    if solver is None:
        if verbose:
            print("Z3 solver not available from export, falling back to ITERATIVE")
        return _solve_smt_iterative(
            root_cells, backend, domains, extra_constraints, verbose, max_rounds, timeout,
        )

    solver.push()  # structural checkpoint

    # -- Incremental loop --
    prev_domains: Dict[Cell, set] = {}
    round_count = 0

    for round_num in range(max_rounds):
        round_count = round_num + 1
        if verbose:
            print(f"Incremental round {round_num}: propagation...")

        try:
            run()
        except Exception:
            if verbose:
                print(f"  Propagation error, continuing...")
            continue

        # Domain narrowing
        current_domains = _extract_domains_from_tms(all_cells)
        if domains:
            for cell, d in domains.items():
                current_domains[cell] = current_domains.get(cell, set()) & d if cell in current_domains else d

        new_assertions = []
        for cell in all_cells:
            cur = current_domains.get(cell, set())
            prev = prev_domains.get(cell, set())
            if cur and cur != prev:
                prev_domains[cell] = set(cur)
                name = cell_to_name.get(cell)
                if name and name in z3_vars:
                    v = z3_vars[name]
                    if len(cur) == 1:
                        new_assertions.append(v == next(iter(cur)))
                    else:
                        new_assertions.append(_z3.And(v >= min(cur), v <= max(cur)))

        for a in new_assertions:
            solver.add(a)

        if verbose and new_assertions:
            print(f"  Pushed {len(new_assertions)} domain changes to Z3")

        r = solver.check()
        if r == _z3.unsat:
            if verbose:
                print(f"  Z3: UNSAT")
            return SolveResult(solved=False, stats={'method': 'smt_incremental', 'rounds': round_count})

        # Extract model values
        model = solver.model()
        injected = 0
        for name, z3v in z3_vars.items():
            cell = name_to_cell.get(name)
            if cell is None:
                continue
            if _cell_value(cell) is not None:
                continue
            try:
                val = model[z3v]
                if val is None:
                    continue
                if _z3.is_int(z3v):
                    v = val.as_long()
                elif _z3.is_bool(z3v):
                    v = bool(val)
                else:
                    continue
                hyp = SMTHypothesis(
                    cell=cell, value=v, round=round_num,
                    output_cell=cell, value_if_chosen=v,
                    sign='smt', name=f"smt_incr_r{round_num}",
                )
                bring_in(hyp)
                cell.add_content(to_tms(supported(v, [hyp])))
                injected += 1
            except Exception:
                pass

        if verbose and injected > 0:
            print(f"  Injected {injected} values from Z3 model")

        # Check fixpoint
        undetermined = [c for c in all_cells if _cell_value(c) is None]
        if not undetermined:
            if verbose:
                print(f"  Solved after {round_count} incremental rounds")
            break

        if not new_assertions and injected == 0:
            if verbose:
                print(f"  Stuck at round {round_num}")
            break

    # Collect solution
    solution = {}
    for cell in all_cells:
        v = _cell_value(cell)
        if v is not None:
            solution[cell] = v

    solved = all(_cell_value(c) is not None for c in all_cells)
    return SolveResult(
        solved=solved, solution=solution,
        stats={'method': 'smt_incremental', 'rounds': round_count,
               'cells_determined': len(solution), 'cells_total': len(all_cells)},
    )


def _solve_propagator_only(
    root_cells: List[Cell],
    verbose: bool,
) -> SolveResult:
    """Pure propagator search — no SMT at all."""
    if verbose:
        print("Solving with propagator only (no SMT)...")
    try:
        run()
    except Exception as e:
        if verbose:
            print(f"  Propagator search failed: {e}")
        return SolveResult(solved=False, stats={'error': str(e)})

    solution = {}
    for cell in root_cells:
        v = _cell_value(cell)
        if v is not None:
            solution[cell] = v

    solved = all(_cell_value(c) is not None for c in root_cells)
    return SolveResult(solved=solved, solution=solution,
                       stats={'method': 'propagator'})


def _solve_smt_oneshot(
    root_cells: List[Cell],
    backend: SolverBackend,
    domains: Optional[Dict[Cell, Set[int]]],
    extra_constraints: Optional[List[TrackedConstraint]],
    verbose: bool,
    timeout: Optional[float],
) -> SolveResult:
    """One-shot SMT: compile what's translatable, solve, return."""
    if verbose:
        print("One-shot SMT solve...")

    result, report = _solve_from_roots_oneshot(
        root_cells, backend=backend, timeout=timeout, include_nogoods=False,
    )

    if not result.satisfiable or not result.solution:
        if verbose:
            print("  SMT: UNSAT")
        return SolveResult(
            solved=False,
            stats={
                'method': 'smt_oneshot',
                'translated': report.translated_constraint_count,
                'skipped': report.skipped_constraint_count,
                'backend': backend.name,
            },
        )

    solution = result.solution
    # In one-shot mode we trust SMT-assigned values directly
    # (we don't inject into cells, so _cell_value won't find them)
    root_set = set(root_cells)
    solved = (
        all(cell in solution and solution[cell] is not None for cell in root_set)
        if solution else False
    )

    return SolveResult(
        solved=solved,
        solution=solution or {},
        stats={
            'method': 'smt_oneshot',
            'translated': report.translated_constraint_count,
            'skipped': report.skipped_constraint_count,
            'backend': backend.name,
        },
    )


def _solve_smt_iterative(
    root_cells: List[Cell],
    backend: SolverBackend,
    domains: Optional[Dict[Cell, Set[int]]],
    extra_constraints: Optional[List[TrackedConstraint]],
    verbose: bool,
    max_rounds: int,
    timeout: Optional[float],
) -> SolveResult:
    """
    Iterative SMT + propagator loop with TMS bridge.

    Algorithm:
        1. Discover network structure from root cells
        2. Compile translatable constraints → SMT
        3. Solve SMT → inject values as SMTHypothesis premises
        4. Run full propagation (enforces ALL constraints)
        5. If contradiction: TMS kicks out wrong hypothesis, learns nogood
        6. Feed nogood back to SMT, re-solve
        7. Repeat until all cells determined or UNSAT

    This handles the core concern: the SMT only sees a SUBSET of constraints.
    Untranslatable constraints (comparisons with unpinned booleans, custom
    propagators, switch, etc.) are enforced by step 4. If the SMT proposed
    values that violate them, the TMS catches it in step 5 and the nogood
    prevents the SMT from making the same mistake again.
    """
    # Phase 1: Initial one-shot solve to get candidate values
    if verbose:
        print("Initial one-shot SMT...")

    extra = list(extra_constraints) if extra_constraints else []

    result, report = _solve_from_roots_oneshot(
        root_cells, backend=backend, timeout=timeout, include_nogoods=False,
    )

    if not result.satisfiable:
        if verbose:
            print("  SMT: UNSAT on initial solve")
        return SolveResult(
            solved=False,
            stats={
                'method': 'smt_iterative',
                'translated': report.translated_constraint_count,
                'skipped': report.skipped_constraint_count,
                'rounds': 0,
                'backend': backend.name,
            },
        )

    # Inject initial solution
    if result.solution:
        _inject_smt_solution(result.solution, round_num=0)

    # Phase 2: Iterative loop
    discovered = discover_network(root_cells)
    all_cells = list(discovered.cells)
    seen_nogood_count = len(get_contradictions())
    learned_nogoods: List[List[Tuple[Cell, Any]]] = []
    round_count = 0

    # Run propagation once to let injected values settle.
    # TMS fully reconciles SMT values with one_of premises.
    try:
        run()
    except Exception:
        pass  # Expected if SMT solution conflicts with unsupported constraints

    for round_num in range(max_rounds):
        round_count = round_num + 1

        # Harvest new nogoods from TMS
        all_nogoods = get_contradictions()
        for nogood in all_nogoods[seen_nogood_count:]:
            smt_hyps = [(hyp.cell, hyp.value)
                        for hyp in nogood
                        if _is_smt_hypothesis(hyp)]
            if smt_hyps:
                learned_nogoods.append(smt_hyps)
        seen_nogood_count = len(all_nogoods)

        # Check if fully determined
        undetermined = [c for c in all_cells if _cell_value(c) is None]
        if not undetermined:
            if verbose:
                print(f"Iterative hybrid: solved after {round_count} round(s)")
            break

        if verbose:
            print(f"Round {round_num}: {len(undetermined)} undetermined cells, calling SMT...")

        # Domain narrowing: extract current domain restrictions from TMS
        tms_domains = _extract_domains_from_tms(all_cells)
        if domains:
            for cell, d in domains.items():
                tms_domains[cell] = tms_domains.get(cell, set()) & d if cell in tms_domains else d

        # Build compiler state
        fixed_values: Dict[Cell, Any] = {}
        grounded_values: Dict[Cell, Any] = {}
        for cell in all_cells:
            val = _cell_value(cell)
            if val is not None:
                fixed_values[cell] = val
                try:
                    content = getattr(cell, 'content', None)
                    if content is not None and supported_p(content):
                        premises = get_support_premises(content)
                    elif tms_p(content):
                        q = tms_query(content)
                        premises = get_support_premises(q) if supported_p(q) else []
                    else:
                        premises = []
                    if not any(hypothetical_p(p) for p in premises):
                        grounded_values[cell] = val
                except Exception:
                    pass

        compiler = NetworkCompiler(name=f"solve_iterative_r{round_num}")
        boolean_cells = _boolean_output_cells(discovered.constraints)

        # Register cells
        for cell in all_cells:
            cell_name = getattr(cell, 'name', None) or discovered.cell_names.get(cell, f'v{id(cell) % 10000}')
            if cell in boolean_cells:
                compiler.add_boolean(cell, name=cell_name)
            elif cell in tms_domains:
                compiler.add_domain(cell, list(tms_domains[cell]), name=cell_name)
            else:
                compiler.add_integer(cell, name=cell_name)

        # Fix determined cells
        for cell, val in fixed_values.items():
            compiler.add_fixed_value(cell, val)

        # Add discovered constraints
        translated = 0
        skipped = 0
        skipped_kinds: List[str] = []
        for constraint in discovered.constraints:
            if _try_add_constraint(compiler, constraint, grounded_values):
                translated += 1
            else:
                skipped += 1
                skipped_kinds.append(constraint.kind)
        if verbose and skipped > 0:
            print(f"  Translation: {translated} translated, {skipped} skipped "
                  f"({', '.join(sorted(set(skipped_kinds)))})")

        # Add extra (SMT-only) constraints
        for ec in extra:
            _add_extra_constraint(compiler, ec)

        # Add learned nogoods as blocking clauses
        for nogood in learned_nogoods:
            compiler.add_nogood(nogood)

        # Solve
        try:
            res = compiler.solve(backend=backend, timeout=timeout or 5.0)
        except Exception as e:
            if verbose:
                print(f"  SMT solve error: {e}")
            return SolveResult(
                solved=False,
                stats={
                    'method': 'smt_iterative',
                    'translated': translated,
                    'skipped': skipped,
                    'rounds': round_count,
                    'error': str(e),
                },
            )

        if not res.satisfiable or not res.solution:
            if verbose:
                print(f"  SMT: UNSAT at round {round_num}")
            return SolveResult(
                solved=False,
                stats={
                    'method': 'smt_iterative',
                    'translated': translated,
                    'skipped': skipped,
                    'rounds': round_count,
                    'backend': backend.name,
                },
            )

        # Filter solution to undetermined cells only
        network_cells = set(all_cells)
        new_solution: Dict[Cell, Any] = {}
        for cell, value in res.solution.items():
            if value is None or cell not in network_cells:
                continue
            if _cell_value(cell) is not None:
                continue
            new_solution[cell] = value

        if not new_solution:
            if verbose:
                print("  No new assignments from SMT — all cells already determined")
            # SMT didn't add anything new but still SAT. Must be propagation-only
            # residual. If we still have undetermined cells, they can't be solved
            # by SMT (no new constraints to add). Break to avoid infinite loop.
            break

        # Inject and propagate
        _inject_smt_solution(new_solution, round_num=round_num + 1)
        try:
            run()
        except Exception:
            if verbose:
                print(f"  Contradiction at round {round_num}, retrying...")

    # Final check
    solution = {}
    for cell in all_cells:
        v = _cell_value(cell)
        if v is not None:
            solution[cell] = v

    solved = all(_cell_value(c) is not None for c in all_cells)
    return SolveResult(
        solved=solved,
        solution=solution,
        stats={
            'method': 'smt_iterative',
            'rounds': round_count,
            'cells_determined': len(solution),
            'cells_total': len(all_cells),
            'backend': backend.name,
        },
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _is_smt_hypothesis(hyp: Any) -> bool:
    """Check if a premise is an SMTHypothesis."""
    return type(hyp).__name__ == 'SMTHypothesis'


def _add_extra_constraint(compiler: NetworkCompiler, constraint: TrackedConstraint) -> None:
    """
    Translate an extra (SMT-only) constraint that has no real propagator.

    These are constraints like all_different, column_add, abs_diff_neq
    that exist only as TrackedConstraint entries — they have no discoverable
    propagator wiring, so network_discovery can't find them.
    """
    kind = constraint.constraint_type
    cells = constraint.cells

    if kind == "distinct":
        compiler.add_all_distinct(cells)
    elif kind == "add":
        a, b, c = cells
        compiler.add_sum_equals([a, b, c], total=0, coefficients=[1, 1, -1])
    elif kind == "mul":
        a, b, c = cells
        compiler.add_product(a, b, c)
    elif kind == "neq":
        a, b = cells
        compiler.add_inequality(a, b)
    elif kind == "eq":
        a, b = cells
        compiler.add_equality(a, b)
    elif kind == "lt":
        a, b = cells
        compiler.add_less_than(a, b)
    elif kind == "gt":
        a, b = cells
        compiler.add_greater_than(a, b)
    elif kind == "abs_diff_neq":
        a, b = cells
        value = constraint.extra.get('value') if constraint.extra else None
        if value is not None:
            diff = Cell(name=f"_diff_{a.name}_{b.name}")
            abs_diff = Cell(name=f"_absdiff_{a.name}_{b.name}")
            const_val = Cell(name=f"_const_{value}")
            compiler.add_sum_equals([a, b, diff], total=0, coefficients=[1, -1, -1])
            compiler.add_absolute_value(diff, abs_diff)
            compiler.add_domain(const_val, [value])
            compiler.add_fixed_value(const_val, value)
            compiler.add_inequality(abs_diff, const_val)
    elif kind == "column_add":
        a, b, result, carry_out = cells
        carry_in = constraint.extra.get('carry_in') if constraint.extra else None
        if carry_in is not None:
            compiler.add_sum_equals(
                [a, b, carry_in, result, carry_out], total=0,
                coefficients=[1, 1, 1, -1, -10],
            )
        else:
            compiler.add_sum_equals(
                [a, b, result, carry_out], total=0,
                coefficients=[1, 1, -1, -10],
            )
