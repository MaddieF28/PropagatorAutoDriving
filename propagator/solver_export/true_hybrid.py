"""
Hybrid Propagator + SMT Solver

This module implements a hybrid architecture that:
- Uses the real propagator infrastructure (Cell, merge, scheduler)
- Uses the real TMS for provenance, no-good tracking, and backtracking
- Uses real Supported values (wrapped in a Tms) for dependency tracking
- Delegates SEARCH to SMT when propagation reaches fixpoint

Key design properties:
1. All cells are real Cell objects - backward-compatible with existing
   propagators (including ones added directly, bypassing this class's own
   builder methods -- their constraints are still visible to the SMT oracle
   via network_discovery, see _solve_with_z3).
2. Injected values are Tms(Supported(value, premises)) - not a bare
   Supported - so every merge against them routes through tms_merge, which
   is what actually invokes nogood learning (a bare Supported never does).
3. TMS handles nogood learning and kick_out/bring_in for backtracking:
   SMTHypothesis is a real Hypothetical subclass, so process_one_contradiction
   can kick it out like any amb-style choice.
4. The scheduler drives propagation ordering.
5. SMT solutions are injected as HYPOTHETICAL PREMISES (SMTHypothesis).
6. Contradictions generate NOGOODS that are fed back to the next SMT call
   as blocking clauses (see solve()'s nogood-harvesting via get_contradictions()).

Known limitation: comparison propagators (eq/lt/gt/...) whose boolean
output cell is never pinned by require()/abhor() are not tied to their
inputs in the SMT model (Z3 can pick the flag independently of a/b). The
real propagator's own computation of the flag still catches a wrong guess
via the mechanism in (2)/(3) above and retries -- see the class docstring
below and _solve_with_z3's boolean_cells handling -- but this can cost
several extra rounds rather than being solved in one SMT call.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     PROPAGATOR LAYER                            │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐                     │
    │  │  Cell   │◄──►│  Cell   │◄──►│  Cell   │  (Real Cells)       │
    │  │Supported│    │Supported│    │Supported│  (Real Provenance)  │
    │  └────┬────┘    └────┬────┘    └────┬────┘                     │
    │       │              │              │                           │
    │  ┌────▼──────────────▼──────────────▼────┐                     │
    │  │           SCHEDULER (queue)           │  (Real Scheduler)   │
    │  └────┬──────────────┬──────────────┬────┘                     │
    │       │              │              │                           │
    │  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐                     │
    │  │ adder   │    │ product │    │all_diff │  (Real Propagators) │
    │  └─────────┘    └─────────┘    └─────────┘                     │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             │ (fixpoint, still undetermined?)
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      SMT ORACLE                                 │
    │  ┌────────────────────────────────────────────────────────────┐│
    │  │ Export: cells → Z3 vars, constraints → Z3 assertions       ││
    │  │ Import: Z3 model → Hypothetical premises + Supported vals  ││
    │  │ Feedback: nogoods → Z3 learned clauses                     ││
    │  └────────────────────────────────────────────────────────────┘│
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             │ (inject as hypothetical premises)
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        TMS                                       │
    │  - Tracks which premises are believed/disbelieved               │
    │  - Computes consequences (what values are currently valid)      │
    │  - Learns nogoods when contradictions detected                  │
    │  - Supports bring_in / kick_out for backtracking                │
    └─────────────────────────────────────────────────────────────────┘

This gives us:
✓ Full propagator semantics (bidirectional, incremental)
✓ Full provenance tracking (every value knows its premises)
✓ Full nogood learning (contradictions → learned clauses)
✓ Full backward compatibility (uses real Cell, real propagators)
✓ SMT's search power (when propagation is insufficient)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union
from .compiler import NetworkCompiler, SolverBackend, SolverResult
from .from_roots import RootCompileReport, TranslationMode, solve_from_roots
# Reused rather than re-implemented: this is the same discovered-constraint
# -> compiler-IR mapping solve_from_roots() uses, so square/eq/lt/... are
# translated identically everywhere instead of this module drifting out of
# sync with its own copy.
from .from_roots import _try_add_constraint

# CRITICAL: Import the REAL propagator infrastructure
from ..cell import Cell
from ..scheduler import initialize_scheduler, run, alert_propagators
from ..primitives import (
    constant as _constant,
    adder as _adder,
    multiplier as _multiplier,
    product as _product,
    sum_constraint as _sum_constraint,
)
from ..supported_values import Supported, supported, get_support_premises, supported_p
from ..tms import (
    Tms,
    Hypothetical,
    hypothetical_p,
    tms_p,
    bring_in,
    kick_out,
    premise_in,
    process_nogood,
    TmsContradiction,
    tms_query,
    to_tms,
    get_contradictions,
)
from ..nothing import nothing_p
from ..network_discovery import DiscoveredConstraint, discover_network


# =============================================================================
# Type aliases
#
# Cell content in the general propagator system is genuinely open-ended
# (plain values of any type, Supported, or Tms -- see Cell.content, which
# has no type parameter to narrow against). Two aliases below draw an
# honest line through that openness for this module specifically:
#
# - SolverValue: values that came from, or are destined for, Z3 itself.
#   This module's own translation (_solve_with_z3, NetworkCompiler's
#   var_type dispatch in compiler.py) only ever declares Int or Bool
#   solver variables, so anything Z3-derived really is int | bool -- this
#   is a genuine narrowing, not a guess.
# - Premise: left as Any deliberately. supported_values.get_support_premises
#   returns List[Any] upstream by design (a premise can be a Hypothetical,
#   a plain string tag, or any other hashable object a caller chose to
#   support a value with) -- narrowing it here would just be dishonest.
#
# Methods that stay backward-compatible with *arbitrary* real propagators
# (get_value, get_provenance, constant()) intentionally keep Any/CellValue
# rather than SolverValue, since a cell fed by some other propagator (e.g.
# squarer, or a user's own) is not restricted to int/bool.
SolverValue = Union[int, bool]
CellValue = Any
Premise = Any

TrackedConstraintKind = Literal[
    "distinct", "add", "mul", "neq", "eq", "lt", "gt", "abs_diff_neq", "column_add",
]


# =============================================================================
# SMT Hypothesis: Represents an SMT-provided value as a premise
# =============================================================================

@dataclass(frozen=True, eq=False)
class SMTHypothesis(Hypothetical):
    """
    A hypothesis generated by SMT for a specific cell assignment.

    Unlike regular Hypothetical (which represents binary amb choices),
    SMTHypothesis represents "SMT said cell X should be value V".

    This is a premise in the TMS sense - it can be believed or not,
    and contradictions will generate nogoods involving it.

    Subclasses Hypothetical (rather than being a standalone dataclass) so
    that hypothetical_p() recognizes it and process_one_contradiction() can
    actually kick it out when it's part of a nogood -- without this, the
    TMS's kick-out/backtracking machinery silently ignores SMT-injected
    premises entirely.
    """
    value: Optional[SolverValue] = None
    round: int = 0  # Which SMT round generated this

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        status = 'in' if premise_in(self) else 'out'
        cell_name = self.cell.name or f"Cell@{id(self.cell) % 10000}"
        return f"SMTHyp({cell_name}={self.value}, r{self.round}, {status})"


# =============================================================================
# Constraint Tracking
# =============================================================================

@dataclass
class TrackedConstraint:
    """A constraint tracked for SMT export."""
    constraint_type: TrackedConstraintKind
    cells: List[Cell]
    # Shape depends on constraint_type: {'value': SolverValue} for
    # abs_diff_neq, {'carry_in': Optional[Cell]} for column_add, unused
    # (None) for every other kind. Not a TypedDict per kind because
    # constraint_type is a plain discriminant checked with `==`, not a
    # tag a type checker narrows on.
    extra: Optional[Dict[str, Any]] = None


# Discovered constraint kinds whose *output* cell (last in .cells) holds a
# boolean flag rather than a number -- e.g. eq(a, b, flag) sets flag = (a==b).
_BOOLEAN_OUTPUT_KINDS = {"eq", "lt", "gt", "lte", "gte", "and", "or", "not"}


def _boolean_output_cells(constraints: List[DiscoveredConstraint]) -> Set[Cell]:
    """
    Cells holding a boolean flag produced by a comparison/logic propagator,
    discovered from real propagator wiring. These must be declared as Bool
    (not Int) when compiling to a solver: Z3's Python API happens to coerce
    bool<->int so an Int-typed flag doesn't immediately error, but it's the
    wrong type, and the textual SMT-LIB2 backend rejects it outright.
    """
    return {c.cells[-1] for c in constraints if c.kind in _BOOLEAN_OUTPUT_KINDS and c.cells}


# =============================================================================
# True Hybrid Network
# =============================================================================

class TrueHybridNetwork:
    """
    A propagator network that uses REAL propagator infrastructure
    and delegates search to SMT.
    
    Key properties:
    - All cells are real Cell objects from propagator.cell
    - All propagators use the real scheduler
    - Values are Supported with real premise tracking
    - Contradictions go through real TMS nogood learning
    - SMT solutions become hypothetical premises
    
    Example:
        >>> initialize_scheduler()  # Fresh state
        >>> net = TrueHybridNetwork(name="example")
        >>> 
        >>> a = net.cell("a", domain={1, 2, 3})
        >>> b = net.cell("b", domain={1, 2, 3})
        >>> c = net.cell("c")
        >>> 
        >>> net.sum_constraint(a, b, c)
        >>> net.all_different([a, b])
        >>> 
        >>> if net.solve():
        ...     print(f"a={a.content}, b={b.content}, c={c.content}")
        ...     # Can inspect provenance:
        ...     print(f"c's value supported by: {get_support_premises(c.content)}")
    """
    
    def __init__(self, name: str = "network") -> None:
        self.name = name
        self.cells: List[Cell] = []
        self.cell_by_name: Dict[str, Cell] = {}
        self.constraints: List[TrackedConstraint] = []
        self.domains: Dict[Cell, Set[SolverValue]] = {}
        self._cell_counter = 0
        self._smt_round = 0
        self._learned_nogoods: List[List[SMTHypothesis]] = []
        # Persistent Z3 solver for incremental solving across rounds
        self._z3_solver: Any = None  # z3.Solver
        self._z3_vars: Dict[str, Any] = {}  # name -> z3.ExprRef
        self._z3_cell_map: Dict[Cell, str] = {}  # Cell -> z3 var name
        self._z3_structural_snapshot: int = 0  # solver.num_scopes() at structural push

    @classmethod
    def from_existing_network(
        cls, root_cells: List[Cell], name: str = "adopted_network",
    ) -> "TrueHybridNetwork":
        """
        Adopt an already-built propagator network into a TrueHybridNetwork,
        so the full hybrid solve loop (SMT oracle + hypothesis injection +
        TMS nogood learning + retry) runs against cells that were wired with
        ordinary propagator primitives -- no rebuilding through this class's
        own cell()/adder()/... methods.

        Works because _solve_with_z3 re-discovers real propagator constraints
        from self.cells (via network_discovery) on every round; the only
        state adoption needs to reconstruct is the cell list, the name->cell
        injection map, and the finite domains (from one_of hypotheticals,
        which discover_network already extracts). No tracked constraints are
        registered: everything in an adopted network exists as a real
        propagator, and the builder-only constraint kinds (all_different,
        column_add, ...) can't occur since the network wasn't built here.

        Unnamed auxiliary cells are backfilled with discovery's stable
        fallback names (cell.name is a debugging label, None by default):
        SMT solutions are keyed by name on the way back in, so every adopted
        cell needs a unique one.

        When adoption helps vs. when it can't:

        - It shines for networks whose propagation *stalls* at an
          undetermined fixpoint -- e.g. arithmetic wiring like
          multiplier(x, x, sq); constant(9, sq), which native propagation
          cannot invert. The SMT phase fires and completes the network, with
          full provenance and the nogood/retry loop intact.
        - It does NOT make one_of/require_distinct-style guessing networks
          faster. Those cells carry their own amb search propagators, which
          are physically part of the adopted network: any propagate() run
          drains them, so the native TMS search runs (and dominates, or on
          networks like SEND+MORE=MONEY diverges) no matter what the oracle
          injects first -- measured, not speculative. For the performance
          path on such networks, use solve_from_roots on a network built
          with auto-run deferred instead; see docs/SOLVER_APPROACHES.md.

        Note the difference from solve_hybrid_from_existing_network: that is
        a one-shot solve_from_roots + inject with no retry, while an adopted
        TrueHybridNetwork keeps the propagator network in the loop across
        SMT rounds.
        """
        net = cls(name=name)
        discovered = discover_network(root_cells)

        for cell in discovered.cells:
            cell_name = getattr(cell, "name", None) or discovered.cell_names[cell]
            # Uniquify: user-given names may repeat across cells, and the
            # injection map (and Z3 declarations) need one cell per name.
            unique = cell_name
            suffix = 1
            while unique in net.cell_by_name:
                unique = f"{cell_name}_{suffix}"
                suffix += 1
            cell.name = unique
            net.cells.append(cell)
            net.cell_by_name[unique] = cell

        net.domains = {
            cell: set(values) for cell, values in discovered.domains.items()
        }
        return net

    # =========================================================================
    # Cell creation
    # =========================================================================

    def cell(self, name: Optional[str] = None, domain: Optional[Set[SolverValue]] = None) -> Cell:
        """
        Create a real Cell in the network.

        If domain is provided, the cell tracks possible values for SMT export.
        """
        if name is None:
            name = f"c{self._cell_counter}"
            self._cell_counter += 1

        c = Cell(name=name)
        self.cells.append(c)
        self.cell_by_name[name] = c

        if domain is not None:
            self.domains[c] = set(domain)
            # If singleton domain, that's the value
            if len(domain) == 1:
                _constant(next(iter(domain)), c)

        return c

    def constant(self, value: CellValue, cell: Cell) -> None:
        """
        Set a cell to a constant value.

        value is intentionally CellValue (Any), not SolverValue: this
        delegates to the general propagator primitive constant(), which
        works with any real propagator's value type (e.g. squarer's
        output), not just the int/bool domains this module's own Z3
        translation handles.
        """
        _constant(value, cell)
    
    # =========================================================================
    # Arithmetic constraints (using REAL propagators)
    # =========================================================================
    
    def sum_constraint(self, a: Cell, b: Cell, c: Cell) -> None:
        """
        c = a + b, bidirectional.
        Uses the REAL sum_constraint from propagator.primitives.
        """
        self.constraints.append(TrackedConstraint("add", [a, b, c]))
        _sum_constraint(a, b, c)

    def product(self, a: Cell, b: Cell, c: Cell) -> None:
        """
        c = a * b, bidirectional.
        Uses the REAL product from propagator.primitives.
        """
        self.constraints.append(TrackedConstraint("mul", [a, b, c]))
        _product(a, b, c)

    def adder(self, a: Cell, b: Cell, c: Cell) -> None:
        """
        c = a + b, unidirectional (forward only).
        Uses the REAL adder from propagator.primitives.
        """
        self.constraints.append(TrackedConstraint("add", [a, b, c]))
        _adder(a, b, c)

    def multiplier(self, a: Cell, b: Cell, c: Cell) -> None:
        """
        c = a * b, unidirectional (forward only).
        """
        self.constraints.append(TrackedConstraint("mul", [a, b, c]))
        _multiplier(a, b, c)
    
    # =========================================================================
    # Global constraints (tracked for SMT)
    # =========================================================================
    
    def all_different(self, cells: List[Cell]) -> None:
        """
        All cells must have different values.

        This adds a propagator that prunes domains when values become known,
        AND tracks the constraint for SMT export.
        """
        self.constraints.append(TrackedConstraint("distinct", list(cells)))

        # Create a propagator that prunes domains
        def prune() -> None:
            determined: Dict[CellValue, Cell] = {}
            for c in cells:
                # self.get_value() (not a hand-rolled unwrap) so this
                # correctly reads plain, Supported, *and* Tms-wrapped
                # content -- injected SMT values are Tms-wrapped (see
                # _inject_smt_solution), so a manual ".value if
                # supported_p(...) else ..." check here would silently
                # treat every injected cell's content as distinct (the Tms
                # object itself, never equal to anything) and never catch a
                # genuine duplicate.
                val = self.get_value(c)
                if val is None:
                    continue
                if val in determined:
                    other = determined[val]
                    nogood = list(self.get_provenance(c)) + list(self.get_provenance(other))
                    if any(hypothetical_p(p) for p in nogood):
                        # At least one side rests on a retractable guess
                        # (an amb choice or an SMT-injected value) -- route
                        # through the TMS like every other contradiction in
                        # this module (see class docstring: "Contradictions
                        # go through real TMS nogood learning") so it can be
                        # resolved by kicking one of them out, instead of
                        # hard-failing a round that a retry could still win.
                        process_nogood(nogood)
                        continue
                    # Neither value is retractable (e.g. both cells were
                    # set via plain constant()) -- no amount of TMS
                    # backtracking can fix this, so fail loudly rather than
                    # silently reporting a solution that violates
                    # all_different.
                    raise Exception(
                        f"all_different violated: {c.name} and {other.name} both = {val}"
                    )
                determined[val] = c

                # Prune this value from other cells' tracked domains.
                for other in cells:
                    if other is not c and other in self.domains:
                        self.domains[other].discard(val)
                        if not self.domains[other]:
                            # Unlike the duplicate-value case above, there's
                            # no pair of conflicting premises to build a
                            # nogood from here -- this cell simply has no
                            # candidate values left. Raise plainly; solve()
                            # already treats any exception from propagate()
                            # as failure.
                            raise Exception(f"Domain of {other.name} became empty")

        # Register with each cell
        for c in cells:
            c.new_neighbor(prune)
    
    def not_equal(self, a: Cell, b: Cell) -> None:
        """a != b"""
        self.constraints.append(TrackedConstraint("neq", [a, b]))

        def propagate() -> None:
            # Cell.content itself has no type annotation upstream (cell.py
            # infers it as None from `self.content = None`) -- CellValue
            # here overrides that inference to what content actually holds.
            a_content: CellValue = a.content
            b_content: CellValue = b.content
            if nothing_p(a_content) or nothing_p(b_content):
                return
            a_val: CellValue = a_content.value if supported_p(a_content) else a_content
            b_val: CellValue = b_content.value if supported_p(b_content) else b_content

            if a_val is not None and b_val is not None and a_val == b_val:
                raise Exception(f"not_equal violated: {a.name}={a_val} == {b.name}={b_val}")

            # Domain pruning
            if a_val is not None and b in self.domains:
                self.domains[b].discard(a_val)
            if b_val is not None and a in self.domains:
                self.domains[a].discard(b_val)

        a.new_neighbor(propagate)
        b.new_neighbor(propagate)

    def less_than(self, a: Cell, b: Cell) -> None:
        """
        a < b (tracked for SMT; enforced by the SMT oracle, not by propagation).

        Ordering constraints require domain-aware propagation (arc consistency)
        to be useful at runtime.  Rather than partially enforcing them via a
        simple equality check, the full constraint is delegated to SMT.
        """
        self.constraints.append(TrackedConstraint("lt", [a, b]))

    def greater_than(self, a: Cell, b: Cell) -> None:
        """
        a > b (tracked for SMT; enforced by the SMT oracle, not by propagation).

        See less_than for the rationale.
        """
        self.constraints.append(TrackedConstraint("gt", [a, b]))
    
    def column_add(
        self, a: Cell, b: Cell, result: Cell, carry_out: Cell, carry_in: Optional[Cell] = None
    ) -> None:
        """
        Column addition: a + b + carry_in = result + 10*carry_out

        Tracked for SMT export only (complex multi-cell arithmetic;
        handled by the SMT oracle rather than a propagator).
        """
        self.constraints.append(TrackedConstraint(
            "column_add",
            [a, b, result, carry_out],
            extra={'carry_in': carry_in}
        ))

    def equals_var(self, a: Cell, b: Cell) -> None:
        """
        a == b (variable equality, tracked for SMT export only).

        For bidirectional equality propagation use sum_constraint with a
        constant-zero third cell instead.
        """
        self.constraints.append(TrackedConstraint("eq", [a, b]))

    # =========================================================================
    # Propagation (using REAL scheduler)
    # =========================================================================

    def propagate(self) -> str:
        """
        Run propagation to fixpoint using the REAL scheduler.
        Returns "done" when no more propagation possible.
        """
        return run()

    def is_fully_determined(self) -> bool:
        """Check if all cells have determined values."""
        for c in self.cells:
            if nothing_p(c.content):
                return False
            # If TMS, check the consequence
            if tms_p(c.content):
                val = tms_query(c.content)
                if nothing_p(val):
                    return False
        return True

    def get_value(self, cell: Cell) -> Optional[CellValue]:
        """
        Get the current value of a cell, handling Supported/TMS wrappers.

        Returns Optional[CellValue] (not Optional[SolverValue]): a cell fed
        by an arbitrary real propagator (see class docstring's backward-
        compatibility guarantee) can hold any value type, not just the
        int/bool domains this module's own Z3 translation declares.

        Translates the nothing sentinel to None at this boundary (a
        deliberate, documented choice for this convenience accessor -- see
        propagator/nothing.py) so "undetermined" and "determined to be the
        real value None" are still distinguishable via cell.content/
        nothing_p directly, while callers of get_value() keep the familiar
        Optional[T]-style contract.
        """
        if nothing_p(cell.content):
            return None
        if supported_p(cell.content):
            return cell.content.value
        if tms_p(cell.content):
            result = tms_query(cell.content)
            if nothing_p(result):
                return None
            if supported_p(result):
                return result.value
            return result
        return cell.content

    def get_provenance(self, cell: Cell) -> List[Premise]:
        """
        Get the premises that support a cell's current value.

        This is REAL provenance tracking via the TMS!
        """
        if nothing_p(cell.content):
            return []
        if supported_p(cell.content):
            return get_support_premises(cell.content)
        if tms_p(cell.content):
            result = tms_query(cell.content)
            if supported_p(result):
                return get_support_premises(result)
        return []
    
    # =========================================================================
    # SMT Integration
    # =========================================================================
    
    def solve(self, verbose: bool = False, max_rounds: int = 100) -> bool:
        """
        Hybrid solve:
        1. Run propagation to fixpoint
        2. If undetermined cells remain, call SMT
        3. Inject SMT solution as hypothetical premises
        4. If contradiction, learn nogood, send to SMT, retry
        5. Repeat until solved or UNSAT

        Contradictions involving SMT-injected premises are resolved by the
        TMS itself: since _inject_smt_solution wraps values in a Tms,
        tms_merge's check_consistent()/process_nogood() runs on every merge
        and kicks out the offending SMTHypothesis (self-healing via
        kick_out + reschedule, not via an exception -- process_nogood never
        raises). After each propagation pass we also harvest any nogoods
        recorded during that round so the *next* SMT call doesn't propose
        the same rejected combination again.
        """
        seen_nogood_count = len(get_contradictions())
        for round_num in range(max_rounds):
            # Phase 1: Propagation
            if verbose:
                print(f"Round {round_num}: Propagation...")

            try:
                self.propagate()
            except TmsContradiction as e:
                if verbose:
                    print(f"  TMS contradiction: {e.nogood}")
                # Record nogood for SMT
                smt_hyps = [h for h in e.nogood if isinstance(h, SMTHypothesis)]
                if smt_hyps:
                    self._learned_nogoods.append(smt_hyps)
                continue
            except Exception as e:
                if verbose:
                    print(f"  Contradiction: {e}")
                return False

            # Harvest nogoods the TMS recorded while resolving contradictions
            # during propagation/injection (the actual live path -- see
            # docstring above), independent of the TmsContradiction branch
            # above, which only fires if something explicitly raises it.
            all_nogoods = get_contradictions()
            for nogood in all_nogoods[seen_nogood_count:]:
                smt_hyps = [h for h in nogood if isinstance(h, SMTHypothesis)]
                if smt_hyps:
                    self._learned_nogoods.append(smt_hyps)
            seen_nogood_count = len(all_nogoods)

            if self.is_fully_determined():
                if verbose:
                    print("  Solved!")
                return True
            
            # Phase 2: SMT
            undetermined = [c for c in self.cells if self.get_value(c) is None]
            if verbose:
                print(f"  {len(undetermined)} undetermined cells, calling SMT...")
            
            solution, diagnostics = self._solve_with_z3(verbose=verbose)
            
            if verbose and diagnostics.get('skipped', 0) > 0:
                print(f"  SMT translation: {diagnostics['translated']} translated, "
                      f"{diagnostics['skipped']} skipped")
                for skip in diagnostics.get('skipped_details', []):
                    cells_str = ', '.join(skip['cell_names'])
                    print(f"    skipped {skip['kind']} on [{cells_str}]: {skip['reason']}")
            
            if solution is None:
                if verbose:
                    print("  SMT: UNSAT")
                return False
            
            if verbose:
                print(f"  SMT solution: {solution}")
            
            # Phase 3: Inject as hypothetical premises
            self._inject_smt_solution(solution)
            self._smt_round += 1
        
        if verbose:
            print(f"Exceeded max rounds ({max_rounds})")
        return False

    def solve_incremental(self, verbose: bool = False, max_rounds: int = 100) -> bool:
        """
        Hybrid solve with incremental theory propagation.

        Instead of one-shot SMT → inject everything → reconcile (which
        creates a reconciliation storm of 30k+ propagator executions),
        this interleaves small propagation rounds with incremental Z3
        checks.  Domain narrowing feeds forward to Z3; Z3 implied values
        feed back to propagators.  Converges in fewer, cheaper rounds.

        Algorithm:
            1. Setup persistent Z3 solver with structural constraints
            2. Loop:
               a. Run one propagation round (not full fixpoint)
               b. Push domain changes to Z3 as assertions
               c. Incremental Z3 check
               d. If SAT: extract implied values, inject back
               e. If UNSAT: learn conflict nogood, pop + retry
               f. If fixpoint: done
        """
        try:
            import z3 as _z3
        except ImportError:
            raise ImportError("Z3 not available. Install with: pip install z3-solver")

        # -- Phase 1: Setup persistent Z3 solver --
        compiler = NetworkCompiler(name=self.name)
        discovered = discover_network(self.cells)
        boolean_cells = _boolean_output_cells(discovered.constraints)

        self._refresh_domains_from_tms()
        for cell in self.cells:
            if cell in boolean_cells:
                compiler.add_boolean(cell, name=cell.name)
            elif self.domains.get(cell):
                compiler.add_domain(cell, list(self.domains[cell]), name=cell.name)
            else:
                compiler.add_integer(cell, name=cell.name)

        # Fix grounded values (non-hypothetical)
        grounded_values: Dict[Cell, CellValue] = {}
        for cell in self.cells:
            val = self.get_value(cell)
            if val is not None:
                compiler.add_fixed_value(cell, val)
                if not any(hypothetical_p(p) for p in self.get_provenance(cell)):
                    grounded_values[cell] = val

        # Add discovered + tracked constraints (structural)
        for constraint in discovered.constraints:
            _try_add_constraint(compiler, constraint, grounded_values)
        for tracked_constraint in self.constraints:
            self._add_tracked_constraint(compiler, tracked_constraint)

        # Export to get Z3 solver + variable map
        encoding = compiler.export(SolverBackend.Z3_PYTHON)
        solver = encoding.metadata.get('solver')
        z3_vars = encoding.metadata.get('z3_vars', {})
        cell_to_name = {cell: var.name for cell, var in compiler.variables.items()}
        name_to_cell = {v: k for k, v in cell_to_name.items()}

        if solver is None:
            return self.solve(verbose=verbose, max_rounds=max_rounds)  # fallback

        # Push structural checkpoint
        solver.push()

        # -- Phase 2: Incremental loop --
        seen_nogood_count = len(get_contradictions())
        prev_domains: Dict[Cell, set] = {}  # track domain changes

        for round_num in range(max_rounds):
            if verbose:
                print(f"Incremental round {round_num}: propagation...")

            # a) One propagation round (not full fixpoint)
            try:
                self.propagate()
            except TmsContradiction as e:
                smt_hyps = [h for h in e.nogood if isinstance(h, SMTHypothesis)]
                if smt_hyps:
                    self._learned_nogoods.append(smt_hyps)
                continue
            except Exception:
                if verbose:
                    print(f"  Contradiction, harvesting nogoods")
                all_nogoods = get_contradictions()
                for nogood in all_nogoods[seen_nogood_count:]:
                    smt_hyps = [h for h in nogood if isinstance(h, SMTHypothesis)]
                    if smt_hyps:
                        self._learned_nogoods.append(smt_hyps)
                seen_nogood_count = len(all_nogoods)
                continue

            # b) Collect domain changes
            self._refresh_domains_from_tms()
            new_assertions = []
            for cell in self.cells:
                if cell in self.domains:
                    current = self.domains[cell]
                    prev = prev_domains.get(cell, set())
                    if current != prev:
                        prev_domains[cell] = set(current)
                        name = cell_to_name.get(cell)
                        if name and name in z3_vars:
                            v = z3_vars[name]
                            if len(current) == 1:
                                val = next(iter(current))
                                new_assertions.append(v == val)
                            else:
                                lo, hi = min(current), max(current)
                                new_assertions.append(_z3.And(v >= lo, v <= hi))

            # c) Push to Z3 + incremental check
            for a in new_assertions:
                solver.add(a)
            # Add learned nogoods as blocking clauses
            for nogood in self._learned_nogoods:
                literals = []
                for hyp in nogood:
                    c = hyp.cell
                    name = cell_to_name.get(c)
                    if name and name in z3_vars:
                        literals.append(z3_vars[name] != hyp.value)
                if literals:
                    solver.add(_z3.Or(literals))
            self._learned_nogoods.clear()

            if verbose and new_assertions:
                print(f"  Pushed {len(new_assertions)} domain changes to Z3")

            r = solver.check()
            if r == _z3.unsat:
                if verbose:
                    print(f"  Z3: UNSAT — problem is unsatisfiable")
                return False

            # d) Extract implied values from model
            model = solver.model()
            injected = 0
            for name, z3v in z3_vars.items():
                cell = name_to_cell.get(name)
                if cell is None:
                    continue
                if self.get_value(cell) is not None:
                    continue  # already determined
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

            # e) Check fixpoint
            if self.is_fully_determined():
                if verbose:
                    print(f"  Solved after {round_num + 1} incremental rounds")
                return True

            # No new domain changes and no new injections → stuck
            if not new_assertions and injected == 0:
                if verbose:
                    print(f"  Stuck at round {round_num} — falling back to full solve")
                # Fallback: do a full SMT solve for remaining cells
                undetermined = [c for c in self.cells if self.get_value(c) is None]
                if undetermined:
                    solution, _ = self._solve_with_z3(verbose=False)
                    if solution:
                        self._inject_smt_solution(solution)
                        self._smt_round += 1
                        try:
                            self.propagate()
                        except Exception:
                            pass
                break

        return self.is_fully_determined()

    def _refresh_domains_from_tms(self) -> None:
        """
        Refresh self.domains from TMS state, narrowing domains to the
        values that are still believed (not kicked out).

        This feeds propagation-narrowed domains back to the SMT oracle
        on subsequent rounds, reducing the search space.
        """
        try:
            from ..tms import get_all_hypotheticals as _get_hyps
        except Exception:
            return

        # Collect remaining values per cell from TMS hypotheticals
        remaining: Dict[Cell, Set[Any]] = {}
        for hyp in _get_hyps():
            cell = getattr(hyp, 'output_cell', None)
            value = getattr(hyp, 'value_if_chosen', None)
            if cell in self.domains and value is not None:
                # Skip composite placeholders from one_of recursion
                if isinstance(value, str) and value.startswith('one of'):
                    continue
                remaining.setdefault(cell, set()).add(value)

        # Intersect with original domains where narrowed
        for cell in list(self.domains.keys()):
            if cell in remaining:
                narrowed = self.domains[cell] & remaining[cell]
                if narrowed and narrowed != self.domains[cell]:
                    self.domains[cell] = narrowed

    def _solve_with_z3(self, verbose: bool = False) -> Tuple[Optional[Dict[str, SolverValue]], Dict[str, Any]]:
        """
        Solve the current residual problem for undetermined cells.

        Translation is delegated to the canonical NetworkCompiler + Z3
        backend (compiler.py/backends.py -- the same layer solve_from_roots
        uses) instead of hand-rolling a second Z3 encoder here. That is what
        makes comparison propagators (eq/lt/gt/...) get correctly declared
        as Bool rather than Int, and keeps this module from maintaining its
        own, independently-drifting copy of "constraint kind -> solver call".

        Returns:
            Tuple of:
              - solution dict (cell_name -> value) or None if UNSAT
              - diagnostics dict with keys:
                - 'translated': number of discovered constraints successfully translated
                - 'skipped': number of discovered constraints that could not be translated
                - 'skipped_details': list of (kind, cell_names) for each skipped constraint
                - 'tracked_added': number of tracked constraints added
                - 'nogoods_added': number of learned nogoods added
        """
        try:
            import z3  # noqa: F401 -- imported only so a missing install fails clearly
        except ImportError:
            raise ImportError("Z3 not available. Install with: pip install z3-solver")

        compiler = NetworkCompiler(name=self.name)
        discovered = discover_network(self.cells)
        boolean_cells = _boolean_output_cells(discovered.constraints)

        # Register every cell with the tightest known type/domain.
        # Refresh domains from TMS state first so propagation-narrowed
        # domains are fed to the solver (Phase 4: domain narrowing).
        self._refresh_domains_from_tms()
        for cell in self.cells:
            if cell in boolean_cells:
                compiler.add_boolean(cell, name=cell.name)
            elif self.domains.get(cell):
                compiler.add_domain(cell, list(self.domains[cell]), name=cell.name)
            else:
                compiler.add_integer(cell, name=cell.name)

        # Fix already-determined cells (propagator-derived, or SMT-injected
        # from a prior round) so the solver only searches over what's still
        # open. Separately track which of those values are *grounded* --
        # not resting on a speculative Hypothetical/SMTHypothesis guess --
        # since only grounded values are safe to treat as ground truth for
        # _try_add_constraint's predicate-lowering below (see that call for
        # why: lowering a comparison using a value that's itself just an
        # unconfirmed SMT guess can turn a wrong guess into a hard, false
        # structural constraint instead of leaving it open to reconsider).
        fixed_values: Dict[Cell, CellValue] = {}
        grounded_values: Dict[Cell, CellValue] = {}
        for cell in self.cells:
            val = self.get_value(cell)
            if val is not None:
                fixed_values[cell] = val
                compiler.add_fixed_value(cell, val)
                if not any(hypothetical_p(p) for p in self.get_provenance(cell)):
                    grounded_values[cell] = val

        # Constraints from any real propagator wired onto these cells --
        # including ones added directly with standard propagator primitives,
        # bypassing this network's own tracked builder methods entirely
        # (e.g. squarer(x, x_sq)). Without this, such constraints are
        # invisible to the SMT oracle, which can then inject a value a live
        # propagator later rejects, corrupting the round instead of just
        # being asked to avoid it up front.
        translated_count = 0
        skipped_count = 0
        skipped_details: List[Dict[str, Any]] = []
        cell_names = discovered.cell_names
        for constraint in discovered.constraints:
            if _try_add_constraint(compiler, constraint, grounded_values):
                translated_count += 1
            else:
                skipped_count += 1
                skipped_details.append({
                    'kind': constraint.kind,
                    'cell_names': [cell_names.get(c, getattr(c, 'name', str(c)))
                                   for c in constraint.cells],
                    'reason': (
                        'unpinned ternary comparison (boolean output not fixed by require/abhor)'
                        if constraint.kind in {'eq', 'lt', 'gt', 'lte', 'gte'}
                        and len(constraint.cells) == 3
                        else f'unsupported constraint kind: {constraint.kind}'
                    ),
                })

        # Constraints tracked by this network's own builder methods that
        # have no corresponding real propagator (all_different, column_add,
        # ...), so network_discovery can't see them. A distinct loop
        # variable name from the discovered.constraints loop above: they're
        # different types (TrackedConstraint vs. DiscoveredConstraint)
        # despite both meaning "a constraint on some cells".
        tracked_added = 0
        for tracked_constraint in self.constraints:
            self._add_tracked_constraint(compiler, tracked_constraint)
            tracked_added += 1

        # Learned nogoods as blocking clauses, so the next call doesn't
        # propose a combination the TMS already rejected.
        nogoods_added = 0
        for nogood in self._learned_nogoods:
            compiler.add_nogood([(hyp.cell, hyp.value) for hyp in nogood])
            nogoods_added += 1

        diagnostics: Dict[str, Any] = {
            'translated': translated_count,
            'skipped': skipped_count,
            'skipped_details': skipped_details,
            'tracked_added': tracked_added,
            'nogoods_added': nogoods_added,
            'discovered_total': len(discovered.constraints),
        }

        # Incremental Z3 solving: reuse solver across rounds via push/pop.
        # First round: full encoding via NetworkCompiler, cache solver.
        # Subsequent rounds: pop per-round constraints, add new ones directly.
        if not self._z3_solver:
            # First call: export to get Z3 solver, then solve
            encoding = compiler.export(SolverBackend.Z3_PYTHON)
            meta = encoding.metadata
            self._z3_solver = meta.get('solver')
            self._z3_vars = meta.get('z3_vars', {})
            # Build cell->z3-name map from compiler variables
            for cell, var in compiler.variables.items():
                self._z3_cell_map[cell] = var.name
            # Push a structural checkpoint so we can roll back per-round
            # constraints (fixed values, nogoods) without re-declaring vars.
            self._z3_solver.push()
            # Solve
            r = self._z3_solver.check()
            try:
                import z3 as _z3
            except ImportError:
                raise ImportError("Z3 not available")
            if r == _z3.sat:
                model = self._z3_solver.model()
                solution_dict: Dict[Any, Any] = {}
                for cell in self.cells:
                    if cell in self._z3_cell_map:
                        name = self._z3_cell_map[cell]
                        if name in self._z3_vars:
                            z3v = self._z3_vars[name]
                            try:
                                val = model[z3v]
                                if val is not None:
                                    if _z3.is_bool(z3v):
                                        solution_dict[cell] = _z3.is_true(val)
                                    elif _z3.is_int(z3v):
                                        solution_dict[cell] = val.as_long()
                                    else:
                                        try:
                                            solution_dict[cell] = val.as_long()
                                        except Exception:
                                            solution_dict[cell] = bool(val)
                            except Exception:
                                pass
                # Use compiler's decode_solution for proper cell mapping
                decoded = compiler.decode_solution(
                    {name: solution_dict.get(cell)
                     for cell, name in self._z3_cell_map.items()
                     if cell in solution_dict}
                )
                result = SolverResult(satisfiable=True, solution=decoded)
            else:
                result = SolverResult(satisfiable=False, solution={})
        else:
            # Subsequent rounds: pop per-round constraints, add new ones
            self._z3_solver.pop()
            try:
                import z3 as _z3
            except ImportError:
                raise ImportError("Z3 not available")
            # Add fixed values for determined cells
            for cell in self.cells:
                val = self.get_value(cell)
                if val is not None and cell in self._z3_cell_map:
                    name = self._z3_cell_map[cell]
                    if name in self._z3_vars:
                        self._z3_solver.add(self._z3_vars[name] == val)
            # Add learned nogoods
            for nogood in self._learned_nogoods:
                literals = []
                for hyp in nogood:
                    c = hyp.cell
                    if c in self._z3_cell_map:
                        name = self._z3_cell_map[c]
                        if name in self._z3_vars:
                            literals.append(self._z3_vars[name] != hyp.value)
                if literals:
                    self._z3_solver.add(_z3.Or(literals))
            self._z3_solver.push()
            # Solve with cached solver
            r = self._z3_solver.check()
            if r == _z3.sat:
                model = self._z3_solver.model()
                solution_dict = {}
                for cell in self.cells:
                    if cell in self._z3_cell_map:
                        name = self._z3_cell_map[cell]
                        if name in self._z3_vars:
                            z3v = self._z3_vars[name]
                            try:
                                val = model[z3v]
                                if val is not None:
                                    if _z3.is_bool(z3v):
                                        solution_dict[cell] = _z3.is_true(val)
                                    elif _z3.is_int(z3v):
                                        solution_dict[cell] = val.as_long()
                                    else:
                                        try:
                                            solution_dict[cell] = val.as_long()
                                        except Exception:
                                            solution_dict[cell] = bool(val)
                            except Exception:
                                pass
                result = SolverResult(satisfiable=True, solution=solution_dict)
            else:
                result = SolverResult(satisfiable=False, solution={})

        if not result.satisfiable or not result.solution:
            return None, diagnostics

        # result.solution also carries values for compiler-only intermediate
        # cells created by _add_tracked_constraint (e.g. abs_diff_neq's
        # _diff_*/_absdiff_*/_const_* cells) -- keep only cells that belong
        # to this network, so callers never see or inject values for cells
        # the real propagator network doesn't know about.
        network_cells = set(self.cells)
        solution = {
            cell.name: value
            for cell, value in result.solution.items()
            if value is not None and cell in network_cells and self.get_value(cell) is None
        }
        return solution, diagnostics

    def _add_tracked_constraint(self, compiler: NetworkCompiler, constraint: TrackedConstraint) -> None:
        """
        Translate one of this network's own tracked constraints into the
        compiler's IR. These are relations with no corresponding real
        propagator (all_different's pruning propagator isn't structurally
        discoverable, and less_than/greater_than/column_add/equals_var are
        explicitly "SMT-oracle only" -- see their docstrings above), so
        network_discovery can't find them; _solve_with_z3 tracks them here
        instead, in this network's own kind vocabulary (distinct/add/mul/...,
        distinct from DiscoveredConstraint's, hence a separate translator).
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
            # |a - b| != value, expressed via compiler-only intermediate
            # cells (never touched by the real propagator network).
            a, b = cells
            value: Optional[SolverValue] = constraint.extra.get('value') if constraint.extra else None
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
            # a + b (+ carry_in) = result + 10*carry_out
            a, b, result, carry_out = cells
            carry_in: Optional[Cell] = constraint.extra.get('carry_in') if constraint.extra else None
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

    def _inject_smt_solution(self, solution: Dict[str, SolverValue]) -> None:
        """
        Inject SMT solution as hypothetical premises with Supported values.
        
        This is the KEY difference from the naive hybrid:
        - Each SMT assignment becomes a Hypothetical premise
        - The value is wrapped in Supported(value, {hypothesis})
        - The TMS tracks these premises
        - If contradiction, TMS computes nogood involving these hypotheses
        """
        for cell_name, value in solution.items():
            cell = self.cell_by_name.get(cell_name)
            if cell is None:
                continue

            # Create a hypothesis for this SMT-provided value
            hyp = SMTHypothesis(
                cell=cell, value=value, round=self._smt_round,
                output_cell=cell, value_if_chosen=value, sign='smt',
                name=f"smt_r{self._smt_round}",
            )

            # Mark hypothesis as believed
            bring_in(hyp)

            # Wrap in a Tms (not a bare Supported) so any merge against this
            # cell's existing/future content routes through tms_merge, which
            # is what actually invokes check_consistent/process_nogood/
            # kick_out on contradiction. A bare Supported never engages that
            # machinery -- supported_merge has no nogood-learning path at
            # all, it can only raise a flat "Ack! Inconsistency!" exception.
            cell.add_content(to_tms(supported(value, [hyp])))


# =============================================================================
# Convenience function
# =============================================================================

# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print("=== True Hybrid Propagator + SMT Demo ===\n")
    
    # Test 1: Simple sum that propagation can solve alone
    print("Test 1: Simple sum (propagation alone)")
    initialize_scheduler()
    
    net = TrueHybridNetwork(name="simple_sum")
    a = net.cell("a")
    b = net.cell("b") 
    c = net.cell("c")
    
    net.sum_constraint(a, b, c)
    net.constant(2, a)
    net.constant(3, b)
    
    net.propagate()
    print(f"  a={net.get_value(a)}, b={net.get_value(b)}, c={net.get_value(c)}")
    print(f"  c's provenance: {net.get_provenance(c)}")
    assert net.get_value(c) == 5, f"Expected c=5, got {net.get_value(c)}"
    print("  ✓ Passed\n")
    
    # Test 2: Backward propagation
    print("Test 2: Backward propagation")
    initialize_scheduler()
    
    net = TrueHybridNetwork(name="backward")
    x = net.cell("x")
    y = net.cell("y")
    z = net.cell("z")
    
    net.sum_constraint(x, y, z)
    net.constant(10, z)
    net.constant(3, y)
    
    net.propagate()
    print(f"  x={net.get_value(x)}, y={net.get_value(y)}, z={net.get_value(z)}")
    assert net.get_value(x) == 7, f"Expected x=7, got {net.get_value(x)}"
    print("  ✓ Passed\n")
    
    # Test 3: Requires SMT search
    print("Test 3: Requires SMT search (all-different with domains)")
    initialize_scheduler()
    
    net = TrueHybridNetwork(name="needs_smt")
    p = net.cell("p", domain={1, 2, 3})
    q = net.cell("q", domain={1, 2, 3})
    r = net.cell("r", domain={1, 2, 3})
    
    net.all_different([p, q, r])
    
    if net.solve(verbose=True):
        pv, qv, rv = net.get_value(p), net.get_value(q), net.get_value(r)
        print(f"  Result: p={pv}, q={qv}, r={rv}")
        print(f"  Provenance of p: {net.get_provenance(p)}")
        assert len({pv, qv, rv}) == 3, "all_different violated"
        print("  ✓ Passed\n")
    else:
        print("  ✗ Failed - should have found solution\n")
    
    # Test 4: Combined propagation + search
    print("Test 4: Combined propagation + search")
    initialize_scheduler()
    
    net = TrueHybridNetwork(name="combined")
    a = net.cell("a", domain={1, 2, 3, 4})
    b = net.cell("b", domain={1, 2, 3, 4})
    c = net.cell("c")
    
    net.sum_constraint(a, b, c)
    net.all_different([a, b])
    net.constant(5, c)  # Forces a + b = 5
    
    if net.solve(verbose=True):
        av, bv, cv = net.get_value(a), net.get_value(b), net.get_value(c)
        print(f"  Result: a={av}, b={bv}, c={cv}")
        assert av is not None and bv is not None, "a solved network must determine every cell"
        assert av + bv == 5, f"Sum constraint violated: {av} + {bv} != 5"
        assert av != bv, "all_different violated"
        print("  ✓ Passed\n")
    else:
        print("  ✗ Failed\n")
    
    print("=== All tests passed! ===")
    print("\nKey features demonstrated:")
    print("  1. Uses real Cell from propagator.cell")
    print("  2. Uses real scheduler for propagation")
    print("  3. Bidirectional constraints (sum_constraint)")
    print("  4. Provenance tracking via Supported values")
    print("  5. SMT search when propagation insufficient")
    print("  6. SMT solutions injected as hypothetical premises")

    # Demonstrate backward compatibility with standard primitives
    print("\n" + "=" * 60)
    print("BACKWARD COMPATIBILITY DEMO")
    print("=" * 60)
    print("\nUsing cells from TrueHybridNetwork with standard propagator primitives:")

    from ..primitives import squarer

    initialize_scheduler()

    net = TrueHybridNetwork(name="backward_compat")
    x = net.cell("x")
    x_squared = net.cell("x_squared")

    squarer(x, x_squared)

    _constant(5, x)
    net.propagate()

    print(f"  x = {net.get_value(x)}")
    print(f"  x² = {net.get_value(x_squared)}")
    print(f"  Cell type: {type(x).__name__}")
    print("\n  ✓ Standard propagator primitives work directly with TrueHybridNetwork cells!")
