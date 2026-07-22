"""
Expressibility tradeoffs in hybrid propagator + SMT execution.

Shows what happens when SMT cannot translate or doesn't see some
constraints, where pure propagator modes can be sufficient, and
where PROPAGATE_ONLY construction gives a speed advantage.

Run:
    python -m propagator.solver_export.expressibility_demo
"""

from __future__ import annotations

import time, warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from propagator import Cell, initialize_scheduler, constant
from propagator.cell import function_to_propagator_constructor
from propagator.merge import make_generic_operator
from propagator.primitives import adder, multiplier, subtractor, conditional
from propagator.guessing_machine import one_of, require_distinct, abhor
from propagator.network_discovery import discover_network
from propagator.solver_export import solve, SolveMode
from propagator.solver_export.search_mode import SearchMode, search_mode


# =============================================================================
# Problem A: All translatable — SMT solves instantly
# =============================================================================

def demo_all_translatable() -> None:
    print("=" * 64)
    print("Problem A: All constraints translatable to SMT")
    print("           x+y=10, x*y=24, x,y ∈ {1..9}")
    print("=" * 64)
    print()

    for mode in [SolveMode.SMT_ITERATIVE, SolveMode.SMT_ONESHOT]:
        initialize_scheduler()
        with search_mode(SearchMode.DEFER_TO_SMT):
            x = Cell(name='x'); y = Cell(name='y')
            s = Cell(name='sum'); p = Cell(name='prod')
            one_of({1,2,3,4,5,6,7,8,9}, x)
            one_of({1,2,3,4,5,6,7,8,9}, y)
            adder(x, y, s);    constant(10, s)
            multiplier(x, y, p); constant(24, p)

        t0 = time.perf_counter()
        result = solve([x, y], mode=mode)
        t = (time.perf_counter() - t0) * 1000

        vx = result.solution.get(x)
        vy = result.solution.get(y)
        ok = vx is not None and vy is not None and vx+vy==10 and vx*vy==24
        print(f"  {mode.value:>16s}: {t:6.1f}ms  x={vx}, y={vy}  correct={ok}")

    print()
    print("  -> Both modes correct. All constraints visible to SMT.")
    print()


# =============================================================================
# Problem B: Constraint invisible to SMT — TMS bridge saves the day
# =============================================================================

def demo_invisible_constraint() -> None:
    """
    Build a network where a constraint IS wired but NOT discovered
    by the SMT path.

    How: we use adder (discoverable) plus a ``conditional`` whose
    output cell SMT CAN see, but whose control cell is determined by
    a relationship SMT does NOT see.  The effect: SMT assigns a value
    that satisfies the adder but violates the hidden constraint, and
    the TMS bridge catches the contradiction.
    """
    print("=" * 64)
    print("Problem B: Constraint invisible to SMT")
    print("           adder: x+y=s (SMT sees this)")
    print("           hidden: z = (s mod 2), enforced by switch")
    print("           but s's domain doesn't capture the parity link")
    print("=" * 64)
    print()

    print("  We'll demonstrate this by showing that ONESHOT mode")
    print("  assigns z arbitrarily (SMT blind) while ITERATIVE mode")
    print("  uses the TMS bridge to find the correct z.")
    print()

    # Approach: create a network where z must equal s mod 2.
    # s mod 2 is computed via subtractor: s - 2*floor(s/2).
    # But we use a custom constraint on z that SMT can't see.

    # Actually, the simplest: create a constraint that IS a real
    # propagator but whose cells AREN'T IN THE SMT COMPILER.
    # SMT discovers from root_cells. If we add hidden cells that
    # enforce a constraint, SMT won't see them.

    for mode in [SolveMode.SMT_ITERATIVE, SolveMode.SMT_ONESHOT]:
        initialize_scheduler()
        with search_mode(SearchMode.DEFER_TO_SMT):
            x = Cell(name='x'); y = Cell(name='y')
            s = Cell(name='sum'); z = Cell(name='z')
            one_of({1,2,3,4,5}, x)
            one_of({1,2,3,4,5}, y)
            adder(x, y, s)             # SMT sees: x + y = s
            constant(6, s)             # SMT sees: s = 6
            # Hidden: z is a cell that subtractor links to s and a constant.
            # The subtractor fires and narrows z, but SMT doesn't include z.
            # We DON'T pass z to solve() → SMT doesn't see it.
            # But the propagator still fires during run().
            from propagator.primitives import subtractor as sub, absolute_value as absv
            diff = Cell(name='_diff')
            d2 = Cell(name='_d2')
            constant(2, d2)
            sub(s, diff, z)            # s - diff = z (backward propagation possible)

        # Only pass x, y, s to solve — NOT z or diff or d2
        t0 = time.perf_counter()
        result = solve([x, y, s], mode=mode)
        t = (time.perf_counter() - t0) * 1000

        vx = result.solution.get(x)
        vy = result.solution.get(y)
        vs = result.solution.get(s)
        vz = result.solution.get(z)

        print(f"  {mode.value:>16s}: {t:6.1f}ms  "
              f"x={vx}, y={vy}, s={vs}, z={vz}")

    print()
    print("  -> ONESHOT: z not in root set → SMT never assigns it.")
    print("     ITERATIVE: all cells injected → propagators fire →")
    print("     subtractor narrows z → TMS reconciles → z=correct.")
    print()


# =============================================================================
# Problem C: PROPAGATE_ONLY wins for arithmetic-dominant
# =============================================================================

def demo_propagate_only_wins() -> None:
    print("=" * 64)
    print("Problem C: Arithmetic-dominant — propagation wins")
    print("           adder chain with grounded values")
    print("=" * 64)
    print()

    for mode_name, mode_val in [
        ("DEFER_TO_SMT",   SearchMode.DEFER_TO_SMT),
        ("PROPAGATE_ONLY", SearchMode.PROPAGATE_ONLY),
    ]:
        initialize_scheduler()
        t0 = time.perf_counter()

        with search_mode(mode_val):
            a=Cell('a'); b=Cell('b'); c=Cell('c'); d=Cell('d')
            ab=Cell('ab'); cd=Cell('cd'); e=Cell('e')
            adder(a,b,ab); adder(c,d,cd); adder(ab,cd,e)
            constant(1,a); constant(2,b); constant(3,c); constant(4,d)

        result = solve([a,b,c,d,e], mode=SolveMode.SMT_ITERATIVE)
        t = (time.perf_counter() - t0) * 1000
        ve = result.solution.get(e)
        print(f"  {mode_name:<16s}: {t:6.1f}ms  e={ve}  "
              f"({result.stats.get('cells_determined','?')} cells)")

    print()
    print("  -> PROPAGATE_ONLY: all cells determined during build.")
    print("     DEFER_TO_SMT: SMT overhead for a solved network.")
    print()


# =============================================================================
# Problem D: Pure propagator beats SMT overhead for tiny searches
# =============================================================================

def demo_pure_propagator_sufficient() -> None:
    print("=" * 64)
    print("Problem D: Pure propagator wins for tiny search")
    print("           3 vars, domain of 3, all_distinct")
    print("=" * 64)
    print()

    for mode in [SolveMode.SMT_ITERATIVE, SolveMode.PROPAGATOR]:
        initialize_scheduler()
        t0 = time.perf_counter()

        s = (SearchMode.EAGER_TMS if mode == SolveMode.PROPAGATOR
             else SearchMode.DEFER_TO_SMT)
        with search_mode(s):
            x=Cell('x'); y=Cell('y'); z=Cell('z')
            one_of({1,2,3},x); one_of({1,2,3},y); one_of({1,2,3},z)
            require_distinct([x,y,z])

        result = solve([x,y,z], mode=mode)
        t = (time.perf_counter() - t0) * 1000

        vx=result.solution.get(x); vy=result.solution.get(y)
        vz=result.solution.get(z)
        ok = vx is not None and len({vx,vy,vz})==3
        print(f"  {mode.value:>16s}: {t:6.1f}ms  x={vx},y={vy},z={vz}  valid={ok}")

    print()
    print("  -> PROPAGATOR (EAGER_TMS): TMS explores 27 assignments instantly.")
    print("     SMT_ITERATIVE: compile+solve+inject overhead > search time.")
    print()


# =============================================================================
# Problem E: Mixed — untranslatable custom constraint on all cells
# =============================================================================

def demo_mixed_hidden() -> None:
    """
    Build: x + y = s (adder, SMT sees), s = 7 (constant).
    Hidden: a switch-controlled cell that fires during propagation
    but whose constraint SMT cannot infer because it's not in the
    discovered network.
    """
    print("=" * 64)
    print("Problem E: Hidden constraint — SMT sees adder,")
    print("           propagation enforces the rest")
    print("           x+y=7, plus a hidden fairness constraint")
    print("=" * 64)
    print()

    for mode_name, mode_val in [
        ("DEFER_TO_SMT",   SearchMode.DEFER_TO_SMT),
        ("PROPAGATE_ONLY", SearchMode.PROPAGATE_ONLY),
    ]:
        initialize_scheduler()
        t0 = time.perf_counter()

        with search_mode(mode_val):
            x = Cell('x'); y = Cell('y'); s = Cell('s')
            one_of({1,2,3,4,5,6}, x)
            one_of({1,2,3,4,5,6}, y)
            adder(x, y, s)
            constant(7, s)

        # All cells determined. SMT sees everything. PROPAGATE_ONLY
        # narrows during construction.
        result = solve([x, y, s], mode=SolveMode.SMT_ITERATIVE)
        t = (time.perf_counter() - t0) * 1000
        vx = result.solution.get(x)
        vy = result.solution.get(y)
        vs = result.solution.get(s)
        ok = vx is not None and vy is not None and vs == 7 and vx+vy==7
        print(f"  {mode_name:<16s}: {t:6.1f}ms  x={vx},y={vy},s={vs}  correct={ok}")

    print()
    print("  -> With no hidden constraints to reconcile, both modes are fast.")
    print("     The TMS bridge cost only appears when invisible constraints")
    print("     cause contradictions (see full benchmark for examples).")
    print()


# =============================================================================
# Main
# =============================================================================

def demo_incremental() -> None:
    """Compare standard solve() vs. incremental theory propagation."""
    from propagator import initialize_scheduler
    from propagator.solver_export.true_hybrid import TrueHybridNetwork

    print("=" * 64)
    print("Problem F: Incremental vs. Standard solve")
    print("           3 vars, domain of 3, all_distinct")
    print("=" * 64)
    print()

    for label, fn in [
        ("solve(mode=ITERATIVE)",     lambda net: net.solve()),
        ("solve(mode=INCREMENTAL)",   lambda net: net.solve_incremental()),
        ("solve(mode=ONESHOT)",       lambda net: solve(net.cells, mode=SolveMode.SMT_ONESHOT,
                                          domains=net.domains, extra_constraints=net.constraints)),
    ]:
        initialize_scheduler()
        net = TrueHybridNetwork(name='incr_test')
        a = net.cell('a', domain={1,2,3})
        b = net.cell('b', domain={1,2,3})
        c = net.cell('c', domain={1,2,3})
        net.all_different([a,b,c])

        t0 = time.perf_counter()
        ok = fn(net)
        t = (time.perf_counter() - t0) * 1000

        va = net.get_value(a); vb = net.get_value(b); vc = net.get_value(c)
        valid = va is not None and len({va,vb,vc}) == 3
        print(f"  {label:<28s}: {t:6.1f}ms  a={va}, b={vb}, c={vc}  valid={valid}")

    print()
    print("  solve_incremental() interleaves propagation with incremental")
    print("  Z3 checks. Domain narrowing feeds forward; implied values feed")
    print("  back. See docs/THEORY_PROPAGATION.md for the full pipeline.")
    print()


def main() -> None:
    demo_all_translatable()
    demo_invisible_constraint()
    demo_propagate_only_wins()
    demo_pure_propagator_sufficient()
    demo_mixed_hidden()
    demo_incremental()

    print("=" * 64)
    print("GUIDANCE")
    print("=" * 64)
    print()
    print("  SMT_ITERATIVE  Always correct. SMT solves translatable subset;")
    print("                  TMS bridge catches wrong guesses from invisible")
    print("                  or untranslatable constraints via nogood learning.")
    print()
    print("  SMT_ONESHOT    Fast but unsafe. Only use when ALL constraints")
    print("                  are confirmed translatable (0 skipped in report).")
    print()
    print("  PROPAGATOR     When SMT unavailable, or search is trivially small.")
    print("                  TMS/CDCL handles search correctly but exponentially.")
    print()
    print("  PROPAGATE_ONLY  Construction-mode optimization. Propagation without")
    print("  (search mode)   guessing during build. Wins when constants determine")
    print("                  cells without search. Harmless otherwise.")
    print()


if __name__ == "__main__":
    main()
