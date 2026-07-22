#!/usr/bin/env python3
"""
Demonstration of CDCL-enhanced propagator search.

This demonstrates how Conflict-Driven Clause Learning is integrated
with the propagator TMS to achieve more efficient search.

Run with:
    python3 -m propagator.examples.demo_cdcl
    
For full benchmarks:
    python3 -m propagator.examples.benchmark_cdcl
"""

from propagator import (
    Cell, initialize_scheduler, 
    enable_cdcl, disable_cdcl, cdcl_enabled, cdcl_stats, get_cdcl_engine,
)
from propagator.guessing_machine import one_of, require, abhor, require_distinct
from propagator.tms import (
    get_contradictions,
    get_number_of_calls_to_fail,
    get_contradiction_details,
    hypothetical_p,
    tms_query, tms_p,
)
from propagator.scheduler import run
import time


def setup_multiple_dwelling():
    """
    Set up the Multiple Dwelling puzzle constraints.
    Returns the cells for each person.
    """
    from propagator.primitives import eq, constant, gt, subtractor, absolute_value
    
    # Create cells for each person
    baker = Cell(name='baker')
    cooper = Cell(name='cooper')
    fletcher = Cell(name='fletcher')
    miller = Cell(name='miller')
    smith = Cell(name='smith')
    
    floors = [1, 2, 3, 4, 5]
    
    # Each person lives on one of the floors
    one_of(floors, baker)
    one_of(floors, fletcher)
    one_of(floors, smith)
    one_of(floors, cooper)
    one_of(floors, miller)
    
    # Everyone lives on different floors
    require_distinct([baker, fletcher, smith, cooper, miller])
    
    # Helper cells
    b_eq_5 = Cell('b_eq_5')
    f_eq_5 = Cell('f_eq_5')
    m_gt_c = Cell('m_gt_c')
    fc = Cell('fc')
    five = Cell('five')
    as_f = Cell('as_f')
    af_c = Cell('af_c')
    c_eq_1 = Cell('c_eq_1')
    f_eq_1 = Cell('f_eq_1')
    sf = Cell('sf')
    one = Cell('one')
    s_f = Cell('s_f')
    f_c = Cell('f_c')
    constant(1, one)
    constant(5, five)
    
    # Baker doesn't live on floor 5
    eq(five, baker, b_eq_5)
    abhor(b_eq_5)
    
    # Cooper doesn't live on floor 1
    eq(one, cooper, c_eq_1)
    abhor(c_eq_1)
    
    # Fletcher doesn't live on floors 1 or 5
    eq(five, fletcher, f_eq_5)
    abhor(f_eq_5)
    eq(one, fletcher, f_eq_1)
    abhor(f_eq_1)
    
    # Miller lives above Cooper
    gt(miller, cooper, m_gt_c)
    require(m_gt_c)
    
    # Smith doesn't live adjacent to Fletcher
    subtractor(smith, fletcher, s_f)
    absolute_value(s_f, as_f)
    eq(one, as_f, sf)
    abhor(sf)
    
    # Fletcher doesn't live adjacent to Cooper
    subtractor(fletcher, cooper, f_c)
    absolute_value(f_c, af_c)
    eq(one, af_c, fc)
    abhor(fc)
    
    return baker, cooper, fletcher, miller, smith


def demo_multiple_dwelling_comparison():
    """
    Compare DDB vs CDCL on the Multiple Dwelling puzzle.
    
    The Multiple Dwelling puzzle is a classic constraint satisfaction problem:
    - Baker, Cooper, Fletcher, Miller, and Smith live on different floors (1-5)
    - Baker doesn't live on 5
    - Cooper doesn't live on 1
    - Fletcher doesn't live on 1 or 5
    - Miller lives on a higher floor than Cooper
    - Smith doesn't live adjacent to Fletcher
    - Fletcher doesn't live adjacent to Cooper
    """
    print("=" * 60)
    print("Multiple Dwelling: DDB vs CDCL Comparison")
    print("=" * 60)
    
    # Run with DDB (CDCL disabled)
    print("\n--- Running with DDB (traditional) ---")
    disable_cdcl()
    initialize_scheduler()
    
    start_time = time.time()
    baker, cooper, fletcher, miller, smith = setup_multiple_dwelling()
    run()
    ddb_time = time.time() - start_time
    ddb_conflicts = get_number_of_calls_to_fail()
    
    # Get DDB solution
    ddb_solution = {}
    for name, cell in [('Baker', baker), ('Cooper', cooper), ('Fletcher', fletcher), 
                        ('Miller', miller), ('Smith', smith)]:
        content = cell.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                ddb_solution[name] = result.value if hasattr(result, 'value') else result
    
    print(f"  Time: {ddb_time:.3f}s")
    print(f"  Conflicts: {ddb_conflicts}")
    print(f"  Solution: {ddb_solution}")
    
    # Run with CDCL enabled
    print("\n--- Running with CDCL ---")
    enable_cdcl()
    initialize_scheduler()
    
    start_time = time.time()
    baker, cooper, fletcher, miller, smith = setup_multiple_dwelling()
    run()
    cdcl_time = time.time() - start_time
    cdcl_conflicts = get_number_of_calls_to_fail()
    
    # Get CDCL solution
    cdcl_solution = {}
    for name, cell in [('Baker', baker), ('Cooper', cooper), ('Fletcher', fletcher), 
                        ('Miller', miller), ('Smith', smith)]:
        content = cell.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                cdcl_solution[name] = result.value if hasattr(result, 'value') else result
    
    print(f"  Time: {cdcl_time:.3f}s")
    print(f"  Conflicts: {cdcl_conflicts}")
    print(f"  Solution: {cdcl_solution}")
    
    # CDCL detailed stats
    engine = get_cdcl_engine()
    stats = engine.stats
    
    print(f"\n--- CDCL Statistics ---")
    print(f"  Decisions: {stats.decisions}")
    print(f"  Backjumps: {stats.backjumps}")
    print(f"  Levels saved: {stats.backjump_levels_saved}")
    print(f"  Clauses learned: {stats.learned_clauses}")
    
    # Comparison
    print(f"\n--- Comparison ---")
    if ddb_time > 0:
        speedup = ddb_time / cdcl_time if cdcl_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
    conflict_reduction = ((ddb_conflicts - cdcl_conflicts) / ddb_conflicts * 100) if ddb_conflicts > 0 else 0
    print(f"  Conflict reduction: {conflict_reduction:.1f}%")
    
    # Verify solutions match
    if ddb_solution == cdcl_solution:
        print(f"  ✓ Both methods found the same solution")
    else:
        print(f"  ⚠ Solutions differ (both may be valid)")


def demo_cdcl_concepts():
    """
    Demonstrate core CDCL concepts.
    """
    print("\n" + "=" * 60)
    print("CDCL Core Concepts")
    print("=" * 60)
    
    print("""
CDCL (Conflict-Driven Clause Learning) improves constraint solving via:

1. DECISION LEVEL TRACKING
   ========================
   Track when each decision was made (decision level).
   This enables knowing which decisions are related.

2. 1-UIP LEARNING  
   ===============
   When a conflict occurs, analyze the implication graph to find
   the minimal set of decisions that caused the conflict.
   
   Example:
     Conflict involves: {h1@L1, h2@L2, h3@L3, h4@L4, h5@L5}
     Analysis shows h4, h5 were implied by h3
     1-UIP learns: {h1, h2, h3} - smaller, more general

3. NON-CHRONOLOGICAL BACKJUMPING
   =============================
   Jump back to the source of the conflict, not just one level.
   
   DDB:   L5 conflict → backtrack to L4 → try again
          L4 conflict → backtrack to L3 → try again...
   
   CDCL:  L5 conflict → analyze → h2@L2 caused it
          Jump directly to L2, skip L3, L4!

4. VSIDS HEURISTICS
   =================
   Variables in recent conflicts get higher "activity" scores.
   When choosing what to try next, prefer active variables.
   This focuses search on the "hot" parts of the problem.
""")


def demo_vsids():
    """
    Demonstrate VSIDS activity tracking.
    """
    print("\n" + "=" * 60)
    print("VSIDS Activity-Based Branching")
    print("=" * 60)
    
    print("""
VSIDS (Variable State Independent Decaying Sum):

  on_conflict(nogood):
      for premise in nogood:
          activity[premise] += 1.0
      # Periodically decay all activities by 0.95
  
  choose_branch(options):
      # Prefer options with higher activity
      return max(options, key=lambda p: activity[p])

Why this works:
- Variables in recent conflicts are likely "contentious"
- Focusing on them finds contradictions faster
- Decay ensures we adapt to changing search landscape

In our implementation:
- Activity is tracked per hypothetical premise
- amb_choose uses activities to order branches
- Activities are bumped when nogoods are processed
""")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("CDCL Integration Demo")
    print("=" * 60)
    print("\nThis demo shows how CDCL techniques improve propagator search.")
    print("For detailed benchmarks, run: python -m propagator.examples.benchmark_cdcl")
    
    demo_cdcl_concepts()
    demo_vsids()
    
    print("\n")
    
    try:
        demo_multiple_dwelling_comparison()
    except Exception as e:
        import traceback
        print(f"Error running demo: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The CDCL integration adds:
  + Decision level tracking (CDCLEngine.make_decision)
  + 1-UIP conflict analysis (CDCLEngine.analyze_conflict)  
  + Non-chronological backjumping (CDCLEngine.backjump)
  + VSIDS activity heuristics (CDCLEngine.bump_activity)

API:
  enable_cdcl()   - Turn on CDCL
  disable_cdcl()  - Turn off CDCL (use traditional DDB)
  cdcl_stats()    - Get statistics about CDCL performance
  
For comprehensive benchmarks:
  python -m propagator.examples.benchmark_cdcl

See CDCL_DESIGN.md for implementation details.
""")


if __name__ == '__main__':
    main()
