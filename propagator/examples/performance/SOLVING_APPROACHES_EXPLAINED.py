#!/usr/bin/env python3
"""
================================================================================
SOLVING APPROACHES EXPLAINED
================================================================================

This document explains the different approaches to solving constraint problems
in this propagator framework, focusing on SPECIFICATION BURDEN - how much
explicit work is required to capture problem constraints.

================================================================================
THE KEY INSIGHT: PROPAGATORS AS IMPLICIT CONSTRAINT ENCODING
================================================================================

Traditional SAT/SMT solving requires you to ENUMERATE constraints explicitly:

    # SAT approach: must list ALL forbidden combinations
    for queen_i in range(n):
        for queen_j in range(i+1, n):
            for row_value in range(n):
                if row_value + (j-i) < n:  # diagonal constraint
                    solver.add_clause(NOT(Q[i,row_value]), NOT(Q[j, row_value+(j-i)]))
                if row_value - (j-i) >= 0:
                    solver.add_clause(NOT(Q[i,row_value]), NOT(Q[j, row_value-(j-i)]))

With propagators, the constraint is COMPUTED, not enumerated:

    # Propagator approach: constraint is computed by the network
    row_diff = Cell()
    subtractor(cells[i], cells[j], row_diff)  # row_diff = cells[i] - cells[j]
    diag_forbidden = Cell()
    eq(row_diff, col_diff, diag_forbidden)    # diag_forbidden = (row_diff == col_diff)
    abhor(diag_forbidden)                      # forbid this condition

The propagator network COMPUTES whether any value assignment violates the
constraint, rather than requiring you to enumerate all forbidden combinations.

================================================================================
APPROACH 1: PROPAGATOR (DDB) - Pure Propagator Network
================================================================================

WHAT YOU WRITE:
    cells = []
    for i in range(n):
        c = Cell(name=f"Q{i}")
        one_of(list(range(n)), c)    # Domain: queen i can be in row 0..n-1
        cells.append(c)
    
    require_distinct(cells)           # No two queens in same row
    
    # Diagonal constraints via propagator wiring
    for i in range(n):
        for j in range(i+1, n):
            diff = j - i
            row_diff = Cell()
            subtractor(cells[i], cells[j], row_diff)
            diag_pos = Cell()
            eq(row_diff, constant(diff), diag_pos)
            abhor(diag_pos)  # forbid positive diagonal
            # ... similar for negative diagonal

HOW IT WORKS:
    1. one_of() creates TMS hypotheticals - possible values with dependencies
    2. require_distinct() creates a network of constraints between hypotheticals
    3. subtractor/eq/abhor create computation pipelines that react to value changes
    4. run() triggers TMS to select hypotheticals and propagate consequences
    5. When abhor() fires on True, TMS backtracks via dependencies

SPECIFICATION BURDEN:
    - Domains: AUTOMATIC via one_of()
    - Constraints: IMPLICIT in propagator wiring
    - Search: AUTOMATIC via TMS

WHAT MAKES IT SPECIAL:
    The constraint "no two queens on same diagonal" is not enumerated as
    forbidden (i,j,row_i,row_j) tuples. Instead, it's COMPUTED by the
    subtractor→eq→abhor pipeline for ANY values that might be assigned.

================================================================================
APPROACH 2: PROPAGATOR + CDCL - With Modern SAT Techniques
================================================================================

Same as Approach 1, but with:
    enable_cdcl()  # Before running

Adds:
    - 1-UIP clause learning (learns from conflicts)
    - Non-chronological backjumping (skips irrelevant decision levels)

Still uses the propagator network - just smarter about search.

================================================================================
APPROACH 3: ROOT-BASED COMPILATION - One API from Existing Networks
================================================================================

WHAT YOU WRITE:
    # Build your normal propagator network
    for i in range(n):
        c = Cell(name=f"Q{i}")
        one_of(list(range(n)), c)
        cells.append(c)
    require_distinct(cells)
    # ... plus arithmetic/order propagators as usual

    # Compile/solve from roots
    result, report = solve_from_roots(
        cells,
        backend=SolverBackend.Z3_PYTHON,
        mode=TranslationMode.HYBRID_ORACLE,
        name="nqueens",
    )

HOW IT WORKS:
    1. Start from root cells
    2. Discover connected network structure and domains
    3. Translate supported constraints automatically
    4. Enforce policy via mode:
       - STRICT: fail if unsupported constraints exist
       - HYBRID_ORACLE: solve supported subset and keep propagator semantics

SPECIFICATION BURDEN:
    - Domains: AUTOMATIC via one_of()
    - Constraints: IMPLICIT in propagator wiring
    - Export path: AUTOMATIC from root discovery

================================================================================
APPROACH 4: DIRECT SAT - No Propagators At All
================================================================================

WHAT YOU WRITE:
    # No TMS, no propagators - just direct specification
    cells = [Cell(name=f"Q{i}") for i in range(n)]
    domains = {c: list(range(n)) for c in cells}
    
    nogoods = []
    for i in range(n):
        for j in range(i+1, n):
            for v in range(n):
                if 0 <= v + (j-i) < n:
                    nogoods.append([(cells[i], v), (cells[j], v+(j-i))])
    
    sat, solution, stats = solve_with_hybrid(
        cells=cells,
        domains=domains,
        constraints={
            'distinct': [cells],
            'nogoods': nogoods,
        },
    )

HOW IT WORKS:
    1. Cells are just identity markers (no TMS involvement)
    2. Domains specified explicitly
    3. Constraints specified explicitly
    4. Direct translation to SAT encoding

SPECIFICATION BURDEN:
    EVERYTHING explicit:
    - Domains: MANUAL
    - Constraints: MANUAL
    
This is essentially what you'd write in MiniZinc or Picat.

================================================================================
APPROACH 5: DIRECT SMT - Native Arithmetic Support (NEW)
================================================================================

WHAT YOU WRITE:
    import z3
    
    # Integer variables with native arithmetic
    queens = [z3.Int(f"Q{i}") for i in range(n)]
    solver = z3.Solver()
    
    # Domains
    for q in queens:
        solver.add(q >= 0, q < n)
    
    # All different
    solver.add(z3.Distinct(queens))
    
    # Diagonals - NATIVE ARITHMETIC!
    for i in range(n):
        for j in range(i+1, n):
            solver.add(z3.Abs(queens[i] - queens[j]) != abs(i - j))
    
    result = solver.check()

HOW IT WORKS:
    1. Use Z3's theory of integers (QF_LIA)
    2. Express arithmetic constraints directly (Abs, <, >, etc.)
    3. No boolean encoding / no nogoods enumeration
    4. Solver handles arithmetic reasoning natively

KEY ADVANTAGE:
    The constraint |Q[i] - Q[j]| != |i - j| is ONE constraint,
    not O(domain_size) nogoods.

SPECIFICATION BURDEN:
    Similar to propagators for arithmetic constraints:
    - Domains: Explicit bounds
    - Constraints: Native expressions (no enumeration!)

WHEN TO USE:
    - Cryptarithmetic (SEND+MORE=MONEY)
    - Linear arithmetic constraints
    - Ordering constraints
    - Any problem with numeric relationships

================================================================================
COMPARISON: WHY PROPAGATORS MATTER
================================================================================

Consider the Multiple Dwelling problem's "not adjacent" constraint:
    Smith does not live on a floor adjacent to Fletcher

PROPAGATOR APPROACH:
    smith_fletcher_diff = Cell()
    subtractor(smith, fletcher, smith_fletcher_diff)
    
    sf_abs = Cell()
    absolute_value(smith_fletcher_diff, sf_abs)
    
    one = Cell()
    constant(1, one)
    
    adjacent = Cell()
    eq(one, sf_abs, adjacent)
    
    abhor(adjacent)
    
    # Works for ANY floor values - constraint is COMPUTED

DIRECT SAT APPROACH:
    # Must enumerate all adjacent pairs
    for floor in [1,2,3,4,5]:
        if floor - 1 >= 1:
            compiler.add_nogood([(smith, floor-1), (fletcher, floor)])
        if floor + 1 <= 5:
            compiler.add_nogood([(smith, floor+1), (fletcher, floor)])
    
    # 8 explicit nogoods for this one constraint

The propagator approach:
- Scales to any domain size without code changes
- Expresses the MEANING of the constraint (|diff| != 1)
- Can work bidirectionally (if fletcher is known, constrains smith)

The direct SAT approach:
- Requires O(domain_size) explicit nogoods
- Must be updated if domain changes
- Loses the semantic meaning of "adjacent"

================================================================================
APPROACH 6: PROPAGATOR → SMT DIRECT TRANSLATION
================================================================================

If propagators and SMT are semantically equivalent, we CAN translate propagator
networks directly to SMT without enumerating nogoods.

CONCEPT:

    from propagator.solver_export.smt import PropagatorTranslator

    translator = PropagatorTranslator()
    a = translator.cell("a", domain=(1, 5))
    b = translator.cell("b", domain=(1, 5))
    diff = translator.cell("diff")
    abs_diff = translator.cell("abs_diff")

    translator.subtractor(a, b, diff)
    translator.absolute_value(diff, abs_diff)
    translator.abhor(abs_diff)  # forbid |a - b| == 0

    result = translator.solve_with_z3()

HOW IT WORKS:
    1. Walk the propagator STRUCTURE (not execution)
    2. Extract RELATIONAL semantics from operational propagators:
       - subtractor(a, b, c) → c = a - b
       - absolute_value(a, b) → b = abs(a)
       - eq(a, b, c) → c = (a == b)
       - abhor(c) → NOT c
    3. COMPOSE them symbolically: abhor(eq(const, abs(sub(qi, qj))))
    4. Emit to SMT: abs(qi - qj) != row_diff

THE KEY MAPPING:

    Propagator Operation          SMT Equivalent
    ────────────────────────────  ──────────────────────────────
    subtractor(a, b, c)           c = a - b
    adder(a, b, c)                c = a + b
    multiplier(a, b, c)           c = a * b
    absolute_value(a, b)          b = abs(a)
    eq(a, b, c)                   c = (a == b)
    lt(a, b, c)                   c = (a < b)
    gt(a, b, c)                   c = (a > b)
    require(c)                    (assert c)
    abhor(c)                      (assert (not c))
    require_distinct(cells)       (assert (distinct ...))
    one_of(vals, c)               (assert (and (>= c min) (<= c max)))

SPECIFICATION BURDEN:

    SAME AS PROPAGATOR - you write propagator code, get SMT semantics:
    - Domains: From one_of / domain bounds
    - Constraints: Extracted from propagator wiring
    - NO EXPLICIT NOGOODS

    Score: Same as native SMT (≈11 for 4-Queens)

WHY ROOT-BASED COMPILATION IS THE DEFAULT:

    The canonical path is to compile/solve from root cells. That keeps the
    authoring model identical to normal propagator code while enabling strict
    translation or hybrid-oracle behavior without hand-maintained export code.

================================================================================
PERFORMANCE TRADEOFFS
================================================================================

Propagators (DDB/CDCL):
    + Implicit constraint encoding
    + Bidirectional propagation
    + Semantic constraint representation
    - Python overhead for propagation
    - TMS infrastructure overhead
    - Slower for pure decidability problems

Direct SAT:
    + Fastest search (optimized solver)
    + No Python in the solving loop
    - Must enumerate all constraints explicitly
    - Loses semantic structure
    - Hard to maintain for complex constraints

================================================================================
SUMMARY: SPECIFICATION BURDEN
================================================================================

OBJECTIVE METRIC (lower is better):
  Score = Domains × 1.0 + Constraints × 1.0 + Enumerations × 2.0 + ArithmeticNogoods × 3.0

                    Domains     Distinct    Arithmetic    Ordering    Score
                    ─────────   ─────────   ──────────    ────────    ─────
Propagator          AUTO        AUTO        IMPLICIT      IMPLICIT    LOW
                    (one_of)    (require_   (propagator   (gt, lt     (no
                                distinct)   wiring)       propagators) enumeration)

Roots-First Export  AUTO        AUTO        IMPLICIT*     IMPLICIT*   LOW-MED
                    (discovered) (discovered) (translated  (translated  (depends on
                                              when supported) when supported) coverage/mode)

SMT                 Explicit    Native      NATIVE        NATIVE      LOW
                    (bounds)    (Distinct)  (Abs, +, -)   (<, >, ==)  (no
                                                                      enumeration)

Direct SAT          MANUAL      MANUAL      MANUAL        MANUAL      HIGH
                    (explicit)  (API)       (nogoods)     (nogoods)   (all
                                                                      enumerated)

EXAMPLE BURDEN SCORES (4-Queens):
    Propagator:       17 (domains + constraint calls, no enumeration)
    Roots-First SMT:  17 (same modeling surface, automatic compilation)
    SMT:              11 (domains + native arithmetic constraints)
    Direct SAT:       89 (must enumerate everything)

================================================================================
"""

if __name__ == "__main__":
    print(__doc__)
