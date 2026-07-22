# Propagator Network → SAT/SMT Compilation

This module provides tools for compiling propagator network constraints to external solver formats, enabling the use of industrial-strength SAT and SMT solvers.

## Overview

The key insight is that **finite-domain propagator networks** map naturally to SAT/SMT problems:

| Propagator Concept | SAT Encoding | SMT Encoding |
|--------------------|--------------|--------------|
| Cell with domain {1,2,3} | 3 boolean vars (x_1, x_2, x_3) | Integer var with bounds |
| `one_of([1,2,3], x)` | ALO + AMO clauses | `(and (>= x 1) (<= x 3))` |
| `require_distinct(x, y)` | Pairwise exclusion | `(distinct x y)` |
| Learned nogood `{(x,1), (y,2)}` | `(¬x_1 ∨ ¬y_2)` | `(not (and (= x 1) (= y 2)))` |

## Supported Formats

### 1. DIMACS CNF (SAT)
- **Solvers**: MiniSat, CaDiCaL, Kissat, Glucose, CryptoMiniSat
- **Strengths**: Extremely fast for purely combinatorial problems
- **File extension**: `.cnf`

### 2. SMT-LIB2 (SMT)
- **Solvers**: Z3, CVC5, Yices, MathSAT
- **Theories**: 
  - `QF_LIA` - Quantifier-Free Linear Integer Arithmetic
  - `QF_BV` - Quantifier-Free Bit Vectors
- **Strengths**: Native support for integers, arithmetic, distinct
- **File extension**: `.smt2`

### 3. Z3 Python API (Direct)
- **Requirements**: `pip install z3-solver`
- **Strengths**: No file I/O, direct access to Z3 features

## Quick Start

### Recommended: Unified Solve

```python
from propagator import Cell
from propagator.guessing_machine import one_of
from propagator.primitives import adder, constant
from propagator.solver_export import solve, SolveMode, search_mode, SearchMode

with search_mode(SearchMode.DEFER_TO_SMT):
    x, y, z = Cell(name="x"), Cell(name="y"), Cell(name="z")
    one_of({1, 2, 3, 4, 5}, x)
    one_of({1, 2, 3, 4, 5}, y)
    adder(x, y, z)
    constant(7, z)

result = solve([x, y, z], mode=SolveMode.SMT_ITERATIVE, verbose=True)
print(result.solved)
print(result.solution)        # {x: 2, y: 5, z: 7}
print(result.stats)           # {'method': 'smt_iterative', 'rounds': 1, ...}
```

### Low-Level: Build A Compiler Manually

```python
from propagator import Cell, NetworkCompiler, SolverBackend

# Create cells
x = Cell(name="x")
y = Cell(name="y")
z = Cell(name="z")

# Build compiler
compiler = NetworkCompiler(name="my_problem")
compiler.add_domain(x, [1, 2, 3, 4, 5])
compiler.add_domain(y, [1, 2, 3, 4, 5])
compiler.add_domain(z, [1, 2, 3, 4, 5])
compiler.add_all_distinct([x, y, z])
compiler.add_less_than(x, y)

# Export to DIMACS
dimacs = compiler.export(SolverBackend.DIMACS_CNF)
print(dimacs.content)

# Export to SMT-LIB2
smt = compiler.export(SolverBackend.SMT_LIB2)
print(smt.content)

# Solve directly with Z3
result = compiler.solve(SolverBackend.Z3_PYTHON)
print(result.solution)  # {x: 1, y: 2, z: 3}
```

## API Reference

### NetworkCompiler

The main class for building and exporting constraint problems.

#### Adding Variables

```python
# Finite domain (required for SAT, optional for SMT)
compiler.add_domain(cell, [1, 2, 3, 4, 5])

# Boolean variable
compiler.add_boolean(cell)

# Integer with bounds
compiler.add_integer(cell, lower_bound=1, upper_bound=10)
```

#### Adding Constraints

```python
# All variables must have different values
compiler.add_all_distinct([x, y, z])

# Equality and inequality
compiler.add_equality(x, y)      # x = y
compiler.add_inequality(x, y)    # x ≠ y

# Comparisons
compiler.add_less_than(x, y)     # x < y
compiler.add_less_equal(x, y)    # x ≤ y
compiler.add_greater_than(x, y)  # x > y
compiler.add_greater_equal(x, y) # x ≥ y

# Fixed value
compiler.add_fixed_value(x, 5)   # x = 5

# Linear arithmetic
compiler.add_sum_equals([x, y, z], total=10)  # x + y + z = 10
compiler.add_sum_equals([x, y], total=10, coefficients=[2, 3])  # 2x + 3y = 10

# Product (expensive for SAT)
compiler.add_product(x, y, result)  # x * y = result

# Boolean constraints
compiler.add_exactly_one([a, b, c])   # exactly one is true
compiler.add_at_most_one([a, b, c])   # at most one is true
compiler.add_at_least_one([a, b, c])  # at least one is true
compiler.add_implies(a, b)            # a → b
compiler.add_or([a, b, c])            # a ∨ b ∨ c
compiler.add_and([a, b, c])           # a ∧ b ∧ c
```

#### Adding Learned Nogoods

```python
# From TMS conflict analysis: {(x,1), (y,2)} leads to contradiction
compiler.add_nogood([(x, 1), (y, 2)])
# Encodes: ¬(x=1 ∧ y=2), i.e., at least one must be false
```

#### Export and Solve

```python
# Export to string
encoding = compiler.export(SolverBackend.DIMACS_CNF)
print(encoding.content)
print(f"Variables: {encoding.var_count}, Clauses: {encoding.clause_count}")

# Write to file
compiler.write_to_file("problem.cnf", SolverBackend.DIMACS_CNF)
compiler.write_to_file("problem.smt2", SolverBackend.SMT_LIB2)

# Solve and get result
result = compiler.solve(SolverBackend.Z3_PYTHON)
if result.satisfiable:
    print(result.solution)  # Dict[Cell, value]
else:
    print("UNSAT")
```

### Convenience Functions

```python
from propagator import Cell, SolverBackend, TranslationMode, one_of, adder, constant
from propagator.solver_export import compile_from_roots, solve_from_roots

x, y, z = Cell(name="x"), Cell(name="y"), Cell(name="z")

one_of([1, 2, 3, 4, 5], x)
one_of([1, 2, 3, 4, 5], y)
adder(x, y, z)
constant(7, z)

report = compile_from_roots(
    [x, y],
    backend=SolverBackend.Z3_PYTHON,
    mode=TranslationMode.STRICT,
)

result, report = solve_from_roots(
    [x, y],
    backend=SolverBackend.Z3_PYTHON,
    mode=TranslationMode.STRICT,
)

print(report.translated_constraint_count, report.skipped_constraint_count)
```

### Direct Solving

```python
from propagator import solve_dimacs, solve_smtlib2

# Solve DIMACS directly
sat, assignment = solve_dimacs(cnf_content, solver="minisat")

# Solve SMT-LIB2 directly  
sat, model = solve_smtlib2(smt_content, solver="z3")
```

## Integration with Propagator Networks

For new code, prefer `compile_from_roots` / `solve_from_roots`.
Use `NetworkAnalyzer` when you specifically need low-level, incremental
compiler construction outside the root-discovery flow.

### Extracting Constraints from Existing Networks

The `NetworkAnalyzer` class helps bridge propagator networks with the compiler:

```python
from propagator.solver_export import NetworkAnalyzer

analyzer = NetworkAnalyzer("my_problem")

# Register cells
analyzer.register_cell(x, domain=[1, 2, 3, 4, 5])
analyzer.register_cell(y, domain=[1, 2, 3, 4, 5])

# Add constraints
analyzer.add_all_distinct([x, y])

# Extract nogoods from TMS (if available)
analyzer.extract_nogoods_from_tms()

# Convert to compiler
compiler = analyzer.to_compiler()
result = compiler.solve(SolverBackend.Z3_PYTHON)
```

### Feeding Solutions Back to Propagators

Once you have a solution from the external solver, you can feed it back:

```python
result = compiler.solve(SolverBackend.Z3_PYTHON)

if result.satisfiable:
    from propagator import constant
    for cell, value in result.solution.items():
        constant(value, cell)
    # Now the propagator network has the solution
```

## Workflow: Hybrid Solving

A powerful pattern is to use SMT for initial solving, then refine with propagators:

```python
# 1. Build problem specification
compiler = NetworkCompiler("hybrid_example")
for cell in cells:
    compiler.add_domain(cell, domain)
# ... add constraints ...

# 2. Get solution from SMT (fast)
result = compiler.solve(SolverBackend.Z3_PYTHON)

if result.satisfiable:
    # 3. Feed solution to propagator network for validation/refinement
    from propagator import constant, initialize_scheduler, run
    
    initialize_scheduler()
    for cell, value in result.solution.items():
        constant(value, cell)
    run()
    
    # Network now has values propagated
```

## SAT vs SMT: When to Use Which

### Use SAT (DIMACS CNF) when:
- Problem is purely combinatorial (no arithmetic)
- Need maximum speed for large instances
- External solver benchmark required
- Domains are small (avoids variable explosion)

### Use SMT when:
- Problem involves arithmetic (`x + y = z`, `x < y`)
- Domains are large (avoid N² clause explosion)
- Need `distinct` constraint (native in SMT)
- Problem mixes Boolean and arithmetic reasoning

## Limitations

1. **Finite domains required for SAT**: SAT encoding needs enumerable domains
2. **Exponential clauses for arithmetic**: `x + y = z` requires O(|D|³) clauses in SAT
3. **No interval arithmetic**: Intervals map poorly to SAT/SMT
4. **No custom merge**: External solvers don't support propagator-style merge

## Examples

### Multiple Dwelling Problem

```python
baker = Cell(name="baker")
cooper = Cell(name="cooper")
fletcher = Cell(name="fletcher")
miller = Cell(name="miller")
smith = Cell(name="smith")

compiler = NetworkCompiler("multiple_dwelling")

# Domains with initial constraints baked in
compiler.add_domain(baker, [1, 2, 3, 4])      # Not top floor
compiler.add_domain(cooper, [2, 3, 4, 5])     # Not bottom floor  
compiler.add_domain(fletcher, [2, 3, 4])      # Not top or bottom
compiler.add_domain(miller, [1, 2, 3, 4, 5])
compiler.add_domain(smith, [1, 2, 3, 4, 5])

# All different
compiler.add_all_distinct([baker, cooper, fletcher, miller, smith])

# Miller higher than Cooper
compiler.add_greater_than(miller, cooper)

# Not adjacent constraints (as nogoods)
for f in [2, 3, 4]:
    for s in [1, 2, 3, 4, 5]:
        if abs(f - s) == 1:
            compiler.add_nogood([(fletcher, f), (smith, s)])

# Solve
result = compiler.solve(SolverBackend.Z3_PYTHON)
print(result.solution)
```

### N-Queens

```python
n = 8
queens = [Cell(name=f"q{i}") for i in range(n)]

compiler = NetworkCompiler(f"{n}_queens")

# Each queen in a column
for q in queens:
    compiler.add_domain(q, list(range(n)))

# No two queens in same column
compiler.add_all_distinct(queens)

# No two queens on same diagonal
for i in range(n):
    for j in range(i + 1, n):
        row_diff = j - i
        for ci in range(n):
            for cj in range(n):
                if abs(ci - cj) == row_diff:
                    compiler.add_nogood([(queens[i], ci), (queens[j], cj)])

result = compiler.solve(SolverBackend.Z3_PYTHON)
```

## Future Enhancements

1. **Incremental solving**: Feed learned clauses back to solver between calls
2. **Proof extraction**: Get UNSAT proofs for conflict analysis
3. **Optimization**: Support for MaxSAT and optimization objectives
4. **Theory integration**: Support for arrays, uninterpreted functions
5. **Parallel solving**: Portfolio approach with multiple solvers
