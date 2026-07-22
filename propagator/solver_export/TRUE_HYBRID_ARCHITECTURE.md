# Hybrid Propagator + SMT: Architecture

This document describes the architecture of `solver_export/true_hybrid.py`, the hybrid solver
that combines the propagator network's incremental reasoning with SMT search.

## What True Propagator Semantics Requires

From Radul's thesis, the key properties are:

1. **Merge Lattice**: Values form a semi-lattice with `merge` as join. This is implemented via `GenericOperator` that dispatches on type, allowing extension for intervals, supported values, TMS, etc.

2. **Dependency Tracking**: Every value knows *why* it believes what it believes. This is `Supported(value, premises)` where premises are hypothetical assumptions.

3. **No-good Learning**: When contradiction detected, compute which premises conflict (the "nogood") and record it to avoid repeating the same mistake.

4. **Truth Maintenance**: The TMS tracks which premises are currently believed, invalidates consequences when premises change, and supports hypothetical reasoning.

## Architecture

`TrueHybridNetwork` uses the **real** propagator infrastructure:

```python
from ..cell import Cell, function_to_propagator_constructor
from ..merge import merge, contradictory_p, the_contradiction
from ..scheduler import initialize_scheduler, run, alert_propagators
from ..primitives import constant, adder, sum_constraint, product
from ..supported_values import Supported, supported, get_support_premises
from ..tms import Tms, Hypothetical, bring_in, kick_out, premise_in, TmsContradiction
```

### SMT as Oracle

SMT acts as a **search oracle** — it handles variable assignment when propagation alone reaches a fixpoint with undetermined cells:

```
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
                         │ (fixpoint reached, still undetermined?)
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
```

### SMT Solutions as Hypothetical Premises

When SMT returns a solution like `{x: 5, y: 3}`, each assignment becomes a **hypothetical premise**:

```python
hyp = SMTHypothesis(cell=cell, value=value, round=self._smt_round)
bring_in(hyp)
sup_val = supported(value, [hyp])
cell.add_content(sup_val)
```

The cell's value has real provenance: `Supported(5, {SMTHyp(x=5, r0)})`.

### Nogood Learning

If these lead to a contradiction, the TMS computes a nogood `{hyp_x, hyp_y}` that is sent back to SMT as a blocking clause:

```python
# In SMT solver:
solver.add(Or(z3_vars["x"] != 5, z3_vars["y"] != 3))
```

## Backward Compatibility

Because `TrueHybridNetwork` uses real `Cell` objects, existing propagator primitives work directly on its cells:

```python
from propagator.primitives import squarer
from propagator.solver_export.true_hybrid import TrueHybridNetwork

net = TrueHybridNetwork()
x = net.cell("x")
x_squared = net.cell("x_squared")
squarer(x, x_squared)  # Works: x is a real Cell
```

## Usage

```python
from propagator.scheduler import initialize_scheduler
from propagator.solver_export.true_hybrid import TrueHybridNetwork

initialize_scheduler()

net = TrueHybridNetwork(name="example")
a = net.cell("a", domain={1, 2, 3, 4})
b = net.cell("b", domain={1, 2, 3, 4})
c = net.cell("c")

net.sum_constraint(a, b, c)
net.all_different([a, b])
net.constant(5, c)  # Forces a + b = 5

if net.solve(verbose=True):
    print(f"a={net.get_value(a)}, b={net.get_value(b)}, c={net.get_value(c)}")
    print(f"Provenance of a: {net.get_provenance(a)}")
```


From Radul's thesis, the key properties are:

1. **Merge Lattice**: Values form a semi-lattice with `merge` as join. This is implemented via `GenericOperator` that dispatches on type, allowing extension for intervals, supported values, TMS, etc.

2. **Dependency Tracking**: Every value knows *why* it believes what it believes. This is `Supported(value, premises)` where premises are hypothetical assumptions.

3. **No-good Learning**: When contradiction detected, compute which premises conflict (the "nogood") and record it to avoid repeating the same mistake.

4. **Truth Maintenance**: The TMS tracks which premises are currently believed, invalidates consequences when premises change, and supports hypothetical reasoning.

## True Hybrid Architecture

The new `solver_export/true_hybrid.py` uses the **real** propagator infrastructure:

```python
# true_hybrid.py - REAL PROPAGATOR INTEGRATION
from ..cell import Cell, function_to_propagator_constructor
from ..merge import merge, contradictory_p, the_contradiction
from ..scheduler import initialize_scheduler, run, alert_propagators
from ..primitives import constant, adder, sum_constraint, product
from ..supported_values import Supported, supported, get_support_premises
from ..tms import Tms, Hypothetical, bring_in, kick_out, premise_in, TmsContradiction
```

### SMT as Oracle, Not Replacement

The key insight is that SMT should be an **oracle for search**, not a replacement for propagator semantics:

```
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
                         │ (fixpoint reached, still undetermined?)
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
```

### SMT Solutions as Hypothetical Premises

When SMT returns a solution like `{x: 5, y: 3}`, we inject these as **hypothetical premises**:

```python
# SMTHypothesis is a premise that the TMS tracks
hyp = SMTHypothesis(cell=cell, value=value, round=self._smt_round)

# Mark hypothesis as believed
bring_in(hyp)

# Add the value as Supported by this hypothesis
sup_val = supported(value, [hyp])
cell.add_content(sup_val)
```

Now the cell's value has **real provenance**: `Supported(5, {SMTHyp(x=5, r0)})`.

### Nogood Learning

If these lead to contradiction, the TMS computes a nogood `{hyp_x, hyp_y}` that we can send back to SMT as a learned clause:

```python
# In SMT solver, add blocking clause:
solver.add(Or(z3_vars["x"] != 5, z3_vars["y"] != 3))
```

This is proper nogood learning - we remember which combinations fail and never try them again.

## Feature Comparison

| Feature            | Original hybrid.py     | TrueHybrid              |
|--------------------|------------------------|-------------------------|
| Cell class         | HybridCell (reimpl.)   | **REAL Cell**           |
| Scheduler          | Manual loop            | **REAL scheduler**      |
| Merge              | Hard-coded _merge()    | **REAL GenericOperator**|
| Provenance         | Dict (fake)            | **REAL Supported**      |
| Nogoods            | None                   | **REAL TMS tracking**   |
| Generic operators  | None                   | **REAL dispatch**       |
| Backward compat    | No                     | **Yes**                 |
| Extensible merge   | No                     | **Yes**                 |

## Backward Compatibility

Because `TrueHybridNetwork` uses real `Cell` objects, you can use **any existing propagator** directly:

```python
from propagator.primitives import squarer, sqrter, product
from propagator.solver_export.true_hybrid import TrueHybridNetwork

net = TrueHybridNetwork()
x = net.cell("x")
x_squared = net.cell("x_squared")

# Use REAL propagator primitives on network cells!
squarer(x, x_squared)  # Works because x is a real Cell

constant(5, x)
net.propagate()
# x_squared.content == 25
```

## What We Can Now Legitimately Claim

With `solver_export/true_hybrid.py`:

✓ **Propagator Semantics**: Uses real Cell, real merge lattice, real scheduler  
✓ **Provenance Tracking**: Every value is `Supported(value, premises)` - real dependency tracking  
✓ **Nogood Learning**: TMS computes and stores nogoods when contradictions occur  
✓ **Backward Compatibility**: Can use existing propagator primitives directly  
✓ **Extensibility**: Can add custom merge operations for new types  
✓ **SMT Search Power**: When propagation insufficient, delegate to SMT  

## Usage

```python
from propagator.scheduler import initialize_scheduler
from propagator.solver_export.true_hybrid import TrueHybridNetwork

initialize_scheduler()

net = TrueHybridNetwork(name="example")

a = net.cell("a", domain={1, 2, 3, 4})
b = net.cell("b", domain={1, 2, 3, 4})
c = net.cell("c")

net.sum_constraint(a, b, c)
net.all_different([a, b])
net.constant(5, c)  # Forces a + b = 5

if net.solve(verbose=True):
    print(f"a={net.get_value(a)}, b={net.get_value(b)}, c={net.get_value(c)}")
    print(f"Provenance of a: {net.get_provenance(a)}")  # Real provenance!
```

Output:
```
Round 0: Propagation...
  2 undetermined cells, calling SMT...
  SMT solution: {'a': 4, 'b': 1}
Round 1: Propagation...
  Solved!
a=4, b=1, c=5
Provenance of a: [SMTHyp(a=4, r0, in)]
```
