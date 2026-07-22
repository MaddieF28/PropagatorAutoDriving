"""
Shared types for the solver_export subsystem.

These are pure data types (enums and dataclasses) with no behavior and no
dependency on NetworkCompiler, the backends, or the solver runner. They
exist in their own module specifically so that compiler.py, backends.py,
and solver_runner.py can all depend on them without depending on each
other -- previously Variable/Constraint/etc. were defined inside
compiler.py, which meant backends.py and solver_runner.py (which need
these types) had to import compiler.py, while compiler.py itself needs to
dispatch into backends.py/solver_runner.py, forming a real import cycle
only avoided via deferred (function-body) imports. See NetworkCompiler in
compiler.py for the class that actually builds and holds these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class SolverBackend(Enum):
    """Supported solver backend formats."""
    DIMACS_CNF = auto()      # Pure SAT (MiniSat, CaDiCaL, Kissat)
    SMT_LIB2 = auto()        # SMT-LIB2 format (Z3, CVC5, Yices)
    SMT_LIB2_QF_LIA = auto() # SMT-LIB2 with Quantifier-Free Linear Integer Arithmetic
    SMT_LIB2_QF_BV = auto()  # SMT-LIB2 with Quantifier-Free Bit Vectors
    Z3_PYTHON = auto()       # Direct Z3 Python API (if z3-solver installed)


class ConstraintType(Enum):
    """Types of constraints that can be encoded."""
    DOMAIN = auto()           # Variable has finite domain
    ALL_DISTINCT = auto()     # All variables must have different values
    EQUALITY = auto()         # Two variables must be equal
    INEQUALITY = auto()       # Two variables must be different
    LESS_THAN = auto()        # x < y
    LESS_EQUAL = auto()       # x <= y
    GREATER_THAN = auto()     # x > y
    GREATER_EQUAL = auto()    # x >= y
    LINEAR_SUM = auto()       # a1*x1 + a2*x2 + ... = c
    PRODUCT = auto()          # x * y = z
    ABSOLUTE_VALUE = auto()   # |x| = y
    IMPLIES = auto()          # x => y (boolean)
    OR = auto()               # x1 ∨ x2 ∨ ... ∨ xn
    AND = auto()              # x1 ∧ x2 ∧ ... ∧ xn
    NOT = auto()              # ¬x
    EXACTLY_ONE = auto()      # Exactly one of the variables is true
    AT_MOST_ONE = auto()      # At most one of the variables is true
    AT_LEAST_ONE = auto()     # At least one of the variables is true
    FIXED_VALUE = auto()      # Variable must have specific value
    CUSTOM_CLAUSE = auto()    # Raw clause (for learned clauses)
    REIFIED_COMPARISON = auto()  # flag == (lhs <op> rhs) for unpinned ternary comparisons


@dataclass
class Variable:
    """
    Represents a variable in the solver encoding.

    For finite domain variables, we create boolean indicator variables:
        x ∈ {1, 2, 3} becomes x_1, x_2, x_3 (booleans)

    For SMT, we can also use native integer/bitvector types.
    """
    name: str
    cell: Any  # Reference to original Cell
    domain: Optional[List[Any]] = None
    var_type: str = "finite_domain"  # or "integer", "boolean", "bitvector"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False


@dataclass
class Constraint:
    """Represents a constraint in the intermediate representation."""
    constraint_type: ConstraintType
    variables: List[Variable]
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Constraint({self.constraint_type.name}, vars={[v.name for v in self.variables]}, {self.params})"


@dataclass
class EncodingResult:
    """Result of encoding the network for a specific backend."""
    backend: SolverBackend
    content: str  # The encoded content (DIMACS or SMT-LIB2)
    var_count: int
    clause_count: int  # For SAT, number of clauses; for SMT, number of asserts
    variable_map: Dict[str, Any]  # Maps solver vars back to (Cell, value)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Result from solving the encoded problem."""
    satisfiable: bool
    solution: Optional[Dict[Any, Any]] = None  # Cell -> value mapping
    raw_assignment: Optional[Dict[str, Any]] = None  # Solver variable assignments
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
