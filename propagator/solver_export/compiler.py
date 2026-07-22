"""
Core compiler for translating propagator networks to solver formats.

This module provides the main NetworkCompiler class that builds an intermediate
representation of constraints, which can then be exported to various solver formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from itertools import combinations

from .types import (
    SolverBackend,
    ConstraintType,
    Variable,
    Constraint,
    EncodingResult,
    SolverResult,
)


class NetworkCompiler:
    """
    Compiles propagator network constraints to solver formats.
    
    This class builds an intermediate representation of constraints that can
    be exported to various solver backends (DIMACS CNF, SMT-LIB2, etc.).
    
    The key insight is that finite-domain propagator networks map naturally to
    SAT/SMT problems:
    - Each cell with domain D becomes |D| boolean variables (or one integer var in SMT)
    - Constraints like all_distinct become clauses/assertions
    - Learned nogoods from the TMS are directly expressible as clauses
    
    Example:
        compiler = NetworkCompiler()
        
        # Define cells and their domains
        x = Cell(name="x")
        y = Cell(name="y")
        compiler.add_domain(x, [1, 2, 3])
        compiler.add_domain(y, [1, 2, 3])
        
        # Add constraints
        compiler.add_all_distinct([x, y])
        
        # Export or solve
        dimacs = compiler.export(SolverBackend.DIMACS_CNF)
        solution = compiler.solve(SolverBackend.SMT_LIB2)
    """
    
    def __init__(self, name: str = "propagator_network"):
        self.name = name
        self.variables: Dict[Any, Variable] = {}  # Cell -> Variable
        self.constraints: List[Constraint] = []
        self.learned_clauses: List[List[Tuple[Any, Any, bool]]] = []  # [(cell, value, positive), ...]
        
        # For SAT encoding (boolean indicators)
        self._sat_var_counter = 0
        self._sat_var_map: Dict[Tuple[Any, Any], int] = {}  # (cell, value) -> SAT var
        self._sat_decode_map: Dict[int, Tuple[Any, Any]] = {}  # SAT var -> (cell, value)
        
        # SMT variable names
        self._smt_var_names: Dict[Any, str] = {}  # Cell -> SMT var name
    
    # =========================================================================
    # Variable and Domain Registration
    # =========================================================================
    
    def add_variable(self, cell: Any, domain: Optional[List[Any]] = None,
                     name: Optional[str] = None, var_type: str = "finite_domain") -> Variable:
        """
        Register a cell as a variable in the encoding.
        
        Args:
            cell: The propagator Cell object
            domain: Finite domain of possible values (required for SAT, optional for SMT integer)
            name: Optional name (defaults to cell.name or generated)
            var_type: "finite_domain", "integer", "boolean", or "bitvector"
            
        Returns:
            The Variable object created
        """
        if cell in self.variables:
            var = self.variables[cell]
            # Update domain if provided and not already set
            if domain is not None and var.domain is None:
                var.domain = list(domain)
            return var
        
        # Generate name if not provided
        if name is None:
            if hasattr(cell, 'name') and cell.name:
                name = cell.name
            else:
                name = f"v{len(self.variables)}"
        
        # Ensure unique name
        base_name = name
        counter = 1
        while any(v.name == name for v in self.variables.values()):
            name = f"{base_name}_{counter}"
            counter += 1
        
        var = Variable(
            name=name,
            cell=cell,
            domain=list(domain) if domain else None,
            var_type=var_type
        )
        self.variables[cell] = var
        self._smt_var_names[cell] = name
        
        # Pre-create SAT variables for domain values
        if domain is not None:
            for value in domain:
                self._get_sat_var(cell, value)
        
        return var
    
    def add_domain(self, cell: Any, domain: List[Any], name: Optional[str] = None) -> Variable:
        """
        Add a cell with a finite domain. Shorthand for add_variable with domain.

        This also implicitly adds the domain constraint (exactly one value).
        """
        # A domain of only True/False is really a boolean variable, not an
        # integer one -- route it accordingly so backends emit Bool/true/false
        # instead of treating it as an Int bounded to 0/1.
        var_type = "boolean" if domain and all(isinstance(v, bool) for v in domain) else "finite_domain"
        var = self.add_variable(cell, domain=domain, name=name, var_type=var_type)
        
        # Add the domain constraint (cell must have exactly one value from domain)
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.DOMAIN,
            variables=[var],
            params={"domain": list(domain)}
        ))
        
        return var
    
    def add_boolean(self, cell: Any, name: Optional[str] = None) -> Variable:
        """Add a boolean variable."""
        return self.add_variable(cell, domain=[True, False], name=name, var_type="boolean")
    
    def add_integer(self, cell: Any, name: Optional[str] = None,
                    lower_bound: Optional[int] = None, 
                    upper_bound: Optional[int] = None) -> Variable:
        """
        Add an integer variable (SMT only, or bounded for SAT).
        
        For SAT encoding, bounds are required to create the domain.
        """
        var = self.add_variable(cell, name=name, var_type="integer")
        
        if lower_bound is not None and upper_bound is not None:
            # Create finite domain for SAT compatibility
            domain = list(range(lower_bound, upper_bound + 1))
            var.domain = domain
            for value in domain:
                self._get_sat_var(cell, value)
            
            # Add domain constraint
            self.constraints.append(Constraint(
                constraint_type=ConstraintType.DOMAIN,
                variables=[var],
                params={"domain": domain, "lower_bound": lower_bound, "upper_bound": upper_bound}
            ))
        
        return var
    
    # =========================================================================
    # SAT Variable Management
    # =========================================================================
    
    def _get_sat_var(self, cell: Any, value: Any) -> int:
        """Get or create a SAT variable for (cell, value) pair."""
        key = (cell, value)
        if key not in self._sat_var_map:
            self._sat_var_counter += 1
            self._sat_var_map[key] = self._sat_var_counter
            self._sat_decode_map[self._sat_var_counter] = key
        return self._sat_var_map[key]
    
    def _get_smt_var_name(self, cell: Any) -> str:
        """Get the SMT variable name for a cell."""
        if cell not in self._smt_var_names:
            if cell in self.variables:
                self._smt_var_names[cell] = self.variables[cell].name
            else:
                self._smt_var_names[cell] = f"v{len(self._smt_var_names)}"
        return self._smt_var_names[cell]
    
    # =========================================================================
    # Constraint Addition
    # =========================================================================
    
    def add_all_distinct(self, cells: List[Any]) -> None:
        """Add constraint that all cells must have different values."""
        vars = [self._ensure_variable(cell) for cell in cells]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.ALL_DISTINCT,
            variables=vars,
            params={}
        ))
    
    def add_equality(self, cell1: Any, cell2: Any) -> None:
        """Add constraint that two cells must be equal."""
        vars = [self._ensure_variable(cell1), self._ensure_variable(cell2)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.EQUALITY,
            variables=vars,
            params={}
        ))
    
    def add_inequality(self, cell1: Any, cell2: Any) -> None:
        """Add constraint that two cells must be different."""
        vars = [self._ensure_variable(cell1), self._ensure_variable(cell2)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.INEQUALITY,
            variables=vars,
            params={}
        ))
    
    def add_less_than(self, cell1: Any, cell2: Any) -> None:
        """Add constraint: cell1 < cell2."""
        vars = [self._ensure_variable(cell1), self._ensure_variable(cell2)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.LESS_THAN,
            variables=vars,
            params={}
        ))
    
    def add_less_equal(self, cell1: Any, cell2: Any) -> None:
        """Add constraint: cell1 <= cell2."""
        vars = [self._ensure_variable(cell1), self._ensure_variable(cell2)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.LESS_EQUAL,
            variables=vars,
            params={}
        ))
    
    def add_greater_than(self, cell1: Any, cell2: Any) -> None:
        """Add constraint: cell1 > cell2."""
        vars = [self._ensure_variable(cell1), self._ensure_variable(cell2)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.GREATER_THAN,
            variables=vars,
            params={}
        ))
    
    def add_greater_equal(self, cell1: Any, cell2: Any) -> None:
        """Add constraint: cell1 >= cell2."""
        vars = [self._ensure_variable(cell1), self._ensure_variable(cell2)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.GREATER_EQUAL,
            variables=vars,
            params={}
        ))
    
    def add_reified_comparison(self, flag_cell: Any, lhs: Any, rhs: Any, op: str) -> None:
        """
        Add a reified comparison: flag == (lhs <op> rhs).
        
        This encodes the constraint that the boolean flag cell equals the
        result of the comparison (lhs <op> rhs), without fixing the flag.
        Used for ternary comparison propagators (eq/lt/gt/lte/gte) whose
        boolean output is not pinned by require()/abhor().
        
        Args:
            flag_cell: The boolean cell holding the comparison result
            lhs: Left-hand side of the comparison
            rhs: Right-hand side of the comparison
            op: One of 'eq', 'lt', 'gt', 'lte', 'gte'
        """
        vars = [
            self._ensure_variable(flag_cell),
            self._ensure_variable(lhs),
            self._ensure_variable(rhs),
        ]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.REIFIED_COMPARISON,
            variables=vars,
            params={'op': op}
        ))
    
    def add_fixed_value(self, cell: Any, value: Any) -> None:
        """Fix a cell to a specific value."""
        var = self._ensure_variable(cell)
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.FIXED_VALUE,
            variables=[var],
            params={"value": value}
        ))
    
    def add_sum_equals(self, cells: List[Any], total: int, 
                       coefficients: Optional[List[int]] = None) -> None:
        """
        Add linear sum constraint: sum(coeff[i] * cells[i]) = total.
        
        If coefficients not provided, all are assumed to be 1.
        """
        vars = [self._ensure_variable(cell) for cell in cells]
        coeffs = coefficients if coefficients else [1] * len(cells)
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.LINEAR_SUM,
            variables=vars,
            params={"total": total, "coefficients": coeffs, "relation": "="}
        ))
    
    def add_sum_less_equal(self, cells: List[Any], bound: int,
                           coefficients: Optional[List[int]] = None) -> None:
        """Add: sum(coeff[i] * cells[i]) <= bound."""
        vars = [self._ensure_variable(cell) for cell in cells]
        coeffs = coefficients if coefficients else [1] * len(cells)
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.LINEAR_SUM,
            variables=vars,
            params={"total": bound, "coefficients": coeffs, "relation": "<="}
        ))
    
    def add_product(self, cell1: Any, cell2: Any, result: Any) -> None:
        """Add constraint: cell1 * cell2 = result."""
        vars = [self._ensure_variable(cell1), 
                self._ensure_variable(cell2),
                self._ensure_variable(result)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.PRODUCT,
            variables=vars,
            params={}
        ))

    def add_absolute_value(self, cell: Any, result: Any) -> None:
        """Add constraint: |cell| = result."""
        vars = [self._ensure_variable(cell), self._ensure_variable(result)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.ABSOLUTE_VALUE,
            variables=vars,
            params={}
        ))
    
    def add_exactly_one(self, cells: List[Any]) -> None:
        """Exactly one of the cells must be true (for boolean cells)."""
        vars = [self._ensure_variable(cell) for cell in cells]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.EXACTLY_ONE,
            variables=vars,
            params={}
        ))
    
    def add_at_most_one(self, cells: List[Any]) -> None:
        """At most one of the cells is true."""
        vars = [self._ensure_variable(cell) for cell in cells]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.AT_MOST_ONE,
            variables=vars,
            params={}
        ))
    
    def add_at_least_one(self, cells: List[Any]) -> None:
        """At least one of the cells is true."""
        vars = [self._ensure_variable(cell) for cell in cells]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.AT_LEAST_ONE,
            variables=vars,
            params={}
        ))
    
    def add_implies(self, antecedent: Any, consequent: Any) -> None:
        """Add implication: antecedent => consequent (for boolean cells)."""
        vars = [self._ensure_variable(antecedent), self._ensure_variable(consequent)]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.IMPLIES,
            variables=vars,
            params={}
        ))
    
    def add_or(self, cells: List[Any]) -> None:
        """Add disjunction constraint (for boolean cells)."""
        vars = [self._ensure_variable(cell) for cell in cells]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.OR,
            variables=vars,
            params={}
        ))
    
    def add_and(self, cells: List[Any]) -> None:
        """Add conjunction constraint (for boolean cells)."""
        vars = [self._ensure_variable(cell) for cell in cells]
        self.constraints.append(Constraint(
            constraint_type=ConstraintType.AND,
            variables=vars,
            params={}
        ))
    
    def add_raw_clause(self, literals: List[Tuple[Any, Any, bool]]) -> None:
        """
        Add a raw clause (useful for learned clauses).
        
        Each literal is (cell, value, positive):
        - positive=True means cell=value must hold
        - positive=False means cell=value must NOT hold (negation)
        
        The clause is the disjunction of all literals.
        """
        self.learned_clauses.append(literals)
    
    def add_nogood(self, nogood: List[Tuple[Any, Any]]) -> None:
        """
        Add a nogood clause: at least one of (cell, value) pairs must not hold.
        
        This is the standard TMS nogood format: these assignments together
        lead to contradiction.
        
        Args:
            nogood: List of (cell, value) pairs that together cause contradiction
        """
        # A nogood {(c1,v1), (c2,v2), (c3,v3)} becomes clause:
        # ¬(c1=v1) ∨ ¬(c2=v2) ∨ ¬(c3=v3)
        # Which in positive form: at least one must be false
        clause = [(cell, value, False) for cell, value in nogood]
        self.add_raw_clause(clause)
    
    def _ensure_variable(self, cell: Any) -> Variable:
        """Ensure cell is registered as a variable, return the Variable."""
        if cell not in self.variables:
            return self.add_variable(cell)
        return self.variables[cell]
    
    # =========================================================================
    # Export Interface
    # =========================================================================
    
    def export(self, backend: SolverBackend = SolverBackend.DIMACS_CNF) -> EncodingResult:
        """
        Export the network to the specified solver format.
        
        Args:
            backend: The target solver format
            
        Returns:
            EncodingResult with the encoded content and metadata
        """
        from .backends import DimacsBackend, SMTLib2Backend
        
        if backend == SolverBackend.DIMACS_CNF:
            encoder = DimacsBackend(self)
            return encoder.encode()
        elif backend in (SolverBackend.SMT_LIB2, SolverBackend.SMT_LIB2_QF_LIA):
            encoder = SMTLib2Backend(self, logic="QF_LIA")
            return encoder.encode()
        elif backend == SolverBackend.SMT_LIB2_QF_BV:
            encoder = SMTLib2Backend(self, logic="QF_BV")
            return encoder.encode()
        elif backend == SolverBackend.Z3_PYTHON:
            # Requires z3-solver package
            try:
                from .backends import Z3Backend
                encoder = Z3Backend(self)
                return encoder.encode()
            except ImportError:
                raise ImportError("Z3 Python bindings not available. Install with: pip install z3-solver")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def solve(self, backend: SolverBackend = SolverBackend.SMT_LIB2,
              solver_path: Optional[str] = None,
              timeout: Optional[float] = None) -> SolverResult:
        """
        Export and solve the problem.
        
        Args:
            backend: Solver backend to use
            solver_path: Path to external solver (auto-detected if None)
            timeout: Timeout in seconds
            
        Returns:
            SolverResult with the solution (if SAT) or UNSAT indication
        """
        from .solver_runner import SolverRunner
        
        encoding = self.export(backend)
        runner = SolverRunner(backend, solver_path=solver_path, timeout=timeout)
        return runner.solve(encoding, self)
    
    def write_to_file(self, filepath: str, backend: SolverBackend = SolverBackend.DIMACS_CNF) -> EncodingResult:
        """Export and write to a file."""
        encoding = self.export(backend)
        with open(filepath, 'w') as f:
            f.write(encoding.content)
        return encoding
    
    # =========================================================================
    # Inspection
    # =========================================================================
    
    def summary(self) -> str:
        """Return a summary of the compiled network."""
        lines = [
            f"NetworkCompiler: {self.name}",
            f"  Variables: {len(self.variables)}",
            f"  Constraints: {len(self.constraints)}",
            f"  Learned clauses: {len(self.learned_clauses)}",
            f"  SAT variables: {self._sat_var_counter}",
            "",
            "Variables:"
        ]
        for cell, var in self.variables.items():
            domain_str = f"domain={var.domain}" if var.domain else "unbounded"
            lines.append(f"  {var.name}: {var.var_type}, {domain_str}")
        
        lines.append("")
        lines.append("Constraints:")
        for c in self.constraints[:10]:  # First 10
            lines.append(f"  {c}")
        if len(self.constraints) > 10:
            lines.append(f"  ... and {len(self.constraints) - 10} more")
        
        return "\n".join(lines)
    
    def decode_solution(self, raw_assignment: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Decode a solver's raw assignment to cell -> value mapping.
        
        Args:
            raw_assignment: Solver-specific assignment (SAT var -> bool, or SMT var -> value)
            
        Returns:
            Dictionary mapping original Cell objects to their values
        """
        result = {}
        
        # Check if this is a SAT assignment (int keys)
        if raw_assignment and isinstance(next(iter(raw_assignment.keys())), int):
            # SAT: find which (cell, value) indicator is true
            for var, is_true in raw_assignment.items():
                if is_true and var in self._sat_decode_map:
                    cell, value = self._sat_decode_map[var]
                    if cell not in result:
                        result[cell] = value
        else:
            # SMT: variable names map directly
            for var_name, value in raw_assignment.items():
                # Find cell by variable name
                for cell, var in self.variables.items():
                    if var.name == var_name:
                        result[cell] = value
                        break
        
        return result
