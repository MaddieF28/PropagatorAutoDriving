"""
Solver execution and result parsing.

This module provides utilities for running external SAT/SMT solvers
and parsing their output back to solution dictionaries.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .compiler import NetworkCompiler, EncodingResult

from .compiler import SolverBackend, SolverResult


class SolverRunner:
    """
    Runs external SAT/SMT solvers and parses results.
    
    Supports automatic solver detection or explicit solver path.
    """
    
    # Known solver executables by backend
    KNOWN_SOLVERS = {
        SolverBackend.DIMACS_CNF: [
            "kissat", "cadical", "minisat", "glucose", "cryptominisat"
        ],
        SolverBackend.SMT_LIB2: [
            "z3", "cvc5", "cvc4", "yices-smt2", "mathsat"
        ],
        SolverBackend.SMT_LIB2_QF_LIA: [
            "z3", "cvc5", "cvc4", "yices-smt2"
        ],
        SolverBackend.SMT_LIB2_QF_BV: [
            "z3", "cvc5", "boolector"
        ],
    }
    
    def __init__(self, backend: SolverBackend, 
                 solver_path: Optional[str] = None,
                 timeout: Optional[float] = None):
        """
        Initialize solver runner.
        
        Args:
            backend: Which solver format to use
            solver_path: Explicit path to solver executable (auto-detect if None)
            timeout: Timeout in seconds for solver execution
        """
        self.backend = backend
        self.solver_path = solver_path or self._find_solver()
        self.timeout = timeout
    
    def _find_solver(self) -> Optional[str]:
        """Find an available solver for this backend."""
        candidates = self.KNOWN_SOLVERS.get(self.backend, [])
        
        for solver in candidates:
            try:
                result = subprocess.run(
                    ["which", solver],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return solver
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return None
    
    def solve(self, encoding: 'EncodingResult', 
              compiler: 'NetworkCompiler') -> SolverResult:
        """
        Run the solver and return the result.
        
        Args:
            encoding: The encoded problem
            compiler: The compiler (for decoding results)
            
        Returns:
            SolverResult with solution if SAT
        """
        if self.backend == SolverBackend.Z3_PYTHON:
            return self._solve_z3_python(encoding, compiler)
        
        if not self.solver_path:
            return SolverResult(
                satisfiable=False,
                error=f"No solver found for {self.backend.name}. "
                      f"Tried: {self.KNOWN_SOLVERS.get(self.backend, [])}"
            )
        
        if self.backend == SolverBackend.DIMACS_CNF:
            return self._solve_dimacs(encoding, compiler)
        else:
            return self._solve_smtlib2(encoding, compiler)
    
    def _solve_z3_python(self, encoding: 'EncodingResult',
                          compiler: 'NetworkCompiler') -> SolverResult:
        """Solve using Z3 Python API directly."""
        try:
            import z3
        except ImportError:
            return SolverResult(
                satisfiable=False,
                error="Z3 Python bindings not installed. Run: pip install z3-solver"
            )
        
        solver = encoding.metadata.get("solver")
        z3_vars = encoding.metadata.get("z3_vars", {})
        
        if solver is None:
            return SolverResult(satisfiable=False, error="Z3 solver not in encoding")
        
        # Set timeout if specified
        if self.timeout:
            solver.set("timeout", int(self.timeout * 1000))
        
        result = solver.check()
        
        if result == z3.sat:
            model = solver.model()
            raw_assignment = {}
            
            for name, z3_var in z3_vars.items():
                val = model.eval(z3_var, model_completion=True)
                if z3.is_int_value(val):
                    raw_assignment[name] = val.as_long()
                elif z3.is_true(val):
                    raw_assignment[name] = True
                elif z3.is_false(val):
                    raw_assignment[name] = False
                else:
                    raw_assignment[name] = str(val)
            
            # Decode to Cell -> value
            solution = {}
            for cell, var in compiler.variables.items():
                if var.name in raw_assignment:
                    solution[cell] = raw_assignment[var.name]
            
            return SolverResult(
                satisfiable=True,
                solution=solution,
                raw_assignment=raw_assignment,
                stats={"solver": "z3-python"}
            )
        elif result == z3.unsat:
            return SolverResult(
                satisfiable=False,
                stats={"solver": "z3-python", "result": "unsat"}
            )
        else:
            return SolverResult(
                satisfiable=False,
                error=f"Z3 returned: {result}",
                stats={"solver": "z3-python"}
            )
    
    def _solve_dimacs(self, encoding: 'EncodingResult',
                      compiler: 'NetworkCompiler') -> SolverResult:
        """Solve using external DIMACS CNF solver."""
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
            f.write(encoding.content)
            cnf_path = f.name
        
        try:
            # Run solver
            cmd = [self.solver_path, cnf_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout + result.stderr
            
            # Parse result
            if "UNSATISFIABLE" in output or "s UNSATISFIABLE" in output:
                return SolverResult(
                    satisfiable=False,
                    stats={"solver": self.solver_path, "output": output[:500]}
                )
            
            if "SATISFIABLE" in output or "s SATISFIABLE" in output:
                # Parse the solution line(s)
                raw_assignment = self._parse_dimacs_solution(output)
                solution = compiler.decode_solution(raw_assignment)
                
                return SolverResult(
                    satisfiable=True,
                    solution=solution,
                    raw_assignment=raw_assignment,
                    stats={"solver": self.solver_path}
                )
            
            return SolverResult(
                satisfiable=False,
                error=f"Could not parse solver output: {output[:500]}"
            )
            
        except subprocess.TimeoutExpired:
            return SolverResult(
                satisfiable=False,
                error=f"Solver timed out after {self.timeout}s"
            )
        except Exception as e:
            return SolverResult(
                satisfiable=False,
                error=f"Solver error: {e}"
            )
        finally:
            os.unlink(cnf_path)
    
    def _parse_dimacs_solution(self, output: str) -> Dict[int, bool]:
        """Parse DIMACS solution format."""
        assignment = {}
        
        # Look for "v" lines or space-separated numbers
        for line in output.split('\n'):
            line = line.strip()
            
            # Skip comments and status lines
            if line.startswith('c') or line.startswith('s'):
                continue
            
            # Variable lines start with 'v' or are just numbers
            if line.startswith('v'):
                line = line[1:].strip()
            
            # Parse literals
            for lit_str in line.split():
                try:
                    lit = int(lit_str)
                    if lit == 0:
                        continue  # End marker
                    var = abs(lit)
                    assignment[var] = (lit > 0)
                except ValueError:
                    continue
        
        return assignment
    
    def _solve_smtlib2(self, encoding: 'EncodingResult',
                       compiler: 'NetworkCompiler') -> SolverResult:
        """Solve using external SMT-LIB2 solver."""
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(encoding.content)
            smt_path = f.name
        
        try:
            # Run solver (different solvers have different invocation patterns)
            solver_name = os.path.basename(self.solver_path)
            
            if "z3" in solver_name:
                cmd = [self.solver_path, smt_path]
            elif "cvc" in solver_name:
                cmd = [self.solver_path, "--lang", "smt2", smt_path]
            elif "yices" in solver_name:
                cmd = [self.solver_path, smt_path]
            else:
                cmd = [self.solver_path, smt_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            output = result.stdout
            
            # Check result
            if "unsat" in output.lower():
                return SolverResult(
                    satisfiable=False,
                    stats={"solver": self.solver_path}
                )
            
            if "sat" in output.lower():
                # Parse the model
                raw_assignment = self._parse_smtlib2_model(output)
                solution = compiler.decode_solution(raw_assignment)
                
                return SolverResult(
                    satisfiable=True,
                    solution=solution,
                    raw_assignment=raw_assignment,
                    stats={"solver": self.solver_path}
                )
            
            return SolverResult(
                satisfiable=False,
                error=f"Could not parse solver output: {output[:500]}"
            )
            
        except subprocess.TimeoutExpired:
            return SolverResult(
                satisfiable=False,
                error=f"Solver timed out after {self.timeout}s"
            )
        except Exception as e:
            return SolverResult(
                satisfiable=False,
                error=f"Solver error: {e}"
            )
        finally:
            os.unlink(smt_path)
    
    def _parse_smtlib2_model(self, output: str) -> Dict[str, Any]:
        """
        Parse SMT-LIB2 model output.
        
        Z3 format:
            (model
              (define-fun x () Int 3)
              (define-fun y () Int 5)
            )
        
        CVC5 format:
            (
              (define-fun x () Int 3)
              (define-fun y () Int 5)
            )
        """
        assignment = {}
        
        # Pattern for define-fun: (define-fun name () Type value)
        pattern = r'\(define-fun\s+(\w+)\s+\(\)\s+\w+\s+(.+?)\)'
        
        for match in re.finditer(pattern, output, re.DOTALL):
            name = match.group(1)
            value_str = match.group(2).strip()
            
            # Parse the value
            if value_str == "true":
                assignment[name] = True
            elif value_str == "false":
                assignment[name] = False
            elif value_str.startswith("(- "):
                # Negative number: (- 5) -> -5
                num = int(value_str[3:-1].strip())
                assignment[name] = -num
            else:
                try:
                    assignment[name] = int(value_str)
                except ValueError:
                    assignment[name] = value_str
        
        return assignment


# Convenience functions
def solve_dimacs(cnf_content: str, solver: str = None, 
                 timeout: float = None) -> Tuple[bool, Optional[Dict[int, bool]]]:
    """
    Solve a DIMACS CNF problem directly.
    
    Args:
        cnf_content: The CNF content in DIMACS format
        solver: Solver executable (auto-detect if None)
        timeout: Timeout in seconds
        
    Returns:
        (satisfiable, assignment) where assignment maps var -> bool
    """
    # Create minimal compiler for decoding
    from .compiler import NetworkCompiler
    compiler = NetworkCompiler("direct")
    
    # Create encoding result
    result = SolverResult(
        satisfiable=False,
        solution=None
    )
    
    runner = SolverRunner(SolverBackend.DIMACS_CNF, solver_path=solver, timeout=timeout)
    
    # Write temp file and solve
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write(cnf_content)
        cnf_path = f.name
    
    try:
        cmd = [runner.solver_path or "minisat", cnf_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout + proc.stderr
        
        if "UNSATISFIABLE" in output:
            return False, None
        
        if "SATISFIABLE" in output:
            assignment = runner._parse_dimacs_solution(output)
            return True, assignment
        
        return False, None
    finally:
        os.unlink(cnf_path)


def solve_smtlib2(smt_content: str, solver: str = None,
                  timeout: float = None) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Solve an SMT-LIB2 problem directly.
    
    Args:
        smt_content: The SMT-LIB2 content
        solver: Solver executable (auto-detect if None)
        timeout: Timeout in seconds
        
    Returns:
        (satisfiable, model) where model maps var_name -> value
    """
    runner = SolverRunner(SolverBackend.SMT_LIB2, solver_path=solver, timeout=timeout)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
        f.write(smt_content)
        smt_path = f.name
    
    try:
        cmd = [runner.solver_path or "z3", smt_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = proc.stdout
        
        if "unsat" in output.lower():
            return False, None
        
        if "sat" in output.lower():
            model = runner._parse_smtlib2_model(output)
            return True, model
        
        return False, None
    finally:
        os.unlink(smt_path)
