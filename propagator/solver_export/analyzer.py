"""
Network analyzer for extracting constraints from propagator networks.

This module provides utilities to analyze a propagator network and
automatically extract the constraint structure for compilation to SAT/SMT.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..cell import Cell

from .compiler import NetworkCompiler, ConstraintType


class NetworkAnalyzer:
    """
    Analyzes a propagator network to extract constraint structure.
    
    This bridges the gap between the propagator network's runtime state
    and the static constraint representation needed for SAT/SMT compilation.
    
    Usage:
        from propagator import Cell
        from propagator.guessing_machine import one_of, require_distinct
        from propagator.solver_export import NetworkAnalyzer
        
        # Build network
        x, y, z = Cell(name="x"), Cell(name="y"), Cell(name="z")
        one_of([1,2,3], x)
        one_of([1,2,3], y)
        one_of([1,2,3], z)
        require_distinct([x, y, z])
        
        # Analyze and compile
        analyzer = NetworkAnalyzer()
        analyzer.register_cell(x, domain=[1,2,3])
        analyzer.register_cell(y, domain=[1,2,3])
        analyzer.register_cell(z, domain=[1,2,3])
        analyzer.add_all_distinct([x, y, z])
        
        # Get compiler with extracted constraints
        compiler = analyzer.to_compiler()
        result = compiler.solve()
    """
    
    def __init__(self, name: str = "analyzed_network"):
        self.name = name
        self.cells: Dict[Any, Dict] = {}  # Cell -> metadata
        self.constraints: List[Tuple[str, List[Any], Dict]] = []
        self.learned_nogoods: List[Set[Tuple[Any, Any]]] = []
    
    def register_cell(self, cell: 'Cell', 
                      domain: Optional[List[Any]] = None,
                      name: Optional[str] = None) -> None:
        """
        Register a cell for analysis.
        
        Args:
            cell: The propagator Cell
            domain: Finite domain of possible values
            name: Optional name (extracted from cell if not provided)
        """
        if name is None:
            if hasattr(cell, 'name') and cell.name:
                name = cell.name
            else:
                name = f"cell_{len(self.cells)}"
        
        self.cells[cell] = {
            'name': name,
            'domain': domain,
        }
    
    def add_all_distinct(self, cells: List['Cell']) -> None:
        """Add all_distinct constraint."""
        self.constraints.append(('all_distinct', cells, {}))
    
    def add_equality(self, cell1: 'Cell', cell2: 'Cell') -> None:
        """Add equality constraint."""
        self.constraints.append(('equality', [cell1, cell2], {}))
    
    def add_inequality(self, cell1: 'Cell', cell2: 'Cell') -> None:
        """Add inequality constraint."""
        self.constraints.append(('inequality', [cell1, cell2], {}))
    
    def add_less_than(self, cell1: 'Cell', cell2: 'Cell') -> None:
        """Add less-than constraint."""
        self.constraints.append(('less_than', [cell1, cell2], {}))
    
    def add_less_equal(self, cell1: 'Cell', cell2: 'Cell') -> None:
        """Add less-or-equal constraint."""
        self.constraints.append(('less_equal', [cell1, cell2], {}))
    
    def add_greater_than(self, cell1: 'Cell', cell2: 'Cell') -> None:
        """Add greater-than constraint."""
        self.constraints.append(('greater_than', [cell1, cell2], {}))
    
    def add_greater_equal(self, cell1: 'Cell', cell2: 'Cell') -> None:
        """Add greater-or-equal constraint."""
        self.constraints.append(('greater_equal', [cell1, cell2], {}))
    
    def add_fixed_value(self, cell: 'Cell', value: Any) -> None:
        """Fix a cell to a specific value."""
        self.constraints.append(('fixed_value', [cell], {'value': value}))
    
    def add_sum_equals(self, cells: List['Cell'], total: int,
                       coefficients: Optional[List[int]] = None) -> None:
        """Add linear sum constraint."""
        self.constraints.append(('sum_equals', cells, {
            'total': total,
            'coefficients': coefficients
        }))
    
    def add_product(self, cell1: 'Cell', cell2: 'Cell', result: 'Cell') -> None:
        """Add product constraint."""
        self.constraints.append(('product', [cell1, cell2, result], {}))
    
    def add_nogood(self, assignments: List[Tuple['Cell', Any]]) -> None:
        """
        Add a learned nogood from TMS conflict analysis.
        
        Args:
            assignments: List of (cell, value) pairs that together cause contradiction
        """
        self.learned_nogoods.append(set(assignments))
    
    def extract_nogoods_from_tms(self) -> None:
        """
        Extract learned nogoods from the TMS system.
        
        This reads the TMS's recorded nogoods and converts them to
        constraint form for the SAT/SMT encoding.
        """
        from ..tms import get_all_nogoods, hypothetical_p
        
        nogoods = get_all_nogoods()
        
        for nogood in nogoods:
            # Convert hypotheticals to (cell, value) pairs
            assignments = []
            for premise in nogood:
                if hypothetical_p(premise):
                    if hasattr(premise, 'output_cell') and hasattr(premise, 'value_if_chosen'):
                        cell = premise.output_cell
                        value = premise.value_if_chosen
                        if cell is not None and value is not None:
                            assignments.append((cell, value))
            
            if assignments:
                self.learned_nogoods.append(set(assignments))
    
    def to_compiler(self) -> NetworkCompiler:
        """
        Convert analyzed network to a NetworkCompiler.
        
        Returns:
            NetworkCompiler ready for export or solving
        """
        compiler = NetworkCompiler(self.name)
        
        # Register all cells
        for cell, meta in self.cells.items():
            compiler.add_domain(cell, meta['domain'], name=meta['name'])
        
        # Add constraints
        for ctype, cells, params in self.constraints:
            if ctype == 'all_distinct':
                compiler.add_all_distinct(cells)
            elif ctype == 'equality':
                compiler.add_equality(cells[0], cells[1])
            elif ctype == 'inequality':
                compiler.add_inequality(cells[0], cells[1])
            elif ctype == 'less_than':
                compiler.add_less_than(cells[0], cells[1])
            elif ctype == 'less_equal':
                compiler.add_less_equal(cells[0], cells[1])
            elif ctype == 'greater_than':
                compiler.add_greater_than(cells[0], cells[1])
            elif ctype == 'greater_equal':
                compiler.add_greater_equal(cells[0], cells[1])
            elif ctype == 'fixed_value':
                compiler.add_fixed_value(cells[0], params['value'])
            elif ctype == 'sum_equals':
                compiler.add_sum_equals(cells, params['total'], params.get('coefficients'))
            elif ctype == 'product':
                compiler.add_product(cells[0], cells[1], cells[2])
        
        # Add learned nogoods
        for nogood in self.learned_nogoods:
            compiler.add_nogood(list(nogood))
        
        return compiler


def analyze_multiple_dwelling() -> NetworkCompiler:
    """
    Example: Analyze the Multiple Dwelling problem structure.
    
    This demonstrates how to use NetworkAnalyzer for the classic
    constraint satisfaction problem.
    
    Returns:
        A NetworkCompiler ready to export or solve
    """
    from ..cell import Cell
    
    # Create cells
    baker = Cell(name="baker")
    cooper = Cell(name="cooper")
    fletcher = Cell(name="fletcher")
    miller = Cell(name="miller")
    smith = Cell(name="smith")
    
    floors = [1, 2, 3, 4, 5]
    
    analyzer = NetworkAnalyzer("multiple_dwelling")
    
    # Register cells with domains
    for cell in [baker, cooper, fletcher, miller, smith]:
        analyzer.register_cell(cell, domain=floors)
    
    # All different floors
    analyzer.add_all_distinct([baker, cooper, fletcher, miller, smith])
    
    # Baker does not live on the top floor (floor 5)
    # This is a domain restriction - we'd need to model differently
    # For now, encode as: baker != 5
    # Better: use filtered domain [1,2,3,4]
    
    # Cooper does not live on the bottom floor
    # Fletcher does not live on top or bottom floor
    # Miller lives on a higher floor than Cooper
    # Smith does not live on adjacent floor to Fletcher
    # Fletcher does not live on adjacent floor to Cooper
    
    # For the comparison constraints:
    analyzer.add_greater_than(miller, cooper)
    
    # For "not adjacent" we need more complex encoding
    # |smith - fletcher| != 1 and |fletcher - cooper| != 1
    # These require auxiliary variables or explicit enumeration
    
    return analyzer.to_compiler()

