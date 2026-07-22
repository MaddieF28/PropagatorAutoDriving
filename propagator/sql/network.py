"""Query network: a collection of named cells and their relationships.

Provides the structure that the parser, stepper, and visualizer operate on.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..cell import Cell


@dataclass
class QueryNetwork:
    """A propagator network representing a SQL query's execution plan."""
    cells: dict = field(default_factory=dict)  # name -> Cell
    operators: list = field(default_factory=list)  # list of (op_type, op_info) tuples
    output_cell: Optional[Cell] = None
    sql: str = ""

    def add_cell(self, name: str, cell: Cell = None) -> Cell:
        if cell is None:
            cell = Cell(name=name)
        self.cells[name] = cell
        return cell

    def get_cell(self, name: str) -> Optional[Cell]:
        return self.cells.get(name)

    def cell_names(self):
        return list(self.cells.keys())

    def __repr__(self):
        return f"QueryNetwork({len(self.cells)} cells, output={self.output_cell})"


def build_network_manual(catalog, plan, use_constraints=False):
    """Build a network from an explicit plan specification.

    plan is a list of dicts, each describing an operator:
        {'op': 'scan', 'table': 'orders', 'alias': 'o'}
        {'op': 'filter', 'input': 'scan_o', 'predicate': fn, 'selectivity': 0.3}
        {'op': 'join', 'left': 'scan_o', 'right': 'scan_c', 'condition': fn, 'selectivity': 0.1}
        {'op': 'aggregate', 'input': 'join_1', 'group_by': [...], 'agg_exprs': [...]}
        {'op': 'project', 'input': 'aggregate', 'columns': [...]}
        {'op': 'sort', 'input': 'project', 'order_by': [...]}
        {'op': 'limit', 'input': 'sort', 'n': 10}

    Returns: QueryNetwork
    """
    from .operators import scan, filter_prop, join_prop, aggregate_prop, project_prop, sort_prop, limit_prop
    from .constraints import filter_constraint, join_constraint

    net = QueryNetwork()
    last_cell = None

    for step in plan:
        op = step['op']

        if op == 'scan':
            alias = step.get('alias', step['table'])
            cell = net.add_cell(f"scan_{alias}")
            # Get table info from catalog, add alias-prefixed keys for join ON conditions
            info = catalog.get_table(step['table'])
            if info is not None:
                from .relation_info import FullRelation
                if isinstance(info, FullRelation) and info.rows:
                    prefixed_rows = []
                    for row in info.rows:
                        new_row = dict(row)
                        for k, v in row.items():
                            new_row[f"{alias}.{k}"] = v
                        prefixed_rows.append(new_row)
                    info = FullRelation(info.columns, prefixed_rows, info.ordering)
                cell.add_content(info)
            last_cell = cell

        elif op == 'filter':
            input_cell = net.get_cell(step['input']) if 'input' in step else last_cell
            cell = net.add_cell(step.get('name', 'filter'))
            sel = step.get('selectivity', 0.33)
            if use_constraints:
                filter_constraint(input_cell, step['predicate'], cell, sel)
            else:
                filter_prop(input_cell, step['predicate'], cell, sel)
            net.operators.append(('filter', {'input': input_cell, 'output': cell}))
            last_cell = cell

        elif op == 'join':
            left = net.get_cell(step['left'])
            right = net.get_cell(step['right'])
            cell = net.add_cell(step.get('name', f"join_{len(net.cells)}"))
            sel = step.get('selectivity', 0.1)
            if use_constraints:
                join_constraint(left, right, step['condition'], cell, sel)
            else:
                join_prop(left, right, step['condition'], cell, sel)
            net.operators.append(('join', {'left': left, 'right': right, 'output': cell}))
            last_cell = cell

        elif op == 'aggregate':
            input_cell = net.get_cell(step['input']) if 'input' in step else last_cell
            cell = net.add_cell(step.get('name', 'aggregate'))
            aggregate_prop(input_cell, step['group_by'], step['agg_exprs'], cell)
            net.operators.append(('aggregate', {'input': input_cell, 'output': cell}))
            last_cell = cell

        elif op == 'project':
            input_cell = net.get_cell(step['input']) if 'input' in step else last_cell
            cell = net.add_cell(step.get('name', 'project'))
            project_prop(input_cell, step['columns'], cell)
            net.operators.append(('project', {'input': input_cell, 'output': cell}))
            last_cell = cell

        elif op == 'sort':
            input_cell = net.get_cell(step['input']) if 'input' in step else last_cell
            cell = net.add_cell(step.get('name', 'sort'))
            sort_prop(input_cell, step['order_by'], cell)
            net.operators.append(('sort', {'input': input_cell, 'output': cell}))
            last_cell = cell

        elif op == 'limit':
            input_cell = net.get_cell(step['input']) if 'input' in step else last_cell
            cell = net.add_cell(step.get('name', 'limit'))
            limit_prop(input_cell, step['n'], cell)
            net.operators.append(('limit', {'input': input_cell, 'output': cell}))
            last_cell = cell

    net.output_cell = last_cell
    return net
