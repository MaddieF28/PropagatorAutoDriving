"""Provenance tracking for SQL propagator networks.

Uses the propagator framework's Supported values to track tuple-level
lineage: which source rows contributed to each output row.
"""

from .relation_info import FullRelation, is_relation_info
from .network import QueryNetwork


def explain_row(row, network: QueryNetwork, output_cell=None):
    """Trace a specific output row back to its source tuples.

    Returns a list of (cell_name, matching_rows) showing provenance chain.
    """
    if output_cell is None:
        output_cell = network.output_cell

    chain = []
    _trace_backward(row, network, output_cell, chain, visited=set())
    return chain


def _trace_backward(target_row, network, cell, chain, visited):
    """Recursively trace a row backward through the network."""
    cell_name = None
    for name, c in network.cells.items():
        if c is cell:
            cell_name = name
            break

    if cell_name is None or cell_name in visited:
        return
    visited.add(cell_name)

    val = cell.content
    if not is_relation_info(val) or not hasattr(val, 'rows'):
        return

    # Find rows in this cell that match the target (by shared columns)
    matching = _find_matching_rows(target_row, val.rows)
    if matching:
        chain.append((cell_name, matching))

    # Find upstream cells (look at operators)
    for op_type, op_info in network.operators:
        if op_info.get('output') is cell:
            if op_type == 'join':
                # Trace into both sides
                _trace_backward(target_row, network, op_info['left'], chain, visited)
                _trace_backward(target_row, network, op_info['right'], chain, visited)
            elif op_type in ('filter', 'aggregate', 'project', 'sort', 'limit'):
                _trace_backward(target_row, network, op_info['input'], chain, visited)


def _find_matching_rows(target, rows):
    """Find rows that share column values with target."""
    if not rows:
        return []
    matching = []
    target_keys = set(target.keys())
    for row in rows:
        row_keys = set(row.keys())
        shared_keys = target_keys & row_keys
        if shared_keys and all(row.get(k) == target.get(k) for k in shared_keys):
            matching.append(row)
    return matching


def why_missing(expected_row, network: QueryNetwork):
    """Trace why an expected row is NOT in the output.

    Walks backward through operators, reporting at each stage
    whether matching rows exist (or were filtered out / didn't join).

    Returns list of (cell_name, op_type, status, detail).
    """
    result = []
    output_cell = network.output_cell

    # Check output
    val = output_cell.content
    if val is not None and is_relation_info(val) and hasattr(val, 'rows'):
        matching = _find_matching_rows(expected_row, val.rows)
        if matching:
            result.append(("output", None, "present", f"Found {len(matching)} matching rows"))
            return result
        result.append(("output", None, "missing", "Row not in output"))

    # Walk backward through operators
    for op_type, op_info in reversed(network.operators):
        cell = op_info.get('output') or op_info.get('input')
        if cell is None:
            continue

        cell_name = None
        for name, c in network.cells.items():
            if c is cell:
                cell_name = name
                break

        # Check the input of this operator
        input_cell = op_info.get('input')
        if input_cell is None and op_type == 'join':
            # Check both sides
            for side, side_cell in [('left', op_info.get('left')), ('right', op_info.get('right'))]:
                if side_cell and side_cell.content and hasattr(side_cell.content, 'rows'):
                    matching = _find_matching_rows(expected_row, side_cell.content.rows)
                    side_name = None
                    for n, c in network.cells.items():
                        if c is side_cell:
                            side_name = n
                            break
                    if matching:
                        result.append((side_name, 'join_input', "present",
                                       f"{side}: {len(matching)} matching rows"))
                    else:
                        result.append((side_name, 'join_input', "missing",
                                       f"{side}: no matching rows — row lost at join"))
        elif input_cell and input_cell.content and hasattr(input_cell.content, 'rows'):
            matching = _find_matching_rows(expected_row, input_cell.content.rows)
            input_name = None
            for n, c in network.cells.items():
                if c is input_cell:
                    input_name = n
                    break
            if matching:
                result.append((input_name, op_type, "present",
                               f"Before {op_type}: {len(matching)} matching rows"))
                # Row exists before this op but not after → this op removed it
                out_cell = op_info.get('output')
                if out_cell and out_cell.content and hasattr(out_cell.content, 'rows'):
                    out_matching = _find_matching_rows(expected_row, out_cell.content.rows)
                    if not out_matching:
                        result.append((cell_name, op_type, "eliminated",
                                       f"Row eliminated by {op_type}"))
            else:
                result.append((input_name, op_type, "missing",
                               f"Row already missing before {op_type}"))

    return result
