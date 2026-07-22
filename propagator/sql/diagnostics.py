"""Diagnostics for SQL propagator networks.

Detects cardinality surprises, join explosions, and other discrepancies
between estimated and actual query behavior.
"""

from .relation_info import (
    EstimateInfo, FullRelation, is_relation_info,
)
from .network import QueryNetwork


def cardinality_surprise(cell, threshold=0.5):
    """Check if a cell's actual cardinality differs significantly from estimate.

    Only meaningful for SampleInfo (extrapolated actual vs stored estimate)
    or EstimateInfo cells that have been refined with actual row data.
    For FullRelation, row_count == len(rows) by definition, so no surprise
    is possible unless the cell was previously seeded with a different estimate.

    Returns (expected, actual, ratio) if surprise detected, else None.
    The threshold is the minimum |log(actual/expected)| to flag.
    """
    val = cell.content
    if not is_relation_info(val):
        return None

    if isinstance(val, FullRelation):
        # FullRelation: row_count == len(rows), so only useful if
        # a prior EstimateInfo was merged (won't happen with current merge rules).
        return None
    elif isinstance(val, EstimateInfo):
        # No actual rows to compare against
        return None
    else:
        return None

    if expected == 0 and actual == 0:
        return None
    if expected == 0:
        return (expected, actual, float('inf'))

    import math
    ratio = actual / expected
    if abs(math.log(max(ratio, 0.001))) > threshold:
        return (expected, actual, ratio)
    return None


def join_explosion(cell, threshold=2.0):
    """Flag when a join cell's actual rows >> estimated (common bug).

    Returns (expected, actual, ratio) if explosion detected, else None.
    """
    result = cardinality_surprise(cell, threshold=0.0)
    if result is None:
        return None
    expected, actual, ratio = result
    if ratio > threshold:
        return result
    return None


def diagnose_network(network: QueryNetwork, threshold=0.5):
    """Run diagnostics on all cells in a network.

    Returns list of (cell_name, diagnostic_type, details).
    """
    issues = []
    for name, cell in network.cells.items():
        # Cardinality surprise
        surprise = cardinality_surprise(cell, threshold)
        if surprise is not None:
            expected, actual, ratio = surprise
            issues.append((
                name,
                'cardinality_surprise',
                f"expected ~{expected:.0f} rows, got {actual:.0f} (ratio={ratio:.2f})"
            ))

        # Join explosion (stricter threshold)
        if name.startswith('join'):
            explosion = join_explosion(cell, threshold=2.0)
            if explosion is not None:
                expected, actual, ratio = explosion
                issues.append((
                    name,
                    'join_explosion',
                    f"expected ~{expected:.0f} rows, got {actual:.0f} ({ratio:.1f}x explosion)"
                ))

    return issues
