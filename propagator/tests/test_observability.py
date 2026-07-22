"""Pytest coverage for hypothesis observability and contradiction introspection.

These tests verify that TMS debugging helpers (describe_nogood, etc.) work
and that contradictions are recorded correctly.
"""

import pytest

from propagator import (
    Cell,
    get_contradictions,
    get_contradiction_details,
    describe_nogood,
    initialize_scheduler,
    run,
    tms_query,
)
from propagator.guessing_machine import one_of, require_distinct


def test_contradictions_recorded_and_describable():
    """Motivation: users must be able to diagnose nogoods after search."""
    initialize_scheduler()

    x = Cell()
    y = Cell()
    x.name = "x"
    y.name = "y"

    one_of([1, 2], x)
    one_of([1, 2], y)
    run()

    require_distinct([x, y])
    run()

    contradictions = get_contradictions()
    assert isinstance(contradictions, list)

    for nogood in contradictions[:5]:
        description = describe_nogood(nogood)
        assert isinstance(description, str)

        details = get_contradiction_details(nogood)
        assert "hypotheticals" in details
        assert "hyp_info" in details
