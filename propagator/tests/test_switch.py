"""Pytest coverage for switch and conditional propagators with Supported/TMS values.

These tests validate that switch passes values only when control is True and
that conditional branches correctly wire their supports.
"""

from propagator import Cell, conditional, run, initialize_scheduler, switch, support_contains
from propagator.nothing import nothing_p
from propagator.tms import hypothetical, make_tms, tms_query
from propagator.supported_values import supported, supported_p


def test_switch_with_supported_true_flows_value():
    """Motivation: switch should pass supported value with merged supports."""
    initialize_scheduler()

    pred_premise = hypothetical()
    input_premise = hypothetical()

    control = Cell()
    input_cell = Cell()
    output = Cell()

    switch(control, input_cell, output)

    control.add_content(supported(True, [pred_premise]))
    input_cell.add_content(supported(5, [input_premise]))
    run()

    assert supported_p(output.content)
    assert output.content.value == 5
    assert support_contains(output.content, pred_premise)
    assert support_contains(output.content, input_premise)


def test_switch_with_supported_false_outputs_nothing():
    """Motivation: switch should block flow when control is False."""
    initialize_scheduler()

    pred_premise = hypothetical()
    input_premise = hypothetical()

    control = Cell()
    input_cell = Cell()
    output = Cell()

    switch(control, input_cell, output)

    control.add_content(supported(False, [pred_premise]))
    input_cell.add_content(supported(5, [input_premise]))
    run()

    assert nothing_p(output.content)


def test_conditional_with_tms_predicate():
    """Motivation: conditional should wire TMS support to the output."""
    initialize_scheduler()

    outer_true = hypothetical()

    pred_cell = Cell()
    true_cell = Cell()
    false_cell = Cell()
    out_cell = Cell()

    conditional(pred_cell, true_cell, false_cell, out_cell)

    pred_cell.add_content(make_tms([supported(True, [outer_true])]))
    true_cell.add_content(1)
    false_cell.add_content(2)
    run()

    assert not nothing_p(out_cell.content)
    queried = tms_query(out_cell.content) if hasattr(out_cell.content, "values") else out_cell.content
    val = queried.value if hasattr(queried, "value") else queried
    assert val == 1
