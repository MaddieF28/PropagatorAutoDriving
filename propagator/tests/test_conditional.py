"""Pytest coverage for generic conditional behavior with Supported and TMS values.

These tests guard the subtle truthiness behavior where Supported(False, ...)
is a truthy Python object but must be treated as logical False.
"""

from propagator import Cell, conditional, constant, get_generic_ignore_first, get_generic_true, support_contains
from propagator.nothing import nothing_p
from propagator.supported_values import supported, supported_p
from propagator.tms import make_tms, mark_premise_in, hypothetical, tms_p, tms_query


def test_generic_true_handles_supported_false():
    """Motivation: Supported(False, ...) must behave as logical False."""
    generic_true = get_generic_true()
    s_true = supported(True, ["premise1"])
    s_false = supported(False, ["premise2"])

    assert generic_true(s_true) is True
    assert generic_true(s_false) is False
    assert bool(s_false) is True  # highlights the Python truthiness pitfall


def test_generic_true_handles_tms_values():
    """Motivation: TMS query results must drive truthiness decisions."""
    generic_true = get_generic_true()

    p1 = hypothetical()
    mark_premise_in(p1)
    tms_true = make_tms([supported(True, [p1])])
    assert generic_true(tms_true) is True

    p2 = hypothetical()
    mark_premise_in(p2)
    tms_false = make_tms([supported(False, [p2])])
    assert generic_true(tms_false) is False

    assert generic_true(make_tms([])) is False


def test_generic_ignore_first_merges_supported_supports():
    """Motivation: predicate evidence must be preserved in results."""
    generic_ignore_first = get_generic_ignore_first()
    pred = supported(True, ["predicate"])
    val = supported(42, ["value_source"])
    result = generic_ignore_first(pred, val)

    assert supported_p(result)
    assert result.value == 42
    assert support_contains(result, "predicate")
    assert support_contains(result, "value_source")

    result2 = generic_ignore_first(pred, 100)
    assert supported_p(result2)
    assert result2.value == 100
    assert support_contains(result2, "predicate")


def test_generic_ignore_first_merges_tms_supports():
    """Motivation: TMS supports must merge across predicate and value branches."""
    generic_ignore_first = get_generic_ignore_first()
    p1 = hypothetical()
    p2 = hypothetical()
    mark_premise_in(p1)
    mark_premise_in(p2)

    tms_pred = make_tms([supported(True, [p1])])
    tms_val = make_tms([supported(42, [p2])])
    result = generic_ignore_first(tms_pred, tms_val)

    assert tms_p(result)
    queried = tms_query(result)
    assert not nothing_p(queried)
    assert queried.value == 42
    assert support_contains(queried, p1)
    assert support_contains(queried, p2)


def test_conditional_plain_bool_and_late_predicate():
    """Motivation: conditional should react as soon as predicate arrives."""
    p = Cell()
    if_true = Cell()
    if_false = Cell()
    output = Cell()
    conditional(p, if_true, if_false, output)

    constant(True, p)
    constant("yes", if_true)
    constant("no", if_false)
    assert output.content == "yes"

    p2 = Cell()
    if_true2 = Cell()
    if_false2 = Cell()
    output2 = Cell()
    conditional(p2, if_true2, if_false2, output2)
    constant("yes", if_true2)
    constant("no", if_false2)
    assert nothing_p(output2.content)
    constant(True, p2)
    assert output2.content == "yes"


def test_conditional_supported_false_is_respected():
    """Motivation: Supported(False) must select the false branch."""
    p = Cell()
    if_true = Cell()
    if_false = Cell()
    output = Cell()
    conditional(p, if_true, if_false, output)

    constant(supported(False, ["predicate_premise"]), p)
    constant(supported("yes", ["true_premise"]), if_true)
    constant(supported("no", ["false_premise"]), if_false)

    assert supported_p(output.content)
    assert output.content.value == "no"
    assert support_contains(output.content, "predicate_premise")
    assert support_contains(output.content, "false_premise")


def test_conditional_plain_predicate_supported_branch():
    """Motivation: supports should flow even when predicate is a plain bool."""
    p = Cell()
    if_true = Cell()
    if_false = Cell()
    output = Cell()
    conditional(p, if_true, if_false, output)

    constant(True, p)
    constant(supported(100, ["value_src"]), if_true)
    constant(200, if_false)

    assert supported_p(output.content)
    assert output.content.value == 100
    assert support_contains(output.content, "value_src")


def test_conditional_tms_merges_supports():
    """Motivation: TMS predicate support should be retained on output."""
    pred_premise = hypothetical()
    true_premise = hypothetical()
    mark_premise_in(pred_premise)
    mark_premise_in(true_premise)

    p = Cell()
    if_true = Cell()
    if_false = Cell()
    output = Cell()
    conditional(p, if_true, if_false, output)

    constant(make_tms([supported(True, [pred_premise])]), p)
    constant(make_tms([supported("selected", [true_premise])]), if_true)
    constant(make_tms([supported("not_selected", [hypothetical()])]), if_false)

    assert tms_p(output.content)
    result = tms_query(output.content)
    assert not nothing_p(result)
    assert result.value == "selected"
    assert support_contains(result, pred_premise)
    assert support_contains(result, true_premise)


def test_conditional_with_nothing_branch_updates():
    """Motivation: output should remain nothing until chosen branch has content."""
    p = Cell()
    if_true = Cell()
    if_false = Cell()
    output = Cell()
    conditional(p, if_true, if_false, output)

    constant(True, p)
    constant("no", if_false)
    assert nothing_p(output.content)

    constant("yes", if_true)
    assert output.content == "yes"
