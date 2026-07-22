"""Regression tests for the None/nothing migration (nothing_p is now strict).

These protect three things specifically:

1. The dangerous failure mode: a strongly-believed TMS value must never be
   reported as a false contradiction just because internal machinery used
   to seed its accumulator with bare None instead of the nothing sentinel
   (see tms.py's _compute_strongest_consequence). A regression here would
   be *silently wrong* (a solvable network reports UNSAT), not a crash --
   exactly the risk this migration was designed to eliminate.
2. The actual goal: a cell can now legitimately hold Python's None as real,
   comparable data, distinguishable from a cell with no content at all.
3. The ergonomic promise: plain Python functions wired through
   function_to_propagator_constructor/switch/conditional can still use
   `return None` to mean "nothing to contribute" without knowing about the
   nothing sentinel -- the coercion happens at the lifting boundary.
"""

import pytest

from propagator import (
    Cell,
    conditional,
    contradictory_p,
    initialize_scheduler,
    merge,
    nothing,
    nothing_p,
    run,
    switch,
)
from propagator.guessing_machine import binary_amb, one_of
from propagator.supported_values import Supported, supported
from propagator.tms import Tms, make_tms, strongest_consequence, tms_p, tms_query


# =============================================================================
# Tier 1: the false-contradiction risk (tms.py's _compute_strongest_consequence)
# =============================================================================

class TestNoFalseContradictionFromNothingSeed:
    """A believed TMS value must merge cleanly, not collide with nothing's
    internal fold-seed (which must be the nothing sentinel, not None)."""

    def test_strongest_consequence_of_one_believed_value_is_not_contradictory(self):
        vs = supported(42, [])  # no premises -> trivially "all premises in"
        tms = make_tms([vs])

        result = strongest_consequence(tms)

        assert not contradictory_p(result)
        assert result.value == 42

    def test_strongest_consequence_of_several_believed_values_is_not_contradictory(self):
        # Multiple *equal* values under different (empty) support sets --
        # this is the shape _compute_strongest_consequence folds over via
        # repeated merge() calls, starting from its internal seed.
        tms = make_tms([supported(7, []), supported(7, [])])

        result = strongest_consequence(tms)

        assert not contradictory_p(result)
        assert result.value == 7

    def test_tms_query_via_one_of_resolves_to_believed_value(self):
        """End-to-end: one_of + a single resolved choice must query correctly,
        not manufacture a contradiction on the very first query."""
        initialize_scheduler()
        cell = Cell()
        one_of([1, 2, 3], cell)
        run()

        # one_of's search should have settled on some value without any
        # false contradiction blocking every branch.
        assert tms_p(cell.content)
        result = tms_query(cell.content)
        assert not contradictory_p(result)
        assert not nothing_p(result)
        assert result.value in (1, 2, 3)

    def test_binary_amb_resolves_without_false_contradiction(self):
        initialize_scheduler()
        control = Cell()
        hyps = binary_amb(control)
        run()

        assert tms_p(control.content)
        result = tms_query(control.content)
        assert not contradictory_p(result)
        assert not nothing_p(result)


# =============================================================================
# Tier 2: real None survives as data
# =============================================================================

class TestNoneIsRealData:
    """None is a genuine, storable value now -- not an alias for nothing."""

    def test_add_content_none_is_stored_not_dropped(self):
        cell = Cell()
        cell.add_content(None)

        assert cell.content is None
        assert cell.content is not nothing
        assert not nothing_p(cell.content)

    def test_merging_none_with_itself_is_fine(self):
        cell = Cell()
        cell.add_content(None)
        cell.add_content(None)  # same value again -- should be a no-op merge

        assert cell.content is None

    def test_merging_none_with_a_different_value_contradicts(self):
        cell = Cell()
        cell.add_content(None)
        with pytest.raises(Exception):
            cell.add_content(42)

    def test_merge_function_treats_none_as_real_data(self):
        assert contradictory_p(merge(None, 42))
        assert merge(None, None) is None

    def test_none_survives_through_supported_wrapper(self):
        vs1 = supported(None, ["premise_a"])
        vs2 = supported(None, ["premise_b"])
        result = merge(vs1, vs2)
        assert not contradictory_p(result)
        assert result.value is None

        conflicting = supported(99, ["premise_c"])
        assert contradictory_p(merge(vs1, conflicting))

    def test_none_survives_through_tms_wrapper(self):
        tms = make_tms([supported(None, [])])
        result = strongest_consequence(tms)
        assert not contradictory_p(result)
        assert result.value is None


# =============================================================================
# Tier 3: plain functions can still `return None` to mean "nothing"
# =============================================================================

class TestPlainFunctionsStillMeanNothingWithNone:
    """The lifting boundary (lift_to_cell_contents) coerces a plain
    function's own `None` return into nothing -- ordinary Python code
    doesn't need to import or know about the sentinel."""

    def test_switch_false_leaves_plain_output_as_nothing(self):
        initialize_scheduler()
        control, input_cell, output = Cell(), Cell(), Cell()
        switch(control, input_cell, output)

        control.add_content(False)
        input_cell.add_content(5)
        run()

        assert nothing_p(output.content)

    def test_switch_false_leaves_supported_output_as_nothing(self):
        initialize_scheduler()
        control, input_cell, output = Cell(), Cell(), Cell()
        switch(control, input_cell, output)

        control.add_content(supported(False, ["pred"]))
        input_cell.add_content(supported(5, ["val"]))
        run()

        assert nothing_p(output.content)

    def test_switch_false_leaves_tms_output_as_nothing(self):
        initialize_scheduler()
        control, input_cell, output = Cell(), Cell(), Cell()
        switch(control, input_cell, output)

        control.add_content(make_tms([supported(False, [])]))
        input_cell.add_content(make_tms([supported(5, [])]))
        run()

        # A Tms whose only entries were all filtered out by the switch
        # (control False -> nothing) has no relevant values either.
        assert nothing_p(output.content) or (
            tms_p(output.content) and not output.content.values
        )

    def test_conditional_neither_branch_ready_leaves_output_as_nothing(self):
        initialize_scheduler()
        p, if_true, if_false, output = Cell(), Cell(), Cell(), Cell()
        conditional(p, if_true, if_false, output)

        p.add_content(True)
        # if_true never gets content -- output must stay nothing, not None.
        run()

        assert nothing_p(output.content)

    def test_plain_lambda_returning_none_short_circuits_cleanly(self):
        """A user-authored propagator function using the ordinary Python
        `return None` idiom for "nothing to contribute" must still work."""
        from propagator.cell import function_to_propagator_constructor

        def maybe_double(x):
            if x is None:
                return None  # ordinary Python idiom, not the sentinel
            return x * 2

        prop = function_to_propagator_constructor(maybe_double)
        input_cell, output_cell = Cell(), Cell()
        prop(input_cell, output_cell)

        initialize_scheduler()
        input_cell.add_content(None)  # a real, stored None as input
        run()

        # maybe_double(None) returned None meaning "nothing" -- the output
        # must be nothing, not a stored None.
        assert nothing_p(output_cell.content)

        # Now give it a real number: the function actually runs.
        input_cell2, output_cell2 = Cell(), Cell()
        prop2 = function_to_propagator_constructor(maybe_double)
        prop2(input_cell2, output_cell2)
        input_cell2.add_content(21)
        run()
        assert output_cell2.content == 42
