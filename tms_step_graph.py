"""
TMS-based counterfactual prediction, replacing predict_effect()'s dict-copy approach.

Builds ONE propagator graph per real step. The four raw quantities that vary
per candidate action (ego_speed, front_gap, front_rel_speed, lane_error) are
written as multi-candidate Tms cells, one Supported(value, [hyp_action]) per
candidate, with all hypotheses initially kicked OUT (mandatory bootstrap step
-- see kick_all_out below). Everything downstream (ttc, front_risk, speed_slow,
speed_fast) is wired as ordinary propagators reading the CURRENTLY BELIEVED
value off those Tms cells -- since kick_out/bring_in enforce that at most one
candidate is ever believed at a time, these downstream cells never need their
own multi-candidate storage; they are just recomputed each time the worldview
toggles (alert_all_propagators reruns every registered propagator).

To evaluate "as if action X": kick_out all other hypotheses, bring_in(hyp_X),
run(), then read the derived cells. This mirrors predict_effect(X, facts)
exactly, without rebuilding a single cell or propagator per step.
"""
from propagator.cell import Cell, propagator
from propagator.nothing import nothing, nothing_p
from propagator.tms import (
    hypothetical, bring_in, kick_out, tms_query, tms_p, make_tms,
)
from propagator.supported_values import supported, supported_p
from propagator.scheduler import run as scheduler_run

DELTA_SPEED = 5


def read_believed(cell):
    """Return the plain currently-believed value of a cell, or `nothing`."""
    c = cell.content
    if nothing_p(c):
        return nothing
    if tms_p(c):
        val = tms_query(c)
        if nothing_p(val):
            return nothing
        if supported_p(val):
            return val.value
        return val  # shouldn't normally happen
    if supported_p(c):
        return c.value
    return c  # plain value, no TMS involved


def write_candidate(cell, hyp, value):
    """Write one candidate's value onto a (possibly already-Tms) cell."""
    sv = supported(value, [hyp])
    if nothing_p(cell.content):
        cell.add_content(make_tms([sv]))
    else:
        cell.add_content(sv)


class StepGraph:
    """One persistent propagator graph for one real environment step."""

    def __init__(self, facts, actions):
        self.facts = facts
        self.actions = actions
        self.hyps = {}  # action -> Hypothetical

        # Root predicted cells: hold a Tms with one candidate per action.
        self.ego_speed = Cell(name="ego_speed_pred")
        self.front_gap = Cell(name="front_gap_pred")
        self.front_rel_speed = Cell(name="front_rel_speed_pred")
        self.lane_error = Cell(name="lane_error_pred")

        # Derived cells: plain values, recomputed on each toggle.
        self.ttc = Cell(name="ttc_pred")
        self.front_risk = Cell(name="front_risk_pred")
        self.speed_slow = Cell(name="speed_slow_pred")
        self.speed_fast = Cell(name="speed_fast_pred")

        self._wire_derived_propagators()
        self._create_and_bootstrap_hypotheses()
        self._write_all_candidates()

    # ---- construction -----------------------------------------------

    def _create_and_bootstrap_hypotheses(self):
        for action in self.actions:
            hyp = hypothetical(name=f"action_{action}")
            self.hyps[action] = hyp
        # MANDATORY: fresh Hypotheticals default to believed (premise_in
        # returns True until explicitly marked out). Kick every one of them
        # out before writing any candidate values, or the very first pair
        # of conflicting Supported values will contradict on merge.
        for hyp in self.hyps.values():
            kick_out(hyp)
        scheduler_run()

    def _predicted_roots_for(self, action):
        """Mirrors the arithmetic branches of the original predict_effect."""
        facts = self.facts
        ego_speed = facts["ego_speed"]
        front_gap = facts["front_gap"]
        front_rel_speed = facts["front_rel_speed"]
        lane_error = facts.get("lane_error")

        if action == 3:
            if ego_speed is not None:
                ego_speed = ego_speed + DELTA_SPEED
        elif action == 4:
            if ego_speed is not None:
                ego_speed = ego_speed - DELTA_SPEED
            if front_rel_speed is not None:
                front_rel_speed = front_rel_speed - DELTA_SPEED
        elif action == 0:
            front_gap = None
            front_rel_speed = None
            if lane_error is not None:
                lane_error = lane_error + 1
        elif action == 2:
            front_gap = None
            front_rel_speed = None
            if lane_error is not None:
                lane_error = lane_error - 1
        # action == 1: no change

        return ego_speed, front_gap, front_rel_speed, lane_error

    def _write_all_candidates(self):
        for action in self.actions:
            hyp = self.hyps[action]
            ego_speed, front_gap, front_rel_speed, lane_error = self._predicted_roots_for(action)
            write_candidate(self.ego_speed, hyp, ego_speed)
            write_candidate(self.front_gap, hyp, front_gap)
            write_candidate(self.front_rel_speed, hyp, front_rel_speed)
            write_candidate(self.lane_error, hyp, lane_error)
        scheduler_run()

    def _wire_derived_propagators(self):
        # ttc: depends on front_gap, front_rel_speed
        def ttc_thunk():
            gap = read_believed(self.front_gap)
            rel_speed = read_believed(self.front_rel_speed)
            if nothing_p(gap):
                return
            if gap is None:
                self.ttc.add_content(float("inf"))
                return
            if nothing_p(rel_speed) or rel_speed is None:
                self.ttc.add_content(None)
                return
            if rel_speed <= 0:
                self.ttc.add_content(float("inf"))
            else:
                self.ttc.add_content(gap / rel_speed)
        propagator([self.front_gap, self.front_rel_speed], ttc_thunk)

        # front_risk: depends on ttc
        def front_risk_thunk():
            ttc = self.ttc.content
            if nothing_p(ttc):
                return
            if ttc is None:
                self.front_risk.add_content(None)
                return
            self.front_risk.add_content(ttc < 0.3)
        propagator([self.ttc], front_risk_thunk)

        # speed_slow / speed_fast: depend on ego_speed
        speed_limit = self.facts["speed_limit"]

        def speed_slow_thunk():
            speed = read_believed(self.ego_speed)
            if nothing_p(speed):
                return
            if speed is None:
                self.speed_slow.add_content(None)
                return
            self.speed_slow.add_content(speed < speed_limit - 10)
        propagator([self.ego_speed], speed_slow_thunk)

        def speed_fast_thunk():
            speed = read_believed(self.ego_speed)
            if nothing_p(speed):
                return
            if speed is None:
                self.speed_fast.add_content(None)
                return
            self.speed_fast.add_content(speed > speed_limit)
        propagator([self.ego_speed], speed_fast_thunk)

        # NOTE: ttc/front_risk cells hold a *plain* value that gets
        # overwritten each toggle via add_content -- but Cell.add_content
        # MERGES rather than overwrites, and default merge on two different
        # plain values is a contradiction! ttc/front_risk/speed_slow/speed_fast
        # must be reset before each write. See `reset_derived()` below.

    def reset_derived(self):
        """Derived plain-value cells must be cleared before each re-derivation,
        since Cell.add_content merges (equality-or-contradiction) rather than
        overwrites; a fresh nothing lets the next value be set cleanly."""
        for cell in (self.ttc, self.front_risk, self.speed_slow, self.speed_fast):
            cell.content = nothing

    # ---- per-candidate evaluation -------------------------------------

    def predict(self, action):
        """Return a dict matching the shape of the original predict_effect()."""
        for a, hyp in self.hyps.items():
            if a == action:
                continue
            kick_out(hyp)
        self.reset_derived()
        bring_in(self.hyps[action])
        scheduler_run()

        predicted = dict(self.facts)
        ego_speed = read_believed(self.ego_speed)
        front_gap = read_believed(self.front_gap)
        front_rel_speed = read_believed(self.front_rel_speed)
        lane_error = read_believed(self.lane_error)

        predicted["ego_speed"] = None if nothing_p(ego_speed) else ego_speed
        predicted["front_gap"] = None if nothing_p(front_gap) else front_gap
        predicted["front_rel_speed"] = None if nothing_p(front_rel_speed) else front_rel_speed
        predicted["lane_error"] = None if nothing_p(lane_error) else lane_error
        predicted["ttc"] = None if nothing_p(self.ttc.content) else self.ttc.content
        predicted["front_risk"] = None if nothing_p(self.front_risk.content) else self.front_risk.content
        predicted["speed_slow"] = None if nothing_p(self.speed_slow.content) else self.speed_slow.content
        predicted["speed_fast"] = None if nothing_p(self.speed_fast.content) else self.speed_fast.content
        # left_feasible/right_feasible never change with the action in the
        # original code (they depend only on left_gap/right_gap/ego_lane/
        # num_lanes), so they pass through unchanged from facts.
        return predicted
