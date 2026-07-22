from __future__ import annotations

from typing import Iterable, List

from .cell import Cell, propagator
from .primitives import conditional, constant, eq
from .scheduler import tag_slow
from .supported_values import supported
from .tms import (
	all_premises_in,
	bring_in,
	make_tms,
	kick_out,
	mark_premise_out,
	pairwise_union,
	premise_nogoods,
	premise_in,
	process_contradictions,
	hypothetical,
)
from .cdcl import get_cdcl_engine


# ============================================================================
# Configuration Flags
# ============================================================================

# Scheme equivalent: (define *false-premise-starts-out* #t)
# When True, the false premise starts marked "out" so the true premise
# is initially believed. This prevents immediate contradictions.
_false_premise_starts_out = True

# Scheme equivalent: (define *avoid-false-true-flips* #f)
# 
# OPTIMIZATION: When enabled, amb_choose short-circuits if either premise
# is already "in", avoiding expensive nogood filtering and worldview flips.
#
# RATIONALE from Scheme implementation:
# The amb_choose propagator can be called repeatedly as the network settles.
# When a premise is already chosen (either true or false is "in"), we don't
# need to re-evaluate the decision - the current choice is stable until
# a contradiction forces a change. This optimization:
#
# 1. REDUCES COMPUTATION: Skips the expensive premise_nogoods() and
#    all_premises_in() filtering when a choice is already made.
#
# 2. PREVENTS FLIP-FLOPS: Avoids scenarios where the algorithm might
#    oscillate between choices due to order-of-execution effects.
#
# 3. MAINTAINS INVARIANT: Ensures "some premise is in" remains true once
#    established, providing stability during propagation.
#
# Trade-off: With this disabled, amb_choose can potentially find a better
# choice if the nogood situation has changed. Scheme defaults to #f.
_avoid_false_true_flips = False

# When True, amb_choose returns immediately without making any choices.
# Used by the PROPAGATE_ONLY execution mode: propagation runs (narrowing
# domains, fixing determined cells) but the search space is never explored
# by the TMS — SMT owns the search.
_guessing_disabled = False


def set_guessing_enabled(enabled: bool) -> None:
    """Enable or disable TMS guessing (amb_choose).

    When disabled, propagation still occurs but the TMS never makes
    choices — no bring_in/kick_out of hypothetical premises. Used to
    let propagation narrow domains without triggering search.
    """
    global _guessing_disabled
    _guessing_disabled = not enabled


def is_guessing_enabled() -> bool:
    """Check if TMS guessing is currently enabled."""
    return not _guessing_disabled


def binary_amb(
	cell: Cell,
	name: str = None,
	output_cell: Cell = None,
	true_value: any = None,
	false_value: any = None,
) -> None:
	"""
	Create a binary ambiguity propagator using hypothetical premises.
	
	Creates a TMS with both True and False values supported by different
	hypothetical premises. The amb_choose propagator selects one based
	on which has no reasons against it.
	
	By default (matching Scheme), the false premise starts "out" so that
	the true premise is initially believed. This prevents immediate
	contradictions when both would otherwise be "in".
	
	Args:
		cell: The predicate cell to add the binary choice to
		name: Optional name for debugging
		output_cell: The output cell that receives a value based on this choice
		true_value: The value that flows to output_cell if true branch is taken
		false_value: The value that flows to output_cell if false branch is taken
	"""
	# Create hypotheticals with full context for debugging
	true_premise = hypothetical(
		sign='true',
		cell=cell,
		name=name,
		output_cell=output_cell,
		value_if_chosen=true_value,
	)
	false_premise = hypothetical(
		sign='false',
		cell=cell,
		name=name,
		output_cell=output_cell,
		value_if_chosen=false_value,
	)

	def amb_choose() -> None:
		# When guessing is disabled (PROPAGATE_ONLY mode), return
		# immediately. Propagation still runs (cells get values via
		# forward/backward deduction) but no choices are explored.
		if _guessing_disabled:
			return

		# OPTIMIZATION: avoid-false-true-flips
		# If either premise is already "in", skip expensive nogood filtering.
		# The current choice is stable until a contradiction forces a change.
		#
		# Scheme equivalent:
		#   (if (and *avoid-false-true-flips*
		#            (or (premise-in? true-premise)
		#                (premise-in? false-premise)))
		#       'ok  ; the some-premise-is-in invariant holds
		#       ...)
		if _avoid_false_true_flips:
			if premise_in(true_premise) or premise_in(false_premise):
				# Some premise is already "in" - no need to re-decide
				return
		
		# Filter nogoods to find reasons against each choice
		reasons_against_true = [
			nogood
			for nogood in premise_nogoods(true_premise)
			if all_premises_in(nogood)
		]
		reasons_against_false = [
			nogood
			for nogood in premise_nogoods(false_premise)
			if all_premises_in(nogood)
		]
		
		# CDCL Enhancement: Use VSIDS to choose branch order
		cdcl_engine = get_cdcl_engine()
		
		# Determine which branch to try
		if not reasons_against_true and not reasons_against_false:
			# Both branches are valid - this is a true DECISION point
			# Use VSIDS heuristic to pick the better one when CDCL is enabled
			if cdcl_engine.enabled:
				branch = cdcl_engine.choose_branch(true_premise, false_premise)
				if branch == 'true':
					kick_out(false_premise)
					bring_in(true_premise)
					cdcl_engine.make_decision(true_premise, false_premise)
				else:
					kick_out(true_premise)
					bring_in(false_premise)
					cdcl_engine.make_decision(false_premise, true_premise)
			else:
				# Default: prefer true branch
				kick_out(false_premise)
				bring_in(true_premise)
			return
		
		if not reasons_against_true:
			# Only true branch is valid - forced by nogoods against false
			# Note: In pure CDCL this would be unit propagation, but in the TMS
			# model we still track it as a decision for level management.
			kick_out(false_premise)
			bring_in(true_premise)
			if cdcl_engine.enabled:
				cdcl_engine.make_decision(true_premise, false_premise)
			return
		if not reasons_against_false:
			# Only false branch is valid - forced by nogoods against true
			kick_out(true_premise)
			bring_in(false_premise)
			if cdcl_engine.enabled:
				cdcl_engine.make_decision(false_premise, true_premise)
			return
		
		# Both branches have reasons against them - contradiction
		kick_out(true_premise)
		kick_out(false_premise)
		
		# Process contradiction: CDCL uses 1-UIP learning, otherwise standard DDB
		if cdcl_engine.enabled:
			combined_nogoods = pairwise_union(reasons_against_true, reasons_against_false)
			if combined_nogoods:
				cdcl_engine.process_conflict_cdcl(combined_nogoods[0])
		else:
			process_contradictions(pairwise_union(reasons_against_true, reasons_against_false))

	# Let's have the false premise start unbelieved.
	# This matches Scheme's *false-premise-starts-out* = #t
	if _false_premise_starts_out:
		mark_premise_out(false_premise)

	constant(
		make_tms([
			supported(True, [true_premise]),
			supported(False, [false_premise]),
		]),
		cell,
	)

	# Tag amb_choose as "slow" so it runs after all fast propagators.
	# This only affects run order with FastSlowScheduler, letting
	# deterministic constraint propagation settle before making choices.
	# Scheme equivalent: (tag-slow! amb-choose)
	tag_slow(amb_choose)
	
	propagator([cell], amb_choose)


def require(cell: Cell) -> None:
	"""Require a cell to be true."""
	constant(True, cell)


def abhor(cell: Cell) -> None:
	"""Require a cell to be false."""
	constant(False, cell)


def require_distinct(cells: Iterable[Cell]) -> None:
	"""Require all supplied cells to be distinct."""
	cells_list = list(cells)
	for i in range(len(cells_list)):
		for j in range(i + 1, len(cells_list)):
			p = Cell()
			eq(cells_list[i], cells_list[j], p)
			abhor(p)


def one_of(values: Iterable, output_cell: Cell) -> None:
	"""Select one of the supplied values into output_cell."""
	values_list = list(values)
	input_cells: List[Cell] = []
	for value in values_list:
		# Create value cells with context pointing to output
		cell = Cell(
			context=f"val:{value}",
			parent=output_cell,
			role="value"
		)
		constant(value, cell)
		input_cells.append(cell)
	one_of_the_cells(input_cells, output_cell, values_list, root_output=output_cell)


def one_of_the_cells(
	input_cells: List[Cell],
	output_cell: Cell,
	values: List = None,
	root_output: Cell = None,
) -> None:
	"""
	Select one of the supplied cells into output_cell via binary choices.
	
	Args:
		input_cells: List of cells containing candidate values
		output_cell: Cell to receive the selected value
		values: Optional list of actual values (for hypothesis debugging)
		root_output: The original output cell (for naming intermediate cells)
	"""
	# Use root_output for naming, fall back to output_cell
	root = root_output if root_output is not None else output_cell
	root_name = root.name if root.name else f"Cell@{id(root) % 10000}"
	
	if len(input_cells) == 2:
		# Create predicate cell with context
		p = Cell(
			role="predicate",
			parent=root,
			context=f"{root_name}?={values[0] if values else '?'}"
		)
		conditional(p, input_cells[0], input_cells[1], output_cell)
		# Pass value info for better hypothesis debugging
		true_val = values[0] if values else None
		false_val = values[1] if values else None
		binary_amb(p, output_cell=root, true_value=true_val, false_value=false_val)
		return
	if len(input_cells) > 2:
		# Create intermediate link cell with context
		remaining_vals = values[1:] if values else None
		link = Cell(
			role="choice",
			parent=root,
			context=f"{root_name}∈{remaining_vals}" if remaining_vals else f"{root_name}:rest"
		)
		# Create predicate cell with context  
		p = Cell(
			role="predicate",
			parent=root,
			context=f"{root_name}?={values[0] if values else '?'}"
		)
		# The false branch leads to a sub-choice among remaining values
		# Pass root_output through to maintain naming
		one_of_the_cells(input_cells[1:], link, remaining_vals, root_output=root)
		conditional(p, input_cells[0], link, output_cell)
		# True = first value, False = one of the remaining values
		true_val = values[0] if values else None
		false_val = f"one of {values[1:]}" if values else None
		binary_amb(p, output_cell=root, true_value=true_val, false_value=false_val)
		return
	raise ValueError("Inadequate choices for one_of_the_cells")