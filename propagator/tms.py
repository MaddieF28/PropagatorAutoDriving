from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional
from itertools import product
import weakref

from .cell import listify
from .scheduler import (
    alert_propagators as _scheduler_alert,
    alert_all_propagators as _scheduler_alert_all,
    run as scheduler_run,
    is_scheduler_running,
)
from .nothing import nothing, nothing_p  # Import from nothing.py (no circular deps)
from .merge import assign_merge_operation, contradictory, merge, any_p
from .primitives import (
    generic_abs,
    generic_add,
    generic_and,
    generic_div,
    generic_eq,
    generic_gt,
    generic_gte,
    generic_lt,
    generic_lte,
    generic_mul,
    generic_not,
    generic_or,
    generic_sqrt,
    generic_square,
    generic_sub,
)
from .supported_values import (
    Supported,
    Support,
    IdentityWrapper,
    coercing,
    flat_p,
    implies,
    supported_p,
    supported_unpacking,
    to_supported,
    get_support_premises,
    _make_support,
    _support_union,
    _unwrap_support,
)


@dataclass(eq=False)
class Tms:
    """
    Represents a Truth Maintenance System (TMS).
    
    Note: eq=False makes this class use identity-based equality and hashing,
    which allows TMS objects to be used as dictionary keys. This is important
    for the consequence cache which needs to key by TMS identity.
    
    Scheme equivalent:
        (define-structure (tms (type vector) (named 'tms) (constructor %make-tms) (print-procedure #f)) values)
    """
    values: List[Any]
    
    def __hash__(self) -> int:
        return id(self)


def make_tms(values: Any) -> Tms:
    """
    Constructor for TMS objects.

    Scheme equivalent:
        (define (make-tms arg) (%make-tms (listify arg)))
    """
    return Tms(values=listify(values))


@dataclass(frozen=True, eq=False)
class Hypothetical:
    """
    Synthetic premise for hypothetical reasoning.

    Each Hypothetical is unique - identity is based on object identity,
    not structural equality. Two different Hypothetical() calls create
    two distinct premises.

    Enhanced to track:
        sign: 'true' or 'false' indicating which branch
        cell: The predicate cell this hypothesis controls
        output_cell: The output cell whose value this hypothesis determines
        value_if_chosen: The actual value that flows to output if this is believed
        name: Optional human-readable name
    
    The key insight is that binary_amb creates hypotheses on a PREDICATE cell (p),
    but what matters to the user is which VALUE flows to the OUTPUT cell.
    
    For `one_of([1, 2], x)`:
        - Hyp(sign='true', output_cell=x, value_if_chosen=1) means "x=1"
        - Hyp(sign='false', output_cell=x, value_if_chosen=2) means "x=2"
    """
    sign: str = 'unknown'
    cell: Any = None  # The predicate cell (p in conditional(p, ...))
    output_cell: Any = None  # The output cell that receives the value
    value_if_chosen: Any = None  # The actual value this hypothesis represents
    name: str = None
    
    def __hash__(self) -> int:
        # Use object identity for hashing
        return id(self)
    
    def _get_cell_description(self, cell) -> str:
        """Get the best description for a cell."""
        if cell is None:
            return None
        # Try describe() method first (for enhanced cells)
        if hasattr(cell, 'describe'):
            return cell.describe()
        # Fall back to name attribute
        if hasattr(cell, 'name') and cell.name:
            return cell.name
        # Last resort: hash ID
        return f"Cell@{id(cell) % 10000}"
    
    def __repr__(self) -> str:
        # Import here to avoid circular dependency
        from . import tms as tms_module
        status = 'in' if tms_module.premise_in(self) else 'out'
        
        # Show the value this hypothesis represents, if known
        if self.value_if_chosen is not None and self.output_cell is not None:
            out_name = self._get_cell_description(self.output_cell)
            return f"Hyp({out_name}={self.value_if_chosen}, {status})"
        
        # Fallback to showing predicate cell info with enhanced description
        cell_info = ''
        if self.cell is not None:
            cell_desc = self._get_cell_description(self.cell)
            cell_info = f', cell={cell_desc}'
        
        name_info = f', {self.name}' if self.name else ''
        return f"Hyp({self.sign}{cell_info}{name_info}, {status})"
    
    def describe(self) -> str:
        """
        Return a detailed human-readable description of what this hypothesis means.
        """
        from . import tms as tms_module
        status = 'believed' if tms_module.premise_in(self) else 'disbelieved'
        
        if self.value_if_chosen is not None and self.output_cell is not None:
            out_name = self._get_cell_description(self.output_cell)
            return f"{out_name} = {self.value_if_chosen} ({status})"
        
        if self.cell is not None:
            cell_name = getattr(self.cell, 'name', None) or f'Cell@{id(self.cell) % 10000}'
            return f"predicate {cell_name} is {self.sign} ({status})"
        
        return f"hypothesis ({status})"


def hypothetical(
    sign: str = 'unknown',
    cell: Any = None,
    name: str = None,
    output_cell: Any = None,
    value_if_chosen: Any = None,
) -> Hypothetical:
    """
    Constructor for Hypothetical objects.
    
    Args:
        sign: 'true' or 'false' indicating which branch
        cell: The predicate cell this hypothesis controls
        output_cell: The output cell whose value this hypothesis determines
        value_if_chosen: The actual value that flows to output if this is believed
        name: Optional human-readable name
    """
    hyp = Hypothetical(
        sign=sign,
        cell=cell,
        name=name,
        output_cell=output_cell,
        value_if_chosen=value_if_chosen,
    )
    # Track for introspection by solver_export
    _all_hypotheticals.add(hyp)
    return hyp


def hypothetical_p(value: Any) -> bool:
    """Predicate for Hypothetical values."""
    return isinstance(value, Hypothetical)


def tms_p(value: Any) -> bool:
    """Predicate for TMS values."""
    return isinstance(value, Tms)


def _contains_eq(items: Iterable[Any], target: Any) -> bool:
    return any(item is target for item in items)


def _lset_leq_eq(a: Iterable[Any], b: Iterable[Any]) -> bool:
    b_list = list(b)
    return all(_contains_eq(b_list, item) for item in a)


def _lset_equal_eq(a: Iterable[Any], b: Iterable[Any]) -> bool:
	return _lset_leq_eq(a, b) and _lset_leq_eq(b, a)


def _lset_difference_eq(a: Iterable[Any], b: Iterable[Any]) -> List[Any]:
    b_list = list(b)
    return [item for item in a if not _contains_eq(b_list, item)]


def _lset_adjoin_eq(items: Iterable[Any], item: Any) -> List[Any]:
    result = list(items)
    if not _contains_eq(result, item):
        result.append(item)
    return result


def _lset_union_eq(*lists: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    for lst in lists:
        for item in lst:
            if not _contains_eq(result, item):
                result.append(item)
    return result


# Premise bookkeeping (global worldview)
_premise_outness: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()
_premise_outness_strong: dict[int, tuple[Any, bool]] = {}
# For hashable but non-weakrefable premises (strings, numbers, tuples)
_premise_outness_hashable: dict[Any, bool] = {}

# Worldview number for cache invalidation
# Incremented whenever a premise is kicked out or brought in
# Scheme equivalent: *worldview-number*
_worldview_number: int = 0

# Consequence cache: maps TMS -> (worldview_number, consequence)
# Valid only if the stored worldview_number matches the current one.
# Scheme equivalent: *consequence-cache*
#
# Implementation note: We use WeakKeyDictionary with TMS objects as keys rather than
# a regular dict with id(tms) as keys for two important reasons:
#
# 1. ID recycling safety: Python's id() returns the memory address of an object, which
#    can be reused after the object is garbage collected. Using id() as a key could
#    cause a new TMS object to match a stale cache entry from a dead TMS that happened
#    to have the same memory address. WeakKeyDictionary avoids this because it uses
#    the actual object as the key - if the TMS dies, the entry is removed automatically.
#
# 2. Memory efficiency: During AMB search, many TMS objects are created and discarded.
#    WeakKeyDictionary automatically cleans up cache entries for dead TMS objects,
#    preventing memory leaks during long searches.
#
# To make TMS usable as a dict key, we set eq=False on the dataclass, which gives it
# identity-based __eq__ and __hash__ (using id()).
# Global registry of all hypotheticals (for introspection by solver_export)
_all_hypotheticals: "weakref.WeakSet[Hypothetical]" = weakref.WeakSet()

_consequence_cache: "weakref.WeakKeyDictionary[Tms, tuple[int, Any]]" = weakref.WeakKeyDictionary()


# CDCL is an optional enhancement layered on top of the TMS (cdcl.py imports
# from tms.py, not the other way around) -- but the TMS still needs to reset
# CDCL state on initialize_tms() and hand off nogood processing to CDCL when
# it's enabled. Rather than tms.py importing cdcl.py (which would make the
# dependency circular, since cdcl.py already imports tms.py), cdcl.py
# registers its own hooks here at import time. Mirrors register_tms_initializer
# in scheduler.py, which breaks the analogous scheduler<->tms cycle the same way.
_cdcl_reset: Optional[Callable[[], None]] = None
_cdcl_enabled_check: Optional[Callable[[], bool]] = None
_cdcl_process_conflict: Optional[Callable[[List[Any]], None]] = None


def register_cdcl_handlers(
    reset: Callable[[], None],
    enabled_check: Callable[[], bool],
    process_conflict: Callable[[List[Any]], None],
) -> None:
    """Called by cdcl.py at import time; see the module comment above."""
    global _cdcl_reset, _cdcl_enabled_check, _cdcl_process_conflict
    _cdcl_reset = reset
    _cdcl_enabled_check = enabled_check
    _cdcl_process_conflict = process_conflict


def initialize_tms() -> str:
    """
    Clear TMS state for a fresh propagator network.
    
    This is called as part of initialize_scheduler() and resets:
    - consequence cache (for performance, must be invalidated)
    - worldview number (resets cache validity)
    - premise_outness (which premises are believed)
    - premise_nogoods (learned conflict clauses)
    - number_of_calls_to_fail counter
    - contradictions_history (for debugging)
    
    Note: The Scheme implementation technically doesn't clear all this state,
    but it runs each test in a fresh process. For Python's pytest where tests
    share process state, we need to clear everything for test isolation.
    
    Scheme equivalent (from truth-maintenance.scm):
        (define initialize-scheduler
          (let ((initialize-scheduler initialize-scheduler))
            (lambda ()
              (initialize-scheduler)
              (set! *consequence-cache* (make-eq-hash-table)))))
    """
    global _worldview_number, _consequence_cache
    global _premise_outness, _premise_outness_strong, _premise_outness_hashable
    global number_of_calls_to_fail, last_nogood, contradictions_history
    global _all_hypotheticals
    
    # Reset worldview number and cache
    _worldview_number = 0
    _consequence_cache = weakref.WeakKeyDictionary()
    
    # Reset premise bookkeeping - important for test isolation
    _premise_outness.clear()
    _premise_outness_strong.clear()
    _premise_outness_hashable.clear()
    
    # Reset hypotheticals registry
    _all_hypotheticals = weakref.WeakSet()
    
    # Reset nogoods - important for test isolation
    try:
        global _premise_nogoods, _premise_nogoods_strong, _premise_nogoods_hashable
        _premise_nogoods.clear()
        _premise_nogoods_strong.clear()
        _premise_nogoods_hashable.clear()
    except NameError:
        pass  # Not yet defined during module load
    
    # Reset contradiction tracking (for debugging/observability)
    number_of_calls_to_fail = 0
    last_nogood = None
    contradictions_history = []
    
    # Reset CDCL state if cdcl.py has registered itself (see
    # register_cdcl_handlers above -- cdcl.py may never have been imported
    # at all in a process that doesn't use it, hence the None check rather
    # than an ImportError guard).
    if _cdcl_reset is not None:
        _cdcl_reset()

    return 'ok'


def get_worldview_number() -> int:
    """
    Get the current worldview number.
    
    This increments each time a premise is kicked out or brought in.
    Useful for debugging and understanding cache behavior.
    
    Returns:
        The current worldview number (0 after initialization)
    """
    return _worldview_number


def _weakrefable(obj: Any) -> bool:
    try:
        weakref.ref(obj)
        return True
    except TypeError:
        return False


def _hashable(obj: Any) -> bool:
    """Check if an object can be used as a dict key (is hashable)."""
    try:
        hash(obj)
        return True
    except TypeError:
        return False


def _premise_outness_get(premise: Any, default: bool) -> bool:
    if _weakrefable(premise):
        return _premise_outness.get(premise, default)
    # For hashable but non-weakrefable (strings, numbers, tuples)
    if _hashable(premise):
        return _premise_outness_hashable.get(premise, default)
    # Fallback for unhashable, non-weakrefable (shouldn't happen in practice)
    entry = _premise_outness_strong.get(id(premise))
    if entry is None:
        return default
    obj, value = entry
    return value if obj is premise else default


def _premise_outness_set(premise: Any, value: bool) -> None:
    if _weakrefable(premise):
        _premise_outness[premise] = value
        return
    # For hashable but non-weakrefable (strings, numbers, tuples)
    if _hashable(premise):
        _premise_outness_hashable[premise] = value
        return
    # Fallback for unhashable, non-weakrefable
    _premise_outness_strong[id(premise)] = (premise, value)


def _premise_outness_del(premise: Any) -> None:
    if _weakrefable(premise):
        _premise_outness.pop(premise, None)
        return
    # For hashable but non-weakrefable (strings, numbers, tuples)
    if _hashable(premise):
        _premise_outness_hashable.pop(premise, None)
        return
    # Fallback for unhashable, non-weakrefable
    entry = _premise_outness_strong.get(id(premise))
    if entry is not None and entry[0] is premise:
        _premise_outness_strong.pop(id(premise), None)


def premise_in(premise: Any) -> bool:
    """
    Check if a premise is in the global worldview.

    Scheme equivalent:
        (define (premise-in? premise)
          (not (hash-table/get *premise-outness* premise #f)))
    """
    return not _premise_outness_get(premise, False)


def mark_premise_in(premise: Any) -> None:
    """
    Mark a premise as in the global worldview.

    Scheme equivalent:
        (define (mark-premise-in! premise)
          (hash-table/remove! *premise-outness* premise))
    """
    _premise_outness_del(premise)


def mark_premise_out(premise: Any) -> None:
    """
    Mark a premise as out of the global worldview.

    Scheme equivalent:
        (define (mark-premise-out! premise)
          (hash-table/put! *premise-outness* premise #t))
    """
    _premise_outness_set(premise, True)


_premise_nogoods: "weakref.WeakKeyDictionary[Any, List[Any]]" = weakref.WeakKeyDictionary()
_premise_nogoods_strong: dict[int, tuple[Any, List[Any]]] = {}
_premise_nogoods_hashable: dict[Any, List[Any]] = {}


def _premise_nogoods_get(premise: Any, default: List[Any]) -> List[Any]:
    if _weakrefable(premise):
        return _premise_nogoods.get(premise, default)
    if _hashable(premise):
        return _premise_nogoods_hashable.get(premise, default)
    entry = _premise_nogoods_strong.get(id(premise))
    if entry is None:
        return default
    obj, value = entry
    return value if obj is premise else default


def _premise_nogoods_set(premise: Any, value: List[Any]) -> None:
    if _weakrefable(premise):
        _premise_nogoods[premise] = value
        return
    if _hashable(premise):
        _premise_nogoods_hashable[premise] = value
        return
    _premise_nogoods_strong[id(premise)] = (premise, value)


def premise_nogoods(premise: Any) -> List[Any]:
    """
    Get nogoods associated with a premise.

    Scheme equivalent:
        (define (premise-nogoods premise)
          (hash-table/get *premise-nogoods* premise '()))
    """
    return _premise_nogoods_get(premise, [])


def set_premise_nogoods(premise: Any, nogoods: List[Any]) -> None:
    """
    Set nogoods associated with a premise.

    Scheme equivalent:
        (define (set-premise-nogoods! premise nogoods)
          (hash-table/put! *premise-nogoods* premise nogoods))
    """
    _premise_nogoods_set(premise, nogoods)


number_of_calls_to_fail: int = 0
last_nogood: List[Any] | None = None
contradictions_history: List[List[Any]] = []

# Contradiction verbosity flag (like Scheme's *contradiction-wallp*)
_contradiction_verbose = False


def get_number_of_calls_to_fail() -> int:
    """
    Get the number of times process_nogood was called.
    
    Use this function instead of importing `number_of_calls_to_fail` directly,
    because the global variable is reset by `initialize_scheduler()` and a
    direct import captures the value at import time.
    """
    return number_of_calls_to_fail


def get_last_nogood() -> List[Any] | None:
    """
    Get the most recent nogood (contradiction) encountered.
    
    Use this function instead of importing `last_nogood` directly,
    because the global variable is reset by `initialize_scheduler()`.
    """
    return last_nogood


def set_contradiction_verbose(enabled: bool) -> None:
    """Enable/disable verbose contradiction printing (like Scheme's *contradiction-wallp*)."""
    global _contradiction_verbose
    _contradiction_verbose = enabled


class TmsContradiction(Exception):
    """Raised when a nogood is detected in the current worldview."""

    def __init__(self, nogood: List[Any]):
        super().__init__(f"contradiction {nogood}")
        self.nogood = nogood


def process_nogood(nogood: List[Any]) -> None:
    """
    Abort the process for a contradiction.

    Scheme equivalent:
        (define (process-nogood! nogood) (abort-process '(contradiction ,nogood)))
    
    When CDCL is enabled, uses CDCL-style conflict analysis and backjumping.
    """
    global number_of_calls_to_fail, last_nogood, contradictions_history, _contradiction_verbose
    number_of_calls_to_fail += 1
    last_nogood = list(nogood)
    contradictions_history.append(list(nogood))
    
    if _contradiction_verbose:
        print(f"[NOGOOD] {describe_nogood(nogood)}")
    
    # Check if CDCL is enabled (via cdcl.py's registered hooks, if it's
    # been imported at all) and use CDCL processing if so.
    if _cdcl_enabled_check is not None and _cdcl_enabled_check():
        _cdcl_process_conflict(nogood)
        return None

    # Fall back to standard processing
    process_one_contradiction(nogood)
    return None


def get_contradictions() -> List[List[Any]]:
    """Return a copy of the contradiction history."""
    return [list(nogood) for nogood in contradictions_history]


def describe_nogood(nogood: List[Any]) -> str:
    """
    Produce a human-readable description of a nogood set.
    
    Shows hypothetical details (sign, cell, status) when available.
    """
    parts = []
    for premise in nogood:
        if hypothetical_p(premise):
            # Enhanced hypothetical with sign/cell info
            parts.append(repr(premise))
        else:
            parts.append(str(premise))
    return "{" + ", ".join(parts) + "}"


def describe_last_contradiction() -> str:
    """Describe the most recent contradiction in human-readable form."""
    if not contradictions_history:
        return "No contradictions recorded"
    return describe_nogood(contradictions_history[-1])


def get_contradiction_details(nogood: List[Any]) -> dict:
    """
    Get structured details about a contradiction (nogood set).
    
    Returns:
        dict with:
        - 'nogood': the raw nogood list
        - 'hypotheticals': list of Hypothetical premises
        - 'grounded': list of non-hypothetical premises
        - 'hyp_info': detailed info about each hypothetical
        - 'explanation': human-readable explanation of the contradiction
    """
    def get_cell_desc(cell):
        """Get the best description for a cell."""
        if cell is None:
            return None
        if hasattr(cell, 'describe'):
            return cell.describe()
        if hasattr(cell, 'name') and cell.name:
            return cell.name
        return f"Cell@{id(cell) % 10000}"
    
    hyps = [p for p in nogood if hypothetical_p(p)]
    grounded = [p for p in nogood if not hypothetical_p(p)]
    
    hyp_info = []
    value_assignments = []
    for h in hyps:
        info = {
            'premise': h,
            'sign': getattr(h, 'sign', 'unknown'),
            'cell': getattr(h, 'cell', None),
            'name': getattr(h, 'name', None),
            'output_cell': getattr(h, 'output_cell', None),
            'value_if_chosen': getattr(h, 'value_if_chosen', None),
            'in': premise_in(h),
        }
        # Get cell descriptions using enhanced describe() method
        if info['cell'] is not None:
            info['cell_name'] = get_cell_desc(info['cell'])
        if info['output_cell'] is not None:
            info['output_cell_name'] = get_cell_desc(info['output_cell'])
        
        # Build value assignment string if we have the info
        if info['value_if_chosen'] is not None and info['output_cell'] is not None:
            cell_name = info.get('output_cell_name', '?')
            value_assignments.append(f"{cell_name}={info['value_if_chosen']}")
        hyp_info.append(info)
    
    # Build explanation
    if value_assignments:
        explanation = f"Contradiction when: {', '.join(value_assignments)}"
    else:
        explanation = f"Contradiction involving {len(hyps)} hypothetical(s)"
    
    return {
        'nogood': nogood,
        'hypotheticals': hyps,
        'grounded': grounded,
        'hyp_info': hyp_info,
        'explanation': explanation,
    }


def explain_contradictions(max_count: int = 10) -> str:
    """
    Generate a human-readable explanation of recent contradictions.
    
    This is the main entry point for understanding why constraints failed.
    Shows which value assignments led to contradictions.
    
    Args:
        max_count: Maximum number of contradictions to explain (default 10)
    
    Returns:
        Multi-line string explaining the contradictions
    """
    if not contradictions_history:
        return "No contradictions detected."
    
    lines = [f"Found {len(contradictions_history)} contradiction(s):\n"]
    
    for i, nogood in enumerate(contradictions_history[:max_count]):
        details = get_contradiction_details(nogood)
        lines.append(f"  {i+1}. {details['explanation']}")
        
        # Show the value assignments involved
        for info in details['hyp_info']:
            if info.get('value_if_chosen') is not None:
                out_name = info.get('output_cell_name', '?')
                status = 'believed' if info['in'] else 'rejected'
                lines.append(f"      - {out_name}={info['value_if_chosen']} was {status}")
    
    if len(contradictions_history) > max_count:
        lines.append(f"\n  ... and {len(contradictions_history) - max_count} more")
    
    return "\n".join(lines)


def check_consistent(vs: Any) -> None:
    """
    Check whether a supported value is contradictory and process its nogood.
    """
    if contradictory(vs):
        if supported_p(vs):
            # Unwrap support to get list of premises for nogood processing
            process_nogood(get_support_premises(vs))
        else:
            process_nogood([])


def pairwise_union(a: List[List[Any]], b: List[List[Any]]) -> List[List[Any]]:
    """Compute all eq-unions of pairs from two nogood lists."""
    result: List[List[Any]] = []
    for left in a:
        for right in b:
            merged = _lset_union_eq(left, right)
            if not any(_lset_equal_eq(merged, existing) for existing in result):
                result.append(merged)
    return result


def assimilate_nogood(premise: Any, new_nogood: List[Any]) -> None:
    """
    Teach a premise about a nogood, removing itself and subsumed items.
    """
    item = [p for p in new_nogood if p is not premise]
    current = premise_nogoods(premise)
    if any(_lset_leq_eq(old, item) for old in current):
        return None
    subsumed = [old for old in current if _lset_leq_eq(item, old)]
    updated = _lset_adjoin_eq(_lset_difference_eq(current, subsumed), item)
    set_premise_nogoods(premise, updated)
    return None


def process_one_contradiction(nogood: List[Any]) -> None:
    """
    Process a single nogood by kicking out a hypothetical if available.
    """
    hyps = [p for p in nogood if hypothetical_p(p)]
    if len(hyps) == 0:
        return None
    kick_out(hyps[0])
    for premise in nogood:
        assimilate_nogood(premise, nogood)
    return None


def process_contradictions(nogoods: List[List[Any]]) -> None:
    """
    Choose and process one contradiction with fewest hypotheticals.
    """
    if len(nogoods) == 0:
        return None
    def hypotheticals_count(nogood: List[Any]) -> int:
        return len([p for p in nogood if hypothetical_p(p)])
    best = sorted(nogoods, key=hypotheticals_count)[0]
    process_one_contradiction(best)
    return None


def subsumes(vs1: Supported, vs2: Supported) -> bool:
    """
    Return True if vs1 subsumes vs2.

    Scheme equivalent:
        (define (subsumes? v&s1 v&s2)
          (and (implies? (v&s-value v&s1) (v&s-value v&s2))
               (lset<= eq? (v&s-support v&s1) (v&s-support v&s2))))
    
    Performance: O(|vs1.support|) with frozenset vs O(n²) with lists.
    """
    return implies(vs1.value, vs2.value) and vs1.support.issubset(vs2.support)


def tms_assimilate_one(tms: Tms, vs: Supported) -> Tms:
    """
    Assimilate a single v&s into a TMS without deduction.

    (define (tms-assimilate-one tms v&s) 
        (if (any (lambda (old-v&s) (subsumes? old-v&s v&s)) 
            (tms-values tms)) 
            tms 
            (let ((subsumed 
                (filter (lambda (old-v&s) (subsumes? v&s old-v&s)) (tms-values tms)))) 
                (make-tms (lset-adjoin eq? 
                    (lset-difference eq? (tms-values tms) subsumed) v&s)))))
    """
    if any(subsumes(old_vs, vs) for old_vs in tms.values):
        return tms

    subsumed = [old_vs for old_vs in tms.values if subsumes(vs, old_vs)]
    remaining = _lset_difference_eq(tms.values, subsumed)
    new_values = _lset_adjoin_eq(remaining, vs)
    return make_tms(new_values)


def tms_assimilate(tms: Tms, stuff: Any) -> Tms:
    """
    Assimilate v&s or TMS into a TMS without deduction.

    Scheme equivalent:
        (define (tms-assimilate tms stuff) 
            (cond ((nothing? stuff) tms) 
                ((v&s? stuff) (tms-assimilate-one tms stuff)) ((tms? stuff) 
                 (fold-left tms-assimilate-one 
                            tms 
                            (tms-values stuff))) 
                (else (error "This should never happen"))))
    """
    if nothing_p(stuff):
        return tms
    if supported_p(stuff):
        return tms_assimilate_one(tms, stuff)
    if tms_p(stuff):
        result = tms
        for vs in stuff.values:
            result = tms_assimilate_one(result, vs)
        return result
    raise ValueError("tms_assimilate expected Supported or Tms")


def all_premises_in(thing: Any) -> bool:
    """
    Check whether all premises referenced by thing are in the worldview.

    Scheme equivalent:
        (define (all-premises-in? thing)
          (if (v&s? thing)
              (all-premises-in? (v&s-support thing))
              (every premise-in? thing)))
    
    Handles both:
    - Support (frozenset[IdentityWrapper]) from Supported.support
    - Plain lists of premises (for nogoods)
    """
    if supported_p(thing):
        return all_premises_in(thing.support)
    if thing is None:
        return True
    # Handle Support (frozenset of IdentityWrapper)
    if isinstance(thing, frozenset):
        return all(premise_in(w.obj) for w in thing)
    # Handle plain list of premises (for nogoods)
    return all(premise_in(premise) for premise in thing)


def _cached_consequence(tms: Tms) -> Any | None:
    """
    Check if we have a valid cached consequence for this TMS.
    
    Returns the cached consequence if valid, None otherwise.
    
    Uses WeakKeyDictionary keyed by the TMS object itself, so:
    - No ID recycling issues (object identity is the key)
    - Entries automatically removed when TMS is garbage collected
    - Only need to check worldview, not object identity
    
    Scheme equivalent:
        (define (cached-consequence tms)
          (let ((answer (hash-table/get *consequence-cache* tms #f)))
            (and answer
                 (= (car answer) *worldview-number*)
                 (cdr answer))))
    """
    cached = _consequence_cache.get(tms)
    if cached is not None:
        cached_worldview, cached_result = cached
        if cached_worldview == _worldview_number:
            return cached_result
    return None


def _cache_consequence(tms: Tms, consequence: Any) -> Any:
    """
    Cache a consequence for the current worldview.
    
    Uses WeakKeyDictionary so entries are automatically removed when TMS is GC'd.
    No need to store TMS reference separately since TMS is the key.
    
    Scheme equivalent:
        (define (cache-consequence! tms consequence)
          (hash-table/put! *consequence-cache* tms
            (cons *worldview-number* (effectful-info (->effectful consequence))))
          consequence)
    """
    _consequence_cache[tms] = (_worldview_number, consequence)
    return consequence


def _compute_strongest_consequence(tms: Tms) -> Any:
    """
    Actually compute the strongest consequence (no caching).
    
    Scheme equivalent:
        (define (compute-strongest-consequence tms)
          (let ((relevant-v&ss
                 (filter v&s-believed? (tms-values tms))))
            (merge* relevant-v&ss)))
    """
    relevant_vss = [vs for vs in tms.values if all_premises_in(vs)]
    result = nothing
    for vs in relevant_vss:
        result = merge(result, vs)
    return result


def strongest_consequence(tms: Tms) -> Any:
    """
    Compute the most informative consequence for the current worldview.
    
    Uses caching based on worldview number to avoid recomputation when
    the set of believed premises hasn't changed.
    
    Scheme equivalent:
        (define (strongest-consequence tms)
          (let ((cached (cached-consequence tms)))
            (or cached
                (cache-consequence! tms (compute-strongest-consequence tms)))))
    """
    cached = _cached_consequence(tms)
    if cached is not None:
        return cached
    return _cache_consequence(tms, _compute_strongest_consequence(tms))



def tms_merge(tms1: Tms, tms2: Tms) -> Tms:
    """
    Merge two TMSes, deducing only relevant consequences.
    """
    candidate = tms_assimilate(tms1, tms2)
    consequence = strongest_consequence(candidate)
    check_consistent(consequence)
    # full_consequence = strongest_consequence(candidate)
    return tms_assimilate(candidate, consequence)
    # return tms_assimilate(tms_assimilate(candidate, consequence), full_consequence)


def tms_query(tms: Tms) -> Any:
    """
    Interpret a TMS in the current worldview, assimilating new consequences.
    (define (tms-query tms) 
        (let ((answer (strongest-consequence tms))) 
            (let ((better-tms (tms-assimilate tms answer))) 
                (if (not (eq? tms better-tms)) 
                    (set-tms-values! tms (tms-values better-tms))) (check-consistent! answer) 
                    answer)))
    """
    answer = strongest_consequence(tms)
    better_tms = tms_assimilate(tms, answer)
    if better_tms is not tms:
        tms.values = better_tms.values
    check_consistent(answer)
    return answer


def tms_contradiction_info(tms: Tms) -> dict | None:
    """
    Get detailed contradiction information for a TMS, if contradictory.
    
    This allows observing the specific condition/hypothesis that caused
    a contradiction in the TMS.
    
    Returns:
        None if TMS is not contradictory in current worldview
        dict with:
        - 'is_contradictory': True
        - 'conflicting_values': list of believed Supported values that conflict
        - 'nogood': the combined support set (premises causing contradiction)
        - 'details': structured breakdown of hypotheticals vs grounded premises
    """
    answer = strongest_consequence(tms)
    
    if not contradictory(answer):
        return None
    
    # Find believed values (all premises in their support are believed)
    believed = [v for v in tms.values if supported_p(v) and all_premises_in(v)]
    
    # Get the nogood (unwrap Support to list of premises)
    nogood = get_support_premises(answer) if supported_p(answer) else []
    
    return {
        'is_contradictory': True,
        'conflicting_values': believed,
        'nogood': nogood,
        'details': get_contradiction_details(nogood),
    }


def kick_out(premise: Any) -> None:
    """
    Mark a premise out and schedule propagators if worldview changed.
    
    Note: This only schedules propagators, it does NOT run them immediately.
    The caller is responsible for calling run() when ready.
    
    Scheme equivalent:
        (define (kick-out! premise)
          (if (premise-in? premise)
              (begin
                (set! *worldview-number* (+ *worldview-number* 1))
                (alert-all-propagators!)))
          (mark-premise-out! premise))
    """
    global _worldview_number
    if premise_in(premise):
        # Increment worldview number to invalidate consequence cache
        _worldview_number += 1
        # Only schedule, don't run - the outer loop will run
        _scheduler_alert_all()
    mark_premise_out(premise)


def bring_in(premise: Any) -> None:
    """
    Mark a premise in and schedule propagators if worldview changed.
    
    Note: This only schedules propagators, it does NOT run them immediately.
    The caller is responsible for calling run() when ready.
    
    Scheme equivalent:
        (define (bring-in! premise)
          (if (not (premise-in? premise))
              (begin
                (set! *worldview-number* (+ *worldview-number* 1))
                (alert-all-propagators!)))
          (mark-premise-in! premise))
    """
    global _worldview_number
    if not premise_in(premise):
        # Increment worldview number to invalidate consequence cache
        _worldview_number += 1
        # Only schedule, don't run - the outer loop will run
        _scheduler_alert_all()
    mark_premise_in(premise)


def to_tms(value: Any) -> Tms:
    """Coerce a value to a TMS."""
    if nothing_p(value):
        return make_tms([])
    if tms_p(value):
        return value
    return make_tms([to_supported(value)])


def tms_unpacking(f):
    """
    Lift a function to operate on TMSes via all relevant v&ss.

    Scheme equivalent:
        (define (tms-unpacking f)
          (lambda args
            (let ((relevant-information (map tms-query args)))
              (if (any nothing? relevant-information)
                  nothing
                  (make-tms (list (apply f relevant-information)))))))
    """
    def wrapper(*args):
        relevant_lists = [list(arg.values) for arg in args]
        if any(len(lst) == 0 for lst in relevant_lists):
            return nothing
        result = make_tms([])
        for combo in product(*relevant_lists):
            if any(contradictory(vs.value) for vs in combo):
                continue
            result = tms_assimilate(result, f(*combo))
        return result
    return wrapper


def full_tms_unpacking(f):
    """
    Lift a function to operate on TMSes and merge supports.

    Scheme equivalent:
        (define (full-tms-unpacking f) (tms-unpacking (v&s-unpacking f)))
    """
    return tms_unpacking(supported_unpacking(f))


# Register merge handlers for TMS
assign_merge_operation(tms_merge, tms_p, tms_p)
assign_merge_operation(coercing(to_tms, tms_merge), tms_p, supported_p)
assign_merge_operation(coercing(to_tms, tms_merge), supported_p, tms_p)
assign_merge_operation(coercing(to_tms, tms_merge), tms_p, flat_p)
assign_merge_operation(coercing(to_tms, tms_merge), flat_p, tms_p)
assign_merge_operation(lambda tms, _: tms, tms_p, nothing_p)
assign_merge_operation(lambda _, tms: tms, nothing_p, tms_p)


# Register generic operator support for TMS
_binary_ops = [
    (generic_add, '+'),
    (generic_sub, '-'),
    (generic_mul, '*'),
    (generic_div, '/'),
    (generic_eq, '='),
    (generic_lt, '<'),
    (generic_gt, '>'),
    (generic_lte, '<='),
    (generic_gte, '>='),
    (generic_and, 'and'),
    (generic_or, 'or'),
]

for op, _ in _binary_ops:
    op.assign_operation(full_tms_unpacking(op), tms_p, tms_p)
    op.assign_operation(coercing(to_tms, op), tms_p, supported_p)
    op.assign_operation(coercing(to_tms, op), supported_p, tms_p)
    op.assign_operation(coercing(to_tms, op), tms_p, flat_p)
    op.assign_operation(coercing(to_tms, op), flat_p, tms_p)

_unary_ops = [
    (generic_abs, 'abs'),
    (generic_square, 'square'),
    (generic_sqrt, 'sqrt'),
    (generic_not, 'not'),
]

for op, _ in _unary_ops:
    op.assign_operation(full_tms_unpacking(op), tms_p)


# Register generic_switch for TMS values
# The switch function is special: it returns None (nothing) when control is False,
# and we need to filter those out rather than creating Supported(None, ...).
from .primitives import generic_switch

def _tms_switch(control_tms: Tms, input_tms: Tms) -> Any:
    """
    Switch for TMS values.
    
    For each combination of control and input values:
    - If control is True: include input with merged supports
    - If control is False: skip (don't include anything)
    
    This is different from full_tms_unpacking because we filter out None results
    rather than wrapping them in Supported.
    """
    result = make_tms([])
    
    for control_vs in control_tms.values:
        # Extract the control value
        if supported_p(control_vs):
            control_val = control_vs.value
            control_support = control_vs.support
        else:
            control_val = control_vs
            control_support = frozenset()  # Empty Support
        
        # If control is False, don't contribute anything
        if not control_val:
            continue
        
        # Control is True - add all input values with merged supports
        for input_vs in input_tms.values:
            if supported_p(input_vs):
                input_val = input_vs.value
                input_support = input_vs.support
            else:
                input_val = input_vs
                input_support = frozenset()  # Empty Support
            
            # Merge supports from control and input (both are Support frozensets)
            merged_support = control_support | input_support
            new_vs = Supported(input_val, merged_support)
            result = tms_assimilate(result, new_vs)
    
    # If result is empty, return nothing
    if not result.values:
        return nothing
    
    return result

# Register for all TMS combinations
generic_switch.assign_operation(_tms_switch, tms_p, tms_p)
generic_switch.assign_operation(
    lambda c, i: _tms_switch(to_tms(c), to_tms(i)),
    tms_p, supported_p
)
generic_switch.assign_operation(
    lambda c, i: _tms_switch(to_tms(c), to_tms(i)),
    supported_p, tms_p
)
generic_switch.assign_operation(
    lambda c, i: _tms_switch(to_tms(c), to_tms(i)),
    tms_p, flat_p
)
generic_switch.assign_operation(
    lambda c, i: _tms_switch(to_tms(c), to_tms(i)),
    flat_p, tms_p
)
generic_switch.assign_operation(
    lambda c, i: _tms_switch(to_tms(c), to_tms(i)),
    tms_p, any_p
)
generic_switch.assign_operation(
    lambda c, i: _tms_switch(to_tms(c), to_tms(i)),
    any_p, tms_p
)


# ============================================================================
# Helper functions for solver_export introspection
# ============================================================================

def get_all_hypotheticals() -> list:
    """
    Return list of all Hypothetical objects created in the current network.
    
    This is primarily used by solver_export/auto_extract.py for introspecting
    existing propagator networks.
    
    Returns:
        List of Hypothetical objects that were created via one_of/binary_amb
    """
    return list(_all_hypotheticals)


def get_all_nogoods() -> list:
    """
    Return all nogoods from all premises in the current network.
    
    Each nogood is a frozenset of premises that are mutually contradictory.
    This is primarily used by solver_export/auto_extract.py for extracting
    learned conflict clauses to pass to external solvers.
    
    Returns:
        List of frozensets, each representing a nogood (conflict clause)
    """
    all_nogoods = []
    
    # Collect from weak references
    for premise, nogoods in _premise_nogoods.items():
        all_nogoods.extend(nogoods)
    
    # Collect from strong references
    for id_key, (premise, nogoods) in _premise_nogoods_strong.items():
        all_nogoods.extend(nogoods)
    
    # Remove duplicates (nogoods are frozensets so hashable)
    seen = set()
    unique_nogoods = []
    for ng in all_nogoods:
        ng_tuple = tuple(sorted(id(p) for p in ng))  # Convert to hashable
        if ng_tuple not in seen:
            seen.add(ng_tuple)
            unique_nogoods.append(ng)
    
    return unique_nogoods


# Register the TMS initializer with the scheduler
# This ensures that initialize_scheduler() also clears TMS state
from .scheduler import register_tms_initializer
register_tms_initializer(initialize_tms)