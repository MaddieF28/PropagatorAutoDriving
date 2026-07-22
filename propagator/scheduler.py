"""
Scheduler module for the propagators library.

This scheduler maintains a queue of propagators (jobs) that need to be run.
Each job is a thunk (callable). Jobs are run serially and are not preempted.

The scheduler queues propagator execution rather than running them immediately,
which prevents infinite recursion when propagators trigger each other during
execution (e.g., when tms_query detects a contradiction and kicks out a premise).

Based on the MIT Scheme implementation in propagator/core/scheduler.scm.

Main public interface:
    initialize_scheduler()    - Clear all scheduler state
    alert_propagators(jobs)   - Schedule a list of jobs to run
    alert_all_propagators()   - Reschedule all jobs ever scheduled
    run()                     - Run scheduled jobs until done

Scheduler variants:
    RoundRobinScheduler       - Default, runs all propagators in rounds
    StackScheduler            - LIFO order, runs one propagator at a time
    FastSlowScheduler         - Two-queue: fast (regular) before slow (amb_choose)

Use set_scheduler_factory() to change the scheduler type.
Use get_scheduler_stats() to inspect scheduler behavior.
"""

from __future__ import annotations

from typing import Callable, List, Set, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
import weakref


class SchedulerType(Enum):
    """Enum for available scheduler types."""
    ROUND_ROBIN = "round_robin"
    STACK = "stack"
    FAST_SLOW_ROUND_ROBIN = "fast_slow_round_robin"
    FAST_SLOW_STACK = "fast_slow_stack"


@dataclass
class SchedulerStats:
    """Statistics about scheduler execution for debugging and performance analysis."""
    scheduler_type: SchedulerType
    total_executions: int = 0
    fast_executions: int = 0
    slow_executions: int = 0
    rounds: int = 0
    
    def reset(self) -> None:
        """Reset all counters."""
        self.total_executions = 0
        self.fast_executions = 0
        self.slow_executions = 0
        self.rounds = 0
    
    def __str__(self) -> str:
        lines = [
            f"Scheduler: {self.scheduler_type.value}",
            f"  Total executions: {self.total_executions}",
            f"  Rounds: {self.rounds}",
        ]
        if self.scheduler_type in (SchedulerType.FAST_SLOW_ROUND_ROBIN, SchedulerType.FAST_SLOW_STACK):
            lines.append(f"  Fast executions: {self.fast_executions}")
            lines.append(f"  Slow executions: {self.slow_executions}")
        return "\n".join(lines)


class SchedulerProtocol(Protocol):
    """Protocol defining the scheduler interface."""
    
    def alert_one(self, propagator: Callable) -> None:
        """Add a propagator to the queue."""
        ...
    
    def clear(self) -> None:
        """Clear all pending propagators."""
        ...
    
    def is_done(self) -> bool:
        """Check if there are no pending propagators."""
        ...
    
    def run(self) -> str:
        """Run all pending propagators until done."""
        ...
    
    def pending_count(self) -> int:
        """Return the number of pending propagators."""
        ...
    
    @property
    def stats(self) -> SchedulerStats:
        """Return execution statistics."""
        ...


# Global registry of "slow" propagators (by identity)
# Uses WeakSet so entries are automatically removed when propagators are GC'd
_slow_propagators: "weakref.WeakSet[Callable]" = weakref.WeakSet()


def tag_slow(propagator: Callable) -> Callable:
    """
    Mark a propagator as "slow" so it runs after all fast propagators.
    
    This is useful for amb_choose propagators which should only run after
    constraint propagation has settled, giving deterministic propagators
    a chance to prune the search space first.
    
    Scheme equivalent:
        (define (tag-slow! thing)
          (eq-put! thing 'slow #t)
          thing)
    
    Args:
        propagator: The propagator to mark as slow
        
    Returns:
        The same propagator (for chaining)
    """
    _slow_propagators.add(propagator)
    return propagator


def is_slow(propagator: Callable) -> bool:
    """
    Check if a propagator is tagged as slow.
    
    Scheme equivalent:
        (define (tagged-slow? thing)
          (eq-get thing 'slow))
    """
    return propagator in _slow_propagators


def untag_slow(propagator: Callable) -> Callable:
    """Remove the slow tag from a propagator."""
    _slow_propagators.discard(propagator)
    return propagator


class RoundRobinScheduler:
    """
    A scheduler that runs all pending propagators in rounds.
    
    Each round executes all currently pending propagators, which may
    schedule new ones for the next round. This is the default scheduler.
    
    Scheme equivalent: round-robin-policy
    """
    
    def __init__(self):
        self._pending: List[Callable] = []
        self._pending_set: Set[int] = set()
        self._stats = SchedulerStats(scheduler_type=SchedulerType.ROUND_ROBIN)
    
    @property
    def stats(self) -> SchedulerStats:
        return self._stats
    
    def alert_one(self, propagator: Callable) -> None:
        """Add a propagator to the queue if not already present."""
        prop_id = id(propagator)
        if prop_id not in self._pending_set:
            self._pending.append(propagator)
            self._pending_set.add(prop_id)
    
    def clear(self) -> None:
        """Clear all pending propagators."""
        self._pending.clear()
        self._pending_set.clear()
    
    def is_done(self) -> bool:
        """Check if there are no pending propagators."""
        return len(self._pending) == 0
    
    def pending_count(self) -> int:
        """Return the number of pending propagators."""
        return len(self._pending)
    
    def run(self) -> str:
        """Run all pending propagators until queue is empty."""
        while not self.is_done():
            self._run_round()
        return 'done'
    
    def _run_round(self) -> None:
        """Execute all currently pending propagators."""
        if not self._pending:
            return
        
        self._stats.rounds += 1
        
        # Take snapshot and clear
        to_run = self._pending
        self._pending = []
        self._pending_set.clear()
        
        # Execute each propagator
        for propagator in to_run:
            self._stats.total_executions += 1
            propagator()


class StackScheduler:
    """
    A scheduler that runs propagators in LIFO (stack) order.
    
    Runs one propagator at a time, always picking the most recently added.
    
    Scheme equivalent: stack-policy
    """
    
    def __init__(self):
        self._pending: List[Callable] = []
        self._pending_set: Set[int] = set()
        self._stats = SchedulerStats(scheduler_type=SchedulerType.STACK)
    
    @property
    def stats(self) -> SchedulerStats:
        return self._stats
    
    def alert_one(self, propagator: Callable) -> None:
        """Add a propagator to the queue if not already present."""
        prop_id = id(propagator)
        if prop_id not in self._pending_set:
            self._pending.append(propagator)
            self._pending_set.add(prop_id)
    
    def clear(self) -> None:
        """Clear all pending propagators."""
        self._pending.clear()
        self._pending_set.clear()
    
    def is_done(self) -> bool:
        """Check if there are no pending propagators."""
        return len(self._pending) == 0
    
    def pending_count(self) -> int:
        """Return the number of pending propagators."""
        return len(self._pending)
    
    def run(self) -> str:
        """Run propagators one at a time in stack order."""
        while not self.is_done():
            self._stats.rounds += 1
            self._stats.total_executions += 1
            propagator = self._pending.pop()
            self._pending_set.discard(id(propagator))
            propagator()
        return 'done'


class FastSlowScheduler:
    """
    A two-queue scheduler that prioritizes fast propagators over slow ones.
    
    Slow propagators (tagged with tag_slow()) are placed in a separate queue
    and only run after all fast propagators have been exhausted. This allows
    deterministic constraint propagation to settle before amb choices are made.
    
    The policy parameter determines how propagators are executed within each queue:
    - 'round_robin': Execute all pending in rounds (default)
    - 'stack': Execute one at a time in LIFO order
    
    Scheme equivalent:
        (define (make-fast-slow-scheduler fast-policy slow-policy) ...)
    """
    
    def __init__(self, policy: str = 'round_robin'):
        self._policy = policy
        self._fast_pending: List[Callable] = []
        self._fast_set: Set[int] = set()
        self._slow_pending: List[Callable] = []
        self._slow_set: Set[int] = set()
        
        scheduler_type = (
            SchedulerType.FAST_SLOW_ROUND_ROBIN if policy == 'round_robin'
            else SchedulerType.FAST_SLOW_STACK
        )
        self._stats = SchedulerStats(scheduler_type=scheduler_type)
    
    @property
    def stats(self) -> SchedulerStats:
        return self._stats
    
    def alert_one(self, propagator: Callable) -> None:
        """Add a propagator to the appropriate queue."""
        prop_id = id(propagator)
        
        if is_slow(propagator):
            if prop_id not in self._slow_set:
                self._slow_pending.append(propagator)
                self._slow_set.add(prop_id)
        else:
            if prop_id not in self._fast_set:
                self._fast_pending.append(propagator)
                self._fast_set.add(prop_id)
    
    def clear(self) -> None:
        """Clear all pending propagators."""
        self._fast_pending.clear()
        self._fast_set.clear()
        self._slow_pending.clear()
        self._slow_set.clear()
    
    def is_done(self) -> bool:
        """Check if there are no pending propagators."""
        return len(self._fast_pending) == 0 and len(self._slow_pending) == 0
    
    def pending_count(self) -> int:
        """Return the total number of pending propagators."""
        return len(self._fast_pending) + len(self._slow_pending)
    
    def fast_pending_count(self) -> int:
        """Return the number of pending fast propagators."""
        return len(self._fast_pending)
    
    def slow_pending_count(self) -> int:
        """Return the number of pending slow propagators."""
        return len(self._slow_pending)
    
    def run(self) -> str:
        """Run all pending propagators, fast queue first."""
        while not self.is_done():
            # Run all fast propagators first
            if self._fast_pending:
                self._run_fast()
            # Then run slow propagators
            elif self._slow_pending:
                self._run_slow()
        return 'done'
    
    def _run_fast(self) -> None:
        """Execute fast propagators according to policy."""
        if self._policy == 'round_robin':
            self._run_fast_round_robin()
        else:
            self._run_fast_stack()
    
    def _run_slow(self) -> None:
        """Execute slow propagators according to policy."""
        if self._policy == 'round_robin':
            self._run_slow_round_robin()
        else:
            self._run_slow_stack()
    
    def _run_fast_round_robin(self) -> None:
        """Execute all fast propagators in one round."""
        if not self._fast_pending:
            return
        
        self._stats.rounds += 1
        to_run = self._fast_pending
        self._fast_pending = []
        self._fast_set.clear()
        
        for propagator in to_run:
            self._stats.total_executions += 1
            self._stats.fast_executions += 1
            propagator()
    
    def _run_fast_stack(self) -> None:
        """Execute one fast propagator (LIFO)."""
        if not self._fast_pending:
            return
        
        self._stats.rounds += 1
        self._stats.total_executions += 1
        self._stats.fast_executions += 1
        propagator = self._fast_pending.pop()
        self._fast_set.discard(id(propagator))
        propagator()
    
    def _run_slow_round_robin(self) -> None:
        """Execute all slow propagators in one round."""
        if not self._slow_pending:
            return
        
        self._stats.rounds += 1
        to_run = self._slow_pending
        self._slow_pending = []
        self._slow_set.clear()
        
        for propagator in to_run:
            self._stats.total_executions += 1
            self._stats.slow_executions += 1
            propagator()
    
    def _run_slow_stack(self) -> None:
        """Execute one slow propagator (LIFO)."""
        if not self._slow_pending:
            return
        
        self._stats.rounds += 1
        self._stats.total_executions += 1
        self._stats.slow_executions += 1
        propagator = self._slow_pending.pop()
        self._slow_set.discard(id(propagator))
        propagator()


# Scheduler factory type
SchedulerFactory = Callable[[], SchedulerProtocol]

# Global scheduler state
_scheduler: SchedulerProtocol | None = None
_scheduler_factory: SchedulerFactory = RoundRobinScheduler
_propagators_ever_alerted: List[Callable] = []
_propagators_ever_alerted_set: Set[int] = set()

# Flag to track if we're currently running the scheduler
_is_running: bool = False

# Callback to initialize TMS state (set by tms module to avoid circular import)
_tms_initializer: Callable[[], str] | None = None


def set_scheduler_factory(factory: SchedulerFactory | SchedulerType) -> None:
    """
    Set the factory function used to create new schedulers.
    
    This affects the scheduler created by the next initialize_scheduler() call.
    Does not affect the current scheduler.
    
    Args:
        factory: Either a SchedulerType enum or a callable that returns a scheduler
        
    Example:
        # Use fast/slow scheduler
        set_scheduler_factory(SchedulerType.FAST_SLOW_ROUND_ROBIN)
        initialize_scheduler()
        
        # Or use a custom factory
        set_scheduler_factory(lambda: FastSlowScheduler(policy='stack'))
        initialize_scheduler()
    """
    global _scheduler_factory
    
    if isinstance(factory, SchedulerType):
        if factory == SchedulerType.ROUND_ROBIN:
            _scheduler_factory = RoundRobinScheduler
        elif factory == SchedulerType.STACK:
            _scheduler_factory = StackScheduler
        elif factory == SchedulerType.FAST_SLOW_ROUND_ROBIN:
            _scheduler_factory = lambda: FastSlowScheduler(policy='round_robin')
        elif factory == SchedulerType.FAST_SLOW_STACK:
            _scheduler_factory = lambda: FastSlowScheduler(policy='stack')
    else:
        _scheduler_factory = factory


def get_scheduler_type() -> SchedulerType:
    """
    Get the type of the current scheduler.
    
    Returns:
        The SchedulerType of the currently active scheduler
    """
    scheduler = _ensure_scheduler()
    return scheduler.stats.scheduler_type


def get_scheduler_stats() -> SchedulerStats:
    """
    Get statistics about the current scheduler's execution.
    
    Returns:
        A SchedulerStats object with execution counts
    """
    scheduler = _ensure_scheduler()
    return scheduler.stats


def reset_scheduler_stats() -> None:
    """Reset the scheduler statistics counters."""
    scheduler = _ensure_scheduler()
    scheduler.stats.reset()


def register_tms_initializer(initializer: Callable[[], str]) -> None:
    """
    Register the TMS initializer callback.
    
    This is called by the tms module to register its initialize_tms function,
    avoiding circular import issues.
    """
    global _tms_initializer
    _tms_initializer = initializer


def initialize_scheduler() -> str:
    """
    Clear all scheduler state and create a fresh scheduler.
    
    Uses the currently configured scheduler factory (set via set_scheduler_factory).
    Also clears TMS state (worldview, premise status, consequence cache)
    if the TMS module has been loaded.
    
    Scheme equivalent:
        (define (initialize-scheduler)
          (set! *scheduler* (make-scheduler))
          (set! *propagators-ever-alerted* (make-eq-oset))
          (set! *consequence-cache* (make-eq-hash-table))
          'ok)
    """
    global _scheduler, _propagators_ever_alerted, _propagators_ever_alerted_set, _is_running
    _scheduler = _scheduler_factory()
    _propagators_ever_alerted = []
    _propagators_ever_alerted_set = set()
    _is_running = False
    
    # Clear slow propagator registry
    _slow_propagators.clear()
    
    # Clear TMS state if the module has registered its initializer
    if _tms_initializer is not None:
        _tms_initializer()
    
    return 'ok'


def _ensure_scheduler() -> SchedulerProtocol:
    """Ensure the scheduler is initialized."""
    global _scheduler
    if _scheduler is None:
        initialize_scheduler()
    return _scheduler


def alert_propagators(propagators: Callable | List[Callable]) -> None:
    """
    Schedule one or more propagators to be executed.
    
    If the scheduler is currently running (inside run()), the propagators
    are queued for the next round. Otherwise, they are queued but not
    automatically executed - you must call run() to execute them.
    
    Scheme equivalent:
        (define (alert-propagators propagators)
          (for-each
           (lambda (propagator)
             (oset-insert *propagators-ever-alerted* propagator)
             ((*scheduler* 'alert-one) propagator))
           (listify propagators)))
    """
    global _propagators_ever_alerted, _propagators_ever_alerted_set
    
    scheduler = _ensure_scheduler()
    
    # Convert to list if single item
    prop_list = propagators if isinstance(propagators, list) else [propagators]
    
    for propagator in prop_list:
        # Track all propagators ever alerted
        prop_id = id(propagator)
        if prop_id not in _propagators_ever_alerted_set:
            _propagators_ever_alerted.append(propagator)
            _propagators_ever_alerted_set.add(prop_id)
        
        # Add to scheduler queue
        scheduler.alert_one(propagator)


def alert_all_propagators() -> None:
    """
    Schedule all known propagators for execution.
    
    Scheme equivalent:
        (define (alert-all-propagators!)
          (for-each (*scheduler* 'alert-one) (all-propagators)))
    """
    scheduler = _ensure_scheduler()
    for propagator in _propagators_ever_alerted:
        scheduler.alert_one(propagator)


def all_propagators() -> List[Callable]:
    """
    Return a list of all propagators ever alerted.
    
    Scheme equivalent:
        (define (all-propagators)
          (oset-members *propagators-ever-alerted*))
    """
    return list(_propagators_ever_alerted)


def run() -> str:
    """
    Run all scheduled propagators until the queue is empty.
    
    This should be called after setting up the network and adding
    initial content to cells. It will execute propagators in rounds
    until no more propagators need to run.
    
    Returns:
        'done' when complete
        
    Scheme equivalent:
        (define (run)
          (set! *last-value-of-run* (with-process-abortion do-run))
          *last-value-of-run*)
    """
    global _is_running
    
    scheduler = _ensure_scheduler()
    
    if _is_running:
        # Already running - don't nest, the outer run() will pick up new work
        return 'nested'
    
    _is_running = True
    try:
        return scheduler.run()
    finally:
        _is_running = False


def is_scheduler_running() -> bool:
    """Check if the scheduler is currently executing propagators."""
    return _is_running


def scheduler_pending_count() -> int:
    """Return the number of pending propagators (for debugging)."""
    scheduler = _ensure_scheduler()
    return scheduler.pending_count()


# Auto-run mode: for backward compatibility
# When True, run() is called automatically after each alert_propagators
# When False, you must call run() manually
_auto_run: bool = True


def set_auto_run(enabled: bool) -> None:
    """
    Set whether to automatically run propagators after alerting.
    
    When enabled (default for backward compatibility), calling alert_propagators
    will also trigger run() if we're not already running.
    
    When disabled, you must call run() manually to execute propagators.
    """
    global _auto_run
    _auto_run = enabled


def get_auto_run() -> bool:
    """Check if auto-run mode is enabled."""
    return _auto_run


def alert_propagators_and_maybe_run(propagators: Callable | List[Callable]) -> None:
    """
    Schedule propagators and run if auto-run is enabled and not already running.
    
    This is the main function used by cells when content changes.
    """
    alert_propagators(propagators)
    
    if _auto_run and not _is_running:
        run()


# Legacy Scheduler class for backward compatibility
class Scheduler(RoundRobinScheduler):
    """Legacy alias for RoundRobinScheduler."""
    pass


# Initialize the scheduler on module load
initialize_scheduler()
