"""
CDCL (Conflict-Driven Clause Learning) enhancement for propagator TMS.

This module provides CDCL-style conflict analysis and backjumping for the
propagator guessing machine, enabling more efficient search via:

1. Decision level tracking - know when each choice was made
2. Implication tracking - know why each value was derived
3. 1-UIP clause learning - learn minimal conflict reasons
4. Non-chronological backjumping - skip irrelevant decision levels
5. Activity-based heuristics (VSIDS) - prioritize contentious variables

Usage:
    from propagator.cdcl import enable_cdcl, disable_cdcl, cdcl_stats
    
    enable_cdcl()  # Enable CDCL enhancements globally
    
    # ... run propagator network with one_of/binary_amb ...
    
    print(cdcl_stats())
    disable_cdcl()

The CDCL engine is designed to work alongside the existing TMS, not replace it.
It hooks into the conflict processing and decision-making paths to provide
enhanced search efficiency.

See docs/CDCL_DESIGN.md for the full design document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import time

from .tms import hypothetical_p, kick_out, assimilate_nogood, process_one_contradiction


# =============================================================================
# CDCL Configuration Constants
# =============================================================================

# VSIDS activity decay factor (applied after each conflict)
# Higher values = slower decay = longer memory of past conflicts
VSIDS_DECAY_FACTOR = 0.95

# Maximum activity value before rescaling to prevent overflow
VSIDS_MAX_ACTIVITY = 1e100

# Maximum number of learned clauses to keep in database
# Excess clauses are pruned by quality (LBD score)
MAX_LEARNED_CLAUSES = 10000

# Restart interval: restart search after this many conflicts
# Set to 0 to disable restarts
RESTART_INTERVAL = 100


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CDCLStats:
    """Statistics about CDCL operations."""
    conflicts: int = 0
    decisions: int = 0
    propagations: int = 0
    learned_clauses: int = 0
    backjumps: int = 0
    backjump_levels_saved: int = 0  # Sum of (current_level - backjump_level)
    chronological_backtracks: int = 0  # Fallback to level-1 backtrack
    restarts: int = 0
    start_time: float = field(default_factory=time.time)
    
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time
    
    def report(self) -> str:
        """Generate a human-readable stats report."""
        lines = [
            "CDCL Statistics:",
            f"  Conflicts:         {self.conflicts}",
            f"  Decisions:         {self.decisions}",
            f"  Propagations:      {self.propagations}",
            f"  Learned clauses:   {self.learned_clauses}",
            f"  Backjumps:         {self.backjumps}",
            f"  Chronological:     {self.chronological_backtracks}",
            f"  Levels saved:      {self.backjump_levels_saved}",
            f"  Restarts:          {self.restarts}",
            f"  Time:              {self.elapsed():.3f}s",
        ]
        if self.conflicts > 0:
            avg_save = self.backjump_levels_saved / self.conflicts
            lines.append(f"  Avg backjump:      {avg_save:.2f} levels")
        return "\n".join(lines)


@dataclass
class LearnedClause:
    """A learned conflict clause with metadata."""
    premises: frozenset  # The premises in the clause (negated: at least one must be false)
    learned_at_level: int
    activity: float = 0.0
    lbd: int = 0  # Literal Block Distance - clause quality metric
    
    def __hash__(self):
        return hash(self.premises)
    
    def __eq__(self, other):
        if not isinstance(other, LearnedClause):
            return False
        return self.premises == other.premises


@dataclass
class ImplicationInfo:
    """Information about an implication (or decision)."""
    premise: Any
    level: int
    is_decision: bool
    reason: Optional[frozenset] = None  # The clause that implied this (None if decision)
    order: int = 0  # When in the trail this was set
    opposite_premise: Optional[Any] = None  # For binary_amb, the other branch


@dataclass 
class DecisionRecord:
    """Record of a decision made at a specific level."""
    level: int
    premise: Any  # The premise brought in
    opposite: Any  # The premise kicked out
    

# =============================================================================
# CDCL Engine
# =============================================================================

class CDCLEngine:
    """
    CDCL conflict analysis engine for propagator TMS.
    
    Integrates with the existing TMS by:
    1. Tracking decision levels when binary_amb makes choices
    2. Recording implications when propagators derive values
    3. Performing 1-UIP analysis on conflicts
    4. Implementing non-chronological backjumping
    5. Using VSIDS activity scores for branch selection
    """
    
    def __init__(self):
        self.enabled = False
        self.decision_level = 0
        self.trail_order = 0
        
        # Maps id(premise) -> ImplicationInfo
        self._implications: Dict[int, ImplicationInfo] = {}
        
        # Decision stack: list of DecisionRecord
        self._decision_stack: List[DecisionRecord] = []
        
        # Activity scores for VSIDS (maps id(premise) -> activity)
        self._activities: Dict[int, float] = {}
        self._activity_increment = 1.0
        self._activity_decay = VSIDS_DECAY_FACTOR
        self._max_activity = VSIDS_MAX_ACTIVITY
        
        # Learned clauses (for clause database)
        self.learned_clauses: List[LearnedClause] = []
        self._max_learned_clauses = MAX_LEARNED_CLAUSES
        
        # Statistics
        self.stats = CDCLStats()
        
        # Hooks for integration
        self._on_conflict_handlers: List[Callable] = []
        self._on_decision_handlers: List[Callable] = []
    
    def enable(self) -> None:
        """Enable CDCL enhancements."""
        self.enabled = True
        self.reset()
    
    def disable(self) -> None:
        """Disable CDCL enhancements."""
        self.enabled = False
    
    def reset(self) -> None:
        """Reset CDCL state for a new problem (keeps learned clauses for incremental)."""
        self.decision_level = 0
        self.trail_order = 0
        self._implications.clear()
        self._decision_stack.clear()
        self.stats = CDCLStats()
    
    def full_reset(self) -> None:
        """Full reset including learned clauses and activities."""
        self.reset()
        self._activities.clear()
        self.learned_clauses.clear()
        self._activity_increment = 1.0
    
    # =========================================================================
    # Decision Level Tracking (Phase 1)
    # =========================================================================
    
    def make_decision(
        self,
        chosen_premise: Any,
        opposite_premise: Any,
    ) -> int:
        """
        Record a decision (choice made by amb_choose).
        
        Args:
            chosen_premise: The premise brought in
            opposite_premise: The premise kicked out
            
        Returns:
            The new decision level.
        """
        self.decision_level += 1
        self.trail_order += 1
        self.stats.decisions += 1
        
        self._implications[id(chosen_premise)] = ImplicationInfo(
            premise=chosen_premise,
            level=self.decision_level,
            is_decision=True,
            reason=None,
            order=self.trail_order,
            opposite_premise=opposite_premise,
        )
        
        self._decision_stack.append(DecisionRecord(
            level=self.decision_level,
            premise=chosen_premise,
            opposite=opposite_premise,
        ))
        
        # Notify handlers
        for handler in self._on_decision_handlers:
            handler(self.decision_level, chosen_premise, opposite_premise)
        
        return self.decision_level
    
    def get_level(self, premise: Any) -> int:
        """Get the decision level of a premise."""
        info = self._implications.get(id(premise))
        return info.level if info else 0
    
    def get_order(self, premise: Any) -> int:
        """Get the trail order of a premise."""
        info = self._implications.get(id(premise))
        return info.order if info else 0
    
    def get_current_level(self) -> int:
        """Get the current decision level."""
        return self.decision_level
    
    # =========================================================================
    # Implication Tracking (Phase 2)
    # =========================================================================
    
    def record_propagation(
        self, 
        premise: Any, 
        reason: Optional[Set[Any]] = None,
    ) -> None:
        """
        Record an implication (value forced by propagation).
        
        Args:
            premise: The premise that was implied
            reason: The premises that caused this implication
        """
        self.trail_order += 1
        self.stats.propagations += 1
        
        reason_set = frozenset(reason) if reason else None
        
        self._implications[id(premise)] = ImplicationInfo(
            premise=premise,
            level=self.decision_level,
            is_decision=False,
            reason=reason_set,
            order=self.trail_order,
        )
    
    def get_reason(self, premise: Any) -> Optional[frozenset]:
        """Get the clause that implied this premise (None if decision)."""
        info = self._implications.get(id(premise))
        if info is None or info.is_decision:
            return None
        return info.reason
    
    def is_decision(self, premise: Any) -> bool:
        """Check if premise was a decision (not an implication)."""
        info = self._implications.get(id(premise))
        return info.is_decision if info else False
    
    # =========================================================================
    # Activity-Based Heuristics - VSIDS (Phase 5)
    # =========================================================================
    
    def bump_activity(self, premise: Any, amount: float = None) -> None:
        """
        Increase the activity of a premise (used in conflict).
        
        VSIDS: premises involved in recent conflicts get higher activity.
        """
        if amount is None:
            amount = self._activity_increment
            
        pid = id(premise)
        current = self._activities.get(pid, 0.0)
        new_activity = current + amount
        self._activities[pid] = new_activity
        
        # Rescale if activities get too large
        if new_activity > self._max_activity:
            self._rescale_activities()
    
    def decay_activities(self) -> None:
        """
        Decay all activities periodically.
        
        Implementation: Instead of multiplying all activities by decay,
        we increase the increment. This has the same effect but is O(1).
        """
        self._activity_increment /= self._activity_decay
    
    def _rescale_activities(self) -> None:
        """Rescale all activities to prevent overflow."""
        scale = 1.0 / self._max_activity
        for pid in self._activities:
            self._activities[pid] *= scale
        self._activity_increment *= scale
    
    def get_activity(self, premise: Any) -> float:
        """Get the activity score of a premise."""
        return self._activities.get(id(premise), 0.0)
    
    def choose_branch(self, true_premise: Any, false_premise: Any) -> str:
        """
        Choose which branch to try first based on activity scores.
        
        VSIDS heuristic: prefer the branch with higher activity.
        Premises involved in more recent conflicts have higher activity.
        
        Returns:
            'true' or 'false'
        """
        true_activity = self.get_activity(true_premise)
        false_activity = self.get_activity(false_premise)
        
        # Prefer higher activity (more contentious = resolve first)
        return 'true' if true_activity >= false_activity else 'false'
    
    # =========================================================================
    # 1-UIP Clause Learning (Phase 3)
    # =========================================================================
    
    def analyze_conflict(self, conflict: Set[Any]) -> Tuple[frozenset, int]:
        """
        Perform 1-UIP conflict analysis.
        
        The First Unique Implication Point (1-UIP) is the first point on the
        implication graph where all paths from decisions to the conflict
        converge. Learning the 1-UIP clause:
        
        1. Is guaranteed to be asserting (exactly one lit at current level)
        2. Causes unit propagation after backjumping
        3. Is typically smaller than the full conflict clause
        
        Args:
            conflict: Set of premises that led to contradiction
            
        Returns:
            (learned_clause, backjump_level)
        """
        self.stats.conflicts += 1
        
        # Notify handlers
        for handler in self._on_conflict_handlers:
            handler(conflict)
        
        if self.decision_level == 0:
            # Conflict at level 0 - problem is unsatisfiable
            return frozenset(conflict), 0
        
        # Initialize with conflict clause
        clause = set(conflict)
        seen = set()  # Premises we've already processed
        
        def at_current_level(p):
            return hypothetical_p(p) and self.get_level(p) == self.decision_level
        
        # Resolve until we have exactly one literal at current level (1-UIP)
        # This is the standard 1-UIP algorithm from SAT solvers.
        #
        # Safety limit: In the worst case, each resolution step removes one literal
        # and adds several from the reason clause. We bound iterations to prevent
        # infinite loops if there's a cycle in the implication graph (shouldn't
        # happen in a correct implementation, but defensive programming).
        iterations = 0
        max_iterations = len(clause) * 2 + 10
        
        while iterations < max_iterations:
            iterations += 1
            current_level_lits = [p for p in clause if at_current_level(p)]
            
            if len(current_level_lits) <= 1:
                # Found 1-UIP: exactly one lit at current level
                break
            
            # Find the most recent literal at current level to resolve
            # (highest trail order, not yet resolved)
            candidates = [p for p in current_level_lits if id(p) not in seen]
            if not candidates:
                break  # Can't make more progress
                
            lit_to_resolve = max(candidates, key=self.get_order)
            seen.add(id(lit_to_resolve))
            
            # Get the reason this literal was implied
            reason = self.get_reason(lit_to_resolve)
            if reason is None:
                # This was a decision - can't resolve further
                # Keep it in the clause
                continue
            
            # Resolution: clause = (clause - {lit}) ∪ reason
            # This is like resolving two clauses in propositional logic
            clause.discard(lit_to_resolve)
            for p in reason:
                if p is not lit_to_resolve:
                    clause.add(p)
        
        # Bump activities of all premises in the learned clause
        for p in clause:
            self.bump_activity(p)
        self.decay_activities()
        
        # Compute backjump level (second-highest level in clause)
        learned = frozenset(clause)
        levels = sorted([
            self.get_level(p) for p in learned if hypothetical_p(p)
        ], reverse=True)
        
        backjump_level = 0
        if len(levels) > 1:
            backjump_level = levels[1]
        elif len(levels) == 1:
            # Unit clause - backjump to level 0
            backjump_level = 0
        
        # Compute LBD (Literal Block Distance) - quality metric
        lbd = len(set(levels))
        
        # Store the learned clause
        self._store_learned_clause(LearnedClause(
            premises=learned,
            learned_at_level=self.decision_level,
            lbd=lbd,
        ))
        
        return learned, backjump_level
    
    def _store_learned_clause(self, clause: LearnedClause) -> None:
        """Store a learned clause, managing the clause database."""
        self.learned_clauses.append(clause)
        self.stats.learned_clauses += 1
        
        # Clause database management: remove low-quality clauses if too many
        if len(self.learned_clauses) > self._max_learned_clauses:
            self._reduce_clause_database()
    
    def _reduce_clause_database(self) -> None:
        """
        Reduce the clause database by removing low-quality clauses.
        
        Keep clauses with low LBD (high quality) and recent activity.
        """
        # Sort by quality: lower LBD and higher activity is better
        def clause_score(c: LearnedClause) -> float:
            return -c.lbd + c.activity * 0.01
        
        self.learned_clauses.sort(key=clause_score, reverse=True)
        # Keep top half
        self.learned_clauses = self.learned_clauses[:self._max_learned_clauses // 2]
    
    # =========================================================================
    # Non-Chronological Backjumping (Phase 4)
    # =========================================================================
    
    def backjump(self, target_level: int) -> List[Any]:
        """
        Backjump to target_level, undoing all decisions above it.
        
        This is the key CDCL optimization: instead of backtracking one
        level at a time, we can jump directly to the level that matters.
        
        Args:
            target_level: Level to backjump to (0 = restart)
            
        Returns:
            List of premises that were kicked out
        """
        levels_saved = self.decision_level - target_level
        if levels_saved > 0:
            self.stats.backjumps += 1
            self.stats.backjump_levels_saved += levels_saved
        else:
            self.stats.chronological_backtracks += 1
        
        kicked = []
        to_remove = []
        
        # Kick out all premises assigned above target level
        for pid, info in self._implications.items():
            if info.level > target_level:
                kick_out(info.premise)
                kicked.append(info.premise)
                to_remove.append(pid)
        
        # Clean up implication records
        for pid in to_remove:
            del self._implications[pid]
        
        # Clean up decision stack
        while self._decision_stack and self._decision_stack[-1].level > target_level:
            self._decision_stack.pop()
        
        self.decision_level = target_level
        return kicked
    
    # =========================================================================
    # Integration with TMS
    # =========================================================================
    
    def process_conflict_cdcl(self, nogood: List[Any]) -> None:
        """
        CDCL-style conflict processing with 1-UIP learning and backjumping.
        
        This is called instead of the standard process_one_contradiction
        when CDCL is enabled.
        """
        if not self.enabled:
            # Fall back to standard processing
            process_one_contradiction(nogood)
            return
        
        # Check if there are hypotheticals to work with
        hyps = [p for p in nogood if hypothetical_p(p)]
        if not hyps:
            # No hypotheticals - this is a ground-level contradiction
            process_one_contradiction(nogood)
            return
        
        # Learn 1-UIP clause and compute backjump level
        learned_clause, backjump_level = self.analyze_conflict(set(nogood))
        
        # Check again after analysis
        learned_hyps = [p for p in learned_clause if hypothetical_p(p)]
        if not learned_hyps:
            process_one_contradiction(list(learned_clause))
            return
        
        # Perform backjump
        self.backjump(backjump_level)
        
        # Assimilate the learned clause to all premises
        # This ensures the same conflict won't be repeated
        learned_list = list(learned_clause)
        for premise in learned_clause:
            assimilate_nogood(premise, learned_list)
    
    def should_restart(self) -> bool:
        """
        Determine if a restart is beneficial.
        
        Returns True every RESTART_INTERVAL conflicts.
        More sophisticated strategies (Luby sequence, glucose-style) could be added.
        """
        if RESTART_INTERVAL == 0:
            return False
        return self.stats.conflicts > 0 and self.stats.conflicts % RESTART_INTERVAL == 0
    
    def restart(self) -> None:
        """Perform a restart (backjump to level 0)."""
        self.backjump(0)
        self.stats.restarts += 1
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def on_conflict(self, handler: Callable) -> None:
        """Register a handler to be called on conflicts."""
        self._on_conflict_handlers.append(handler)
    
    def on_decision(self, handler: Callable) -> None:
        """Register a handler to be called on decisions."""
        self._on_decision_handlers.append(handler)
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def report(self) -> str:
        """Generate a statistics report."""
        return self.stats.report()
    
    def dump_state(self) -> dict:
        """Dump current state for debugging."""
        return {
            'enabled': self.enabled,
            'decision_level': self.decision_level,
            'trail_order': self.trail_order,
            'num_implications': len(self._implications),
            'num_decisions': len(self._decision_stack),
            'num_learned': len(self.learned_clauses),
            'stats': {
                'conflicts': self.stats.conflicts,
                'decisions': self.stats.decisions,
                'backjumps': self.stats.backjumps,
                'levels_saved': self.stats.backjump_levels_saved,
            },
        }


# =============================================================================
# Global Engine Instance and API
# =============================================================================

_cdcl_engine: Optional[CDCLEngine] = None


def get_cdcl_engine() -> CDCLEngine:
    """Get the global CDCL engine, creating it if needed."""
    global _cdcl_engine
    if _cdcl_engine is None:
        _cdcl_engine = CDCLEngine()
    return _cdcl_engine


def enable_cdcl() -> CDCLEngine:
    """
    Enable CDCL enhancements globally.
    
    Returns the CDCL engine for inspection/configuration.
    """
    engine = get_cdcl_engine()
    engine.enable()
    return engine


def disable_cdcl() -> None:
    """Disable CDCL enhancements globally."""
    engine = get_cdcl_engine()
    engine.disable()


def reset_cdcl() -> None:
    """Reset CDCL state (but keep learned clauses)."""
    engine = get_cdcl_engine()
    engine.reset()


def full_reset_cdcl() -> None:
    """Full reset of CDCL including learned clauses."""
    engine = get_cdcl_engine()
    engine.full_reset()


def cdcl_enabled() -> bool:
    """Check if CDCL is currently enabled."""
    engine = get_cdcl_engine()
    return engine.enabled


def process_conflict_cdcl(nogood: List[Any]) -> None:
    """Run CDCL-style conflict analysis and backjumping for a nogood.

    Thin module-level wrapper (matches enable_cdcl/reset_cdcl/etc. above)
    so tms.py can register this as a callback without needing the engine
    singleton itself -- see register_cdcl_handlers in tms.py.
    """
    get_cdcl_engine().process_conflict_cdcl(nogood)


def cdcl_stats() -> str:
    """Get CDCL statistics report."""
    engine = get_cdcl_engine()
    return engine.report()


def cdcl_conflicts() -> int:
    """Get the number of CDCL conflicts."""
    engine = get_cdcl_engine()
    return engine.stats.conflicts


def cdcl_backjumps() -> int:
    """Get the number of backjumps."""
    engine = get_cdcl_engine()
    return engine.stats.backjumps


def cdcl_levels_saved() -> int:
    """Get the total decision levels saved by backjumping."""
    engine = get_cdcl_engine()
    return engine.stats.backjump_levels_saved


# Register with tms.py so it can reset CDCL state and hand off nogood
# processing without ever importing this module -- see register_cdcl_handlers
# in tms.py for the full rationale (this mirrors register_tms_initializer's
# use in scheduler.py/tms.py for the same kind of cycle).
from .tms import register_cdcl_handlers

register_cdcl_handlers(full_reset_cdcl, cdcl_enabled, process_conflict_cdcl)
