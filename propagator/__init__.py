"""Propagator library for constraint propagation and reactive computation."""

# Scheduler (must be imported first to avoid circular imports)
from .scheduler import (
    initialize_scheduler,
    run,
    all_propagators,
    is_scheduler_running,
    set_auto_run,
    get_auto_run,
    # Scheduler types and configuration
    SchedulerType,
    SchedulerStats,
    set_scheduler_factory,
    get_scheduler_type,
    get_scheduler_stats,
    reset_scheduler_stats,
    # Slow propagator tagging
    tag_slow,
    is_slow,
    untag_slow,
    # Scheduler classes (for advanced use)
    RoundRobinScheduler,
    StackScheduler,
    FastSlowScheduler,
)

# Core cell and propagator functionality
from .cell import (
    Cell,
    alert_propagators,
    alert_all_propagators,
    listify,
    lift_to_cell_contents,
    lift_to_provenance_aware_cell_contents,
    propagator,
    function_to_propagator_constructor,
    provenance_aware_function_to_propagator_constructor,
    compound_propagator,
)

# Primitive propagators (for numbers)
from .primitives import (
    constant,
    adder,
    subtractor,
    multiplier,
    divider,
    absolute_value,
    squarer,
    sqrter,
    eq,
    lt,
    gt,
    lte,
    gte,
    inverter,
    conjoiner,
    disjoiner,
    conditional,
    switch,
    # Legacy/alternative names for common patterns
    equal_to,
    less_than,
    greater_than,
    abs_value,
    absoluter,  # alias for absolute_value
    neg,
    negate,
    and_gate,
    or_gate,
    # Generic operators
    generic_add,
    generic_sub,
    generic_mul,
    generic_div,
    generic_abs,
    generic_square,
    generic_sqrt,
    generic_eq,
    generic_lt,
    generic_gt,
    generic_lte,
    generic_gte,
    generic_not,
    generic_and,
    generic_or,
    generic_switch,
    # Multidirectional constraint propagators that impose relations rather than computing outputs. They compute based on which inputs are available, and merge information automatically.
    product,
    sum_constraint,
    difference,
    quadratic,
)

# # Constraint propagators (multidirectional via composition)
# from .constraints import (
#     product,
#     sum_constraint,
#     difference,
#     quadratic,
# )

# Nothing sentinel (foundational module with no dependencies)
from .nothing import (
    nothing,
    nothing_p,
)

# Merge system
from .merge import (
    the_contradiction,
    contradictory_p,
    contradictory,
    make_generic_operator,
    merge,
    generic_merge,  # Raw merge without equivalent? short-circuit
    assign_merge_operation,
    any_p,
    # Equivalence system (optimization for merge)
    equivalent,
    generic_equivalent,
    assign_equivalent_operation,
)

# Interval data structures and functions
from .intervals import (
    Interval,
    make_interval,
    interval_low,
    interval_high,
    add_interval,
    mul_interval,
    mul_interval_complete,
    sub_interval,
    div_interval,
    square_interval,
    sqrt_interval,
    empty_interval_p,
    intersect_intervals,
    to_interval,
    coercing,
)

# Monotonic log values
from .log import (
    LogEntry,
    Log,
    make_log_entry,
    make_log,
    append_log_entry,
    count_log,
    timestamps,
    entries_between,
    window_log,
    count_window,
    latest,
    latest_payload,
    latest_before,
    filter_log,
    filter_after_timestamp,
    filter_before_timestamp,
    map_log,
    map_payload_values,
    singleton_log,
    entry_to_log,
    normalize_log_increment,
)

# Dependency (supported values)
from .supported_values import (
    Supported,
    supported,
    more_informative_support,
    merge_supports,
    get_support_premises,
    support_contains,
    # Support set utilities (frozenset-based)
    Support,
    IdentityWrapper,
)

# Generic conditional operations (for proper handling of supported values)
from .generic_conditionals import (
    true_p,
    ignore_first,
    get_generic_true,
    get_generic_ignore_first,
)

# Truth maintenance system (TMS)
from .tms import (
    Tms,
    make_tms,
    Hypothetical,
    hypothetical,
    hypothetical_p,
    tms_p,
    tms_merge,
    tms_assimilate,
    tms_query,
    tms_contradiction_info,
    strongest_consequence,
    tms_unpacking,
    full_tms_unpacking,
    kick_out,
    bring_in,
    premise_in,
    mark_premise_in,
    mark_premise_out,
    premise_nogoods,
    set_premise_nogoods,
    initialize_tms,
    get_worldview_number,
    # Contradiction tracking - prefer getter functions over direct variable access
    # Direct variables are reset by initialize_scheduler() and imports capture old refs
    get_number_of_calls_to_fail,  # Preferred: returns current value
    get_last_nogood,               # Preferred: returns current value  
    get_contradictions,            # Preferred: returns copy of history
    number_of_calls_to_fail,       # Deprecated: direct access, may be stale after init
    last_nogood,                   # Deprecated: direct access, may be stale after init
    contradictions_history,        # Deprecated: direct access, may be stale after init
    describe_nogood,
    describe_last_contradiction,
    get_contradiction_details,
    set_contradiction_verbose,
    process_nogood,
    process_contradictions,
    process_one_contradiction,
    assimilate_nogood,
    pairwise_union,
    TmsContradiction,
    to_tms,
)

# Guessing machine utilities
from .guessing_machine import (
    binary_amb,
    require,
    abhor,
    require_distinct,
    one_of,
    one_of_the_cells,
)

# CDCL (Conflict-Driven Clause Learning) enhancements
from .cdcl import (
    CDCLEngine,
    CDCLStats,
    enable_cdcl,
    disable_cdcl,
    reset_cdcl,
    full_reset_cdcl,
    cdcl_enabled,
    cdcl_stats,
    cdcl_conflicts,
    cdcl_backjumps,
    cdcl_levels_saved,
    get_cdcl_engine,
)

# SAT/SMT Solver Export (compile propagator networks to external solvers)
from .solver_export import (
    NetworkCompiler, SolverBackend, ConstraintType, SolverResult,
    solve, SolveMode, SolveResult, SearchMode, search_mode,
    compile_from_roots, solve_from_roots, TranslationMode,
    UnsupportedTranslationError, TrueHybridNetwork,
    solve_dimacs, solve_smtlib2,
)

# Circuit-style visualization of propagator networks
from .circuit_viz import (
    CircuitTrace,
    CircuitCapture,
    capture_circuit,
    register_trace_spec,
    CellAnnotation,
    PremiseColorMap,
)

# Interval propagators (note: different from primitives - work with Intervals)
# from .interval_propagators import (
#     adder as interval_adder,
#     multiplier as interval_multiplier,
#     subtractor as interval_subtractor,
#     divider as interval_divider,
#     squarer as interval_squarer,
#     sqrter as interval_sqrter,
# )

__all__ = [
    # Scheduler
    'initialize_scheduler',
    'run',
    'all_propagators',
    'is_scheduler_running',
    'set_auto_run',
    'get_auto_run',
    # Scheduler types and configuration
    'SchedulerType',
    'SchedulerStats',
    'set_scheduler_factory',
    'get_scheduler_type',
    'get_scheduler_stats',
    'reset_scheduler_stats',
    # Slow propagator tagging
    'tag_slow',
    'is_slow',
    'untag_slow',
    # Scheduler classes
    'RoundRobinScheduler',
    'StackScheduler',
    'FastSlowScheduler',
    # Core
    'Cell',
    'alert_propagators',
    'alert_all_propagators',
    'listify',
    'lift_to_cell_contents',
    'lift_to_provenance_aware_cell_contents',
    'propagator',
    'function_to_propagator_constructor',
    'provenance_aware_function_to_propagator_constructor',
    'conditional',
    'switch',
    'compound_propagator',
    # Primitives (numbers)
    'constant',
    'adder',
    'subtractor',
    'multiplier',
    'divider',
    'absolute_value',
    'squarer',
    'sqrter',
    'eq',
    'lt',
    'gt',
    'lte',
    'gte',
    'inverter',
    'conjoiner',
    'disjoiner',
    # Legacy names and aliases
    'equal_to',
    'less_than',
    'greater_than',
    'abs_value',
    'absoluter',
    'neg',
    'negate',
    'and_gate',
    'or_gate',
    # Generic operators
    'generic_add',
    'generic_sub',
    'generic_mul',
    'generic_div',
    'generic_abs',
    'generic_square',
    'generic_sqrt',
    'generic_eq',
    'generic_lt',
    'generic_gt',
    'generic_lte',
    'generic_gte',
    'generic_not',
    'generic_and',
    'generic_or',
    'make_generic_operator',
    # Merge system
    'the_contradiction',
    'contradictory_p',
    'contradictory',
    'merge',
    'assign_merge_operation',
    'nothing_p',
    'any_p',
    # Intervals
    'Interval',
    'make_interval',
    'interval_low',
    'interval_high',
    'add_interval',
    'mul_interval',
    'mul_interval_complete',
    'sub_interval',
    'div_interval',
    'square_interval',
    'sqrt_interval',
    'empty_interval_p',
    'intersect_intervals',
    'to_interval',
    'coercing',
    # Logs
    'LogEntry',
    'Log',
    'make_log_entry',
    'make_log',
    'append_log_entry',
    'count_log',
    'timestamps',
    'entries_between',
    'window_log',
    'count_window',
    'latest',
    'latest_payload',
    'latest_before',
    'filter_log',
    'filter_after_timestamp',
    'filter_before_timestamp',
    'map_log',
    'map_payload_values',
    'detect_bursts_in_log',
    'singleton_log',
    'entry_to_log',
    'normalize_log_increment',
    # Interval propagators
    # 'interval_adder',
    # 'interval_multiplier',
    # 'interval_subtractor',
    # 'interval_divider',
    # 'interval_squarer',
    # 'interval_sqrter',
    # Dependency (supported values)
    'Supported',
    'supported',
    'more_informative_support',
    'merge_supports',
    # Generic conditionals
    'true_p',
    'ignore_first',
    'get_generic_true',
    'get_generic_ignore_first',
    # TMS
    'Tms',
    'make_tms',
    'Hypothetical',
    'hypothetical',
    'hypothetical_p',
    'tms_p',
    'tms_merge',
    'tms_assimilate',
    'tms_query',
    'strongest_consequence',
    'tms_unpacking',
    'full_tms_unpacking',
    'kick_out',
    'bring_in',
    'premise_in',
    'mark_premise_in',
    'mark_premise_out',
    'premise_nogoods',
    'set_premise_nogoods',
    'initialize_tms',
    'get_worldview_number',
    # Contradiction tracking - prefer getter functions
    'get_number_of_calls_to_fail',  # Preferred
    'get_last_nogood',               # Preferred
    'get_contradictions',            # Preferred
    'number_of_calls_to_fail',       # Deprecated: may be stale after init
    'last_nogood',                   # Deprecated: may be stale after init
    'contradictions_history',        # Deprecated: may be stale after init
    'describe_nogood',
    'describe_last_contradiction',
    'get_contradiction_details',
    'set_contradiction_verbose',
    'process_nogood',
    'process_contradictions',
    'process_one_contradiction',
    'assimilate_nogood',
    'pairwise_union',
    'TmsContradiction',
    'to_tms',
    # Guessing machine
    'binary_amb',
    'require',
    'abhor',
    'require_distinct',
    'one_of',
    'one_of_the_cells',
    # CDCL enhancements
    'CDCLEngine',
    'CDCLStats',
    'enable_cdcl',
    'disable_cdcl',
    'reset_cdcl',
    'full_reset_cdcl',
    'cdcl_enabled',
    'cdcl_stats',
    'cdcl_conflicts',
    'cdcl_backjumps',
    'cdcl_levels_saved',
    'get_cdcl_engine',
    # SAT/SMT Solver Export
    'NetworkCompiler',
    'SolverBackend',
    'ConstraintType',
    'SolverResult',
    'NetworkAnalyzer',
    'compile_from_roots',
    'solve_from_roots',
    'TranslationMode',
    'UnsupportedTranslationError',
    'TrueHybridNetwork',
    'solve_hybrid',
    'solve_hybrid_from_existing_network',
    'solve_dimacs',
    'solve_smtlib2',
    # Circuit visualization
    'CircuitTrace',
    'CircuitCapture',
    'capture_circuit',
    'register_trace_spec',
    'CellAnnotation',
    'PremiseColorMap',
    # Expression-oriented API
    'Expr',
    'amb',
    'require_distinct_exprs',
    'const',
    'query',
]

# Expression-oriented API (cleaner syntax)
from .expression import (
    Expr,
    amb,
    require_distinct_exprs,
    const,
    query,
)

# Import expression-aware require/abhor that handle both Expr and Cell
from .expression import require as _expr_require, abhor as _expr_abhor

# Override the guessing_machine versions with Expr-aware versions
def require(expr_or_cell):
    """Require a boolean expression/cell to be True. Works with Expr or Cell."""
    return _expr_require(expr_or_cell)

def abhor(expr_or_cell):
    """Forbid a boolean expression/cell. Works with Expr or Cell."""
    return _expr_abhor(expr_or_cell)