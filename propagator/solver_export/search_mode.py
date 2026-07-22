"""
Search mode control for hybrid propagator + SMT execution.

Provides clear, intention-revealing context managers that replace
the confusing ``set_auto_run(True/False)`` global flag.

Three modes
===========

=============== ============================================= ================================
Mode             During construction                         Best for
=============== ============================================= ================================
DEFER_TO_SMT     Wire only. Zero search.                      Any network with SMT available.
                 SMT owns 100% of search.                     The default. Safe and fast.

EAGER_TMS        TMS search runs in full during construction. Pure propagator search.
                 CDCL learning happens as constraints are     No SMT available, or small
                 added.                                       networks where TMS is fast enough.

PROPAGATE_ONLY   Propagation runs but guessing is disabled.   Networks with arithmetic
                 Domains are narrowed, values fixed, but      constraints where propagation
                 no choices are explored. Then SMT solves.    alone can narrow far enough
                                                              to make SMT instantaneous.
=============== ============================================= ================================

Usage
=====

Recommended (DEFER_TO_SMT)::

    from propagator.solver_export import search_mode, solve, SolveMode

    with search_mode(SolveMode.SMT_ITERATIVE):
        x = Cell(name='x')
        y = Cell(name='y')
        one_of({1, 2, 3}, x)
        one_of({1, 2, 3}, y)
        adder(x, y, z)
        constant(5, z)

    result = solve([x, y, z], mode=SolveMode.SMT_ITERATIVE)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from propagator.scheduler import set_auto_run, get_auto_run
from propagator.guessing_machine import set_guessing_enabled, is_guessing_enabled
from .solve import SolveMode as _SolveModeSolve


class SearchMode:
    """
    Namespace for construction-time search mode identifiers.

    These are NOT the same as ``SolveMode``, which controls the *solve*
    phase. ``SearchMode`` controls what happens during *construction*.

    =============== ============================================= ================================
    Mode            During construction                          Best for
    =============== ============================================= ================================
    DEFER_TO_SMT    Wire only, zero search.                      (default) SMT available.
                    SMT owns all search.
    EAGER_TMS       Full TMS search (CDCL/DDB) during build.     No SMT. Small networks.
    PROPAGATE_ONLY  Propagation without guessing.                Arithmetic with SMT.
                    Domains narrow but no choices explored.
    =============== ============================================= ================================
    """
    DEFER_TO_SMT = "defer_to_smt"
    EAGER_TMS = "eager_tms"
    PROPAGATE_ONLY = "propagate_only"


def search_mode(mode: str | SearchMode) -> 'SearchContext':
    """
    Context manager that sets the construction-time execution mode.

    Args:
        mode: One of ``SearchMode.DEFER_TO_SMT``, ``SearchMode.EAGER_TMS``,
              or ``SearchMode.PROPAGATE_ONLY`` (also accepts strings).

    Example::

        with search_mode(SearchMode.DEFER_TO_SMT):
            # Wire network. Zero search runs during this block.
            one_of({1, 2, 3}, x)
            adder(x, y, z)

        # Outside the block, auto_run and guessing restore to defaults.
        result = solve([x, y, z])
    """
    mode = str(mode)
    return SearchContext(_mode_to_config(mode))


def _mode_to_config(mode: str) -> dict:
    """Map a search mode string to auto_run + guessing config."""
    if mode == "eager_tms":
        return {"auto_run": True, "guessing": True}
    elif mode == "propagate_only":
        return {"auto_run": True, "guessing": False}
    else:  # defer_to_smt (default)
        return {"auto_run": False, "guessing": True}


class SearchContext:
    """Internal: holds the config and restores state on exit."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._prev_auto_run: bool = True
        self._prev_guessing: bool = True

    def __enter__(self) -> 'SearchContext':
        self._prev_auto_run = get_auto_run()
        self._prev_guessing = is_guessing_enabled()
        set_auto_run(self._config["auto_run"])
        set_guessing_enabled(self._config["guessing"])
        return self

    def __exit__(self, *args: object) -> None:
        set_auto_run(self._prev_auto_run)
        set_guessing_enabled(self._prev_guessing)
