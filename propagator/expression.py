"""
Expression-oriented frontend for propagators.

This module provides a more natural, expression-like syntax for building
propagator networks, hiding the "assembly language" of explicit cell creation.

Instead of:
    baker = Cell('baker')
    one_of([1,2,3,4,5], baker)
    b_eq_5 = Cell('b_eq_5')
    five = Cell('five')
    constant(5, five)
    eq(five, baker, b_eq_5)
    abhor(b_eq_5)

You can write:
    baker = amb(1, 2, 3, 4, 5, name='baker')
    abhor(baker.eq(5))

The key ideas:
1. Cells support arithmetic operators that auto-create result cells
2. Comparison methods (.eq, .gt, .lt) return boolean cells
3. Boolean expressions compose with &, |, ~ operators
4. Bidirectional constraints are expressed with .constrain_*() methods
5. Lazy/compound networks are expressed with the lazy() decorator
6. amb() returns a cell directly
7. Expressions compose naturally: (smith - fletcher).abs().eq(1)

═══════════════════════════════════════════════════════════════════════════════
TRANSLATION SEMANTICS
═══════════════════════════════════════════════════════════════════════════════

Every Expr(c) is a thin wrapper over a Cell c. Building an expression tree is
the same as wiring a propagator network — there is no deferred evaluation.

1. UNIDIRECTIONAL COMPUTATION
   `e1 op e2` allocates a fresh result cell r, installs a propagator that
   reads e1.cell and e2.cell and writes to r, then returns Expr(r).

       e1 + e2  ≡  r = Cell(); adder(e1.cell, e2.cell, r); Expr(r)

   Information flows *only* from inputs to r. If r is later constrained,
   that information does NOT back-propagate to e1 or e2.

   Type dispatch: all arithmetic lowers to generic_add / generic_sub / …
   These are GenericOperator instances. Type-specific handlers (Interval,
   Supported, Tms) are registered on those operators. Because this layer
   calls the same propagator constructors as the assembly layer, ALL type
   extensions are inherited automatically — no code changes needed here
   when a new type is added to primitives.py.

2. BIDIRECTIONAL CONSTRAINTS  (.constrain_sum / .constrain_product / …)
   `e1.constrain_sum(e2, total)` installs three propagators (forward +
   two reverse directions) so information can flow in any direction.
   These mirror the multidirectional helpers in primitives.py exactly.

       e1.constrain_sum(e2, total)
           ≡  sum_constraint(e1.cell, e2.cell, total.cell)

   All three cells must be Expr objects (or Cells) supplied by the caller.
   No fresh cell is allocated because the caller controls all three roles.

3. BOOLEAN COMPOSITION  (& | ~)
   Boolean-valued Exprs (results of .eq / .gt / etc.) can be combined.

       e1 & e2  ≡  r = Cell(); conjoiner(e1.cell, e2.cell, r); Expr(r)
       e1 | e2  ≡  r = Cell(); disjoiner(e1.cell, e2.cell, r); Expr(r)
       ~e       ≡  r = Cell(); inverter(e.cell, r);             Expr(r)

   Python's __and__ / __or__ / __invert__ are used so that `&` / `|` / `~`
   work, but NOT `and` / `or` / `not` (those cannot be overloaded in Python
   as they short-circuit on truthiness, not on cell identity).

4. CONDITIONAL ROUTING  cond(predicate, if_true, if_false)
   Routes one of two cells to a result depending on a boolean cell.

       cond(p, t, f)
           ≡  r = Cell(); conditional(p.cell, t.cell, f.cell, r); Expr(r)

   This is unidirectional: p, t, f are inputs, r is output.

5. LAZY / COMPOUND NETWORKS  @lazy(*triggers)
   Some networks must defer construction until at least one trigger cell has
   content (e.g., recursive networks, fall_duration). The lazy() decorator
   wraps compound_propagator.

       @lazy(time)
       def fall_network():
           g = const(Interval(9.789, 9.832))
           ...

   The decorated function is called at most once, and only after one of the
   trigger expressions has received content.

6. EXTENSION POINT  register_expr_operator(symbol, propagator_ctor)
   Adds a new binary operator to Expr without subclassing.

       register_expr_operator('%', modulo_propagator)
       # Then: expr1 % expr2 works

7. ESCAPE HATCH
   `expr.cell` gives the raw Cell whenever the expression layer is
   insufficient and assembly-level control is needed.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from .cell import Cell, compound_propagator
from .primitives import (
    constant,
    conditional as cell_conditional,
    eq as eq_propagator, gt as gt_propagator,
    adder, subtractor, multiplier, divider,
    absolute_value as abs_propagator,
    conjoiner, disjoiner, inverter,
    product, sum_constraint, difference, quadratic,
)
from .guessing_machine import one_of, require as require_cell, abhor as abhor_cell, require_distinct as require_distinct_cells

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A propagator constructor: takes N input Cells and one output Cell, returns None.
# The exact arity depends on the operation; Callable[..., None] covers all arities.
PropagatorCtor = Callable[..., None]

# Anything that can be used where an Expr is expected.
ExprOrCell = Union['Expr', Cell]

# TypeVar for the class-decorator that installs promoted operators.
_ExprClass = TypeVar('_ExprClass', bound=type)

# ---------------------------------------------------------------------------
# Extension registry: symbol → propagator_constructor
# Allows adding new binary operators without subclassing Expr.
# ---------------------------------------------------------------------------
_BINARY_OPS: Dict[str, PropagatorCtor] = {}


def register_expr_operator(symbol: str, propagator_ctor: PropagatorCtor) -> None:
    """
    Register a new binary operator on Expr.

    After registration, instances of Expr gain a method named after the symbol
    accessible via Expr._dispatch(symbol, other).  This does NOT install a
    Python dunder — use a thin subclass or wrapper if you need syntax like %.

    Args:
        symbol: A short string key (e.g. '%', 'xor').
        propagator_ctor: A propagator constructor with signature
                         (input1: Cell, input2: Cell, output: Cell) -> None.

    Example:
        >>> from propagator.primitives import modulo_propagator
        >>> register_expr_operator('%', modulo_propagator)
        >>> result = baker._dispatch('%', 5)
    """
    _BINARY_OPS[symbol] = propagator_ctor


# =============================================================================
# Operator promotion
#
# Every method body in the expression layer has the identical shape:
#
#   def __add__(self, other):
#       other_e = self._to_expr(other)
#       r = self._make_result_cell('+', self, other_e)
#       adder(self._cell, other_e._cell, r)
#       return Expr(r)
#
# The ONLY thing that varies is (dunder_name, propagator_ctor, symbol).
# So we declare a compact table and generate the methods automatically.
# Adding a new primitive operator requires exactly ONE new line below.
#
# Columns:
#   python_dunder : str          — method name to install on Expr
#   propagator    : callable     — propagator constructor (inputs..., output)
#   symbol        : str          — label used in auto-created cell names
#   has_reverse   : bool         — whether to also install __r<dunder>
#
# Note on `has_reverse`:
#   The reverse dunder `__radd__` is called when the LEFT operand does not
#   know how to handle the operation (e.g. `5 + expr`). For non-commutative
#   ops (-, /) the operand order in the propagator call must be swapped:
#       __rsub__(self, other)  means  other - self
#       → subtractor(other_cell, self._cell, result)
# =============================================================================

_BINARY_PROMOTE: List[Tuple[str, PropagatorCtor, str, bool]] = [
    # (dunder,          propagator,     symbol,  has_reverse)
    ('__add__',         adder,          '+',     True),
    ('__sub__',         subtractor,     '-',     True),
    ('__mul__',         multiplier,     '*',     True),
    ('__truediv__',     divider,        '/',     True),
    ('__and__',         conjoiner,      'and',   True),
    ('__or__',          disjoiner,      'or',    True),
    # Named comparison methods (no reverse — these are not dunder ops)
    ('eq',              eq_propagator,  '==',    False),
    ('gt',              gt_propagator,  '>',     False),
]

_UNARY_PROMOTE: List[Tuple[str, PropagatorCtor, str]] = [
    # (dunder,      propagator,       symbol)
    ('__abs__',     abs_propagator,   'abs'),
    ('__invert__',  inverter,         'not'),
]


def _install_promoted_operators(cls: _ExprClass) -> _ExprClass:
    """
    Class decorator: auto-install operator methods on Expr from the promotion
    tables.  Runs once at class-definition time; zero runtime overhead.
    """
    def _make_fwd(ctor: PropagatorCtor, sym: str) -> Callable[['Expr', Any], 'Expr']:
        """Forward binary: self op other."""
        def method(self: 'Expr', other: Any) -> 'Expr':
            other_e = cls._to_expr(other)
            r = cls._make_result_cell(sym, self, other_e)
            ctor(self._cell, other_e._cell, r)
            return Expr(r)
        method.__name__ = sym
        return method

    def _make_rev(ctor: PropagatorCtor, sym: str) -> Callable[['Expr', Any], 'Expr']:
        """Reverse binary: other op self  (operand order swapped in ctor call)."""
        def method(self: 'Expr', other: Any) -> 'Expr':
            other_e = cls._to_expr(other)
            r = cls._make_result_cell(sym, other_e, self)
            ctor(other_e._cell, self._cell, r)
            return Expr(r)
        method.__name__ = 'r' + sym
        return method

    def _make_unary(ctor: PropagatorCtor, sym: str) -> Callable[['Expr'], 'Expr']:
        def method(self: 'Expr') -> 'Expr':
            r = cls._make_result_cell(sym, self)
            ctor(self._cell, r)
            return Expr(r)
        method.__name__ = sym
        return method

    for dunder, ctor, sym, has_rev in _BINARY_PROMOTE:
        setattr(cls, dunder, _make_fwd(ctor, sym))
        if has_rev:
            r_dunder = '__r' + dunder[2:]   # __add__ → __radd__
            setattr(cls, r_dunder, _make_rev(ctor, sym))

    for dunder, ctor, sym in _UNARY_PROMOTE:
        setattr(cls, dunder, _make_unary(ctor, sym))

    return cls


@_install_promoted_operators
class Expr:
    """
    Expression wrapper around a Cell that supports operator syntax.

    Arithmetic, comparison, and boolean operators are auto-installed from
    _BINARY_PROMOTE / _UNARY_PROMOTE at class-definition time via the
    @_install_promoted_operators decorator.  Adding a new primitive operator
    to the expression layer requires only one line in those tables.

    Methods remaining explicitly defined here are those that *compose*
    multiple propagators (ne, lt, ge, le, __neg__, abs) or that have a
    structurally different shape (bidirectional constraints, _dispatch).
    """
    
    def __init__(self, cell: Cell) -> None:
        self._cell: Cell = cell

    @property
    def cell(self) -> Cell:
        """Access the underlying cell."""
        return self._cell

    @property
    def content(self) -> Any:
        """Access the cell's content directly (None when no information yet)."""
        return self._cell.content

    @property
    def name(self) -> Optional[str]:
        """Get the cell's name, or None if unnamed."""
        return self._cell.name
    
    def __repr__(self) -> str:
        return f"Expr({self._cell.describe()})"
    
    # =========================================================================
    # Helper to ensure we're working with Expr objects
    # =========================================================================
    
    @staticmethod
    def _to_expr(value: Any, name: Optional[str] = None) -> 'Expr':
        """Convert a value to an Expr, creating a constant cell if needed."""
        if isinstance(value, Expr):
            return value
        if isinstance(value, Cell):
            return Expr(value)
        # It's a raw value - create a constant cell
        c = Cell(name=name, context=f"const:{value}")
        constant(value, c)
        return Expr(c)

    @staticmethod
    def _make_result_cell(op: str, *operands: Any) -> Cell:
        """Create a result cell with a descriptive name."""
        operand_names: List[str] = []
        for o in operands:
            if isinstance(o, Expr):
                operand_names.append(o._cell.describe())
            else:
                operand_names.append(str(o))
        context = f"{op}({', '.join(operand_names)})"
        return Cell(context=context, role="result")
    
    # =========================================================================
    # Methods auto-installed by @_install_promoted_operators:
    #   __add__, __radd__, __sub__, __rsub__, __mul__, __rmul__
    #   __truediv__, __rtruediv__, __and__, __rand__, __or__, __ror__
    #   __abs__, __invert__
    #   eq, gt
    # =========================================================================

    # =========================================================================
    # Remaining explicit methods: those that compose multiple propagators
    # =========================================================================

    def __neg__(self) -> 'Expr':
        """-self  (0 - self, composes constant + subtractor)"""
        return self._to_expr(0) - self

    def abs(self) -> 'Expr':
        """Absolute value (method alias for __abs__, auto-installed)."""
        return self.__abs__()

    # --- Derived comparisons: compose eq/gt + inverter ---

    def ne(self, other: Any) -> 'Expr':
        """self != other  (= ~self.eq(other))"""
        result = self._make_result_cell('!=', self, self._to_expr(other))
        inverter(self.eq(other)._cell, result)
        return Expr(result)

    def lt(self, other: Any) -> 'Expr':
        """self < other  (= other.gt(self))"""
        return self._to_expr(other).gt(self)

    def ge(self, other: Any) -> 'Expr':
        """self >= other  (= ~self.lt(other))"""
        result = self._make_result_cell('>=', self, self._to_expr(other))
        inverter(self.lt(other)._cell, result)
        return Expr(result)

    def le(self, other: Any) -> 'Expr':
        """self <= other  (= ~self.gt(other))"""
        result = self._make_result_cell('<=', self, self._to_expr(other))
        inverter(self.gt(other)._cell, result)
        return Expr(result)

    # =========================================================================
    # Bidirectional constraint methods
    #
    # These wire the multidirectional helpers from primitives.py so information
    # can flow in any direction.  All three cells must be supplied by the caller;
    # no fresh cell is allocated here.
    #
    # Translation:
    #   a.constrain_sum(b, total)
    #       ≡  sum_constraint(a.cell, b.cell, total.cell)
    #          which installs: adder(a,b,total) + subtractor(total,a,b)
    #                                           + subtractor(total,b,a)
    # =========================================================================

    def constrain_sum(self, other: ExprOrCell, total: ExprOrCell) -> None:
        """
        Impose: total = self + other  (bidirectional).

        Any one of the three cells can be the unknown:
            self + other → total
            total - other → self
            total - self → other

        Args:
            other: The second addend (Expr or Cell).
            total: The sum cell (Expr or Cell).

        Example:
            x = cell('x'); y = cell('y'); z = cell('z')
            x.constrain_sum(y, z)   # z = x + y, in all directions
        """
        other_expr = self._to_expr(other)
        total_expr = self._to_expr(total) if not isinstance(total, Expr) else total
        sum_constraint(self._cell, other_expr._cell, total_expr._cell)

    def constrain_product(self, other: ExprOrCell, total: ExprOrCell) -> None:
        """
        Impose: total = self * other  (bidirectional).

        Any one of the three cells can be the unknown:
            self * other → total
            total / other → self
            total / self → other

        Example:
            x.constrain_product(y, z)   # z = x * y, in all directions
        """
        other_expr = self._to_expr(other)
        total_expr = self._to_expr(total) if not isinstance(total, Expr) else total
        product(self._cell, other_expr._cell, total_expr._cell)

    def constrain_diff(self, other: ExprOrCell, diff: ExprOrCell) -> None:
        """
        Impose: diff = self - other  (bidirectional).

        Any one of the three cells can be the unknown:
            self - other → diff
            diff + other → self
            self - diff → other

        Example:
            x.constrain_diff(y, d)   # d = x - y, in all directions
        """
        other_expr = self._to_expr(other)
        diff_expr = self._to_expr(diff) if not isinstance(diff, Expr) else diff
        difference(self._cell, other_expr._cell, diff_expr._cell)

    def constrain_square(self, squared: ExprOrCell) -> None:
        """
        Impose: squared = self²  (bidirectional).

        Either cell can be the unknown:
            self → squared = self²
            squared → self = sqrt(squared)

        Example:
            x.constrain_square(x2)   # x2 = x², or x = sqrt(x2)
        """
        sq_expr = self._to_expr(squared) if not isinstance(squared, Expr) else squared
        quadratic(self._cell, sq_expr._cell)

    # =========================================================================
    # Extension dispatch
    # =========================================================================

    def _dispatch(self, symbol: str, other: Any) -> 'Expr':
        """
        Invoke a registered binary operator by symbol.

        Example:
            register_expr_operator('%', modulo_propagator)
            result = baker._dispatch('%', 5)
        """
        ctor = _BINARY_OPS.get(symbol)
        if ctor is None:
            raise KeyError(
                f"No expression operator registered for {symbol!r}. "
                f"Use register_expr_operator({symbol!r}, propagator_ctor) first."
            )
        other_expr = self._to_expr(other)
        result = self._make_result_cell(symbol, self, other_expr)
        ctor(self._cell, other_expr._cell, result)
        return Expr(result)


# =============================================================================
# Top-level expression-oriented API
# =============================================================================

def amb(*values: Any, name: Optional[str] = None) -> Expr:
    """
    Create an ambiguous choice among values.
    
    Returns an Expr that will be constrained to one of the given values.
    
    Example:
        baker = amb(1, 2, 3, 4, 5, name='baker')
    """
    cell = Cell(name=name)
    one_of(list(values), cell)
    return Expr(cell)


def require(expr: ExprOrCell) -> None:
    """
    Require a boolean expression to be True.
    
    Example:
        require(miller.gt(cooper))
    """
    if isinstance(expr, Expr):
        require_cell(expr._cell)
    else:
        require_cell(expr)


def abhor(expr: ExprOrCell) -> None:
    """
    Forbid a boolean expression from being True.
    
    Example:
        abhor(baker.eq(5))
    """
    if isinstance(expr, Expr):
        abhor_cell(expr._cell)
    else:
        abhor_cell(expr)


def require_distinct_exprs(exprs: Sequence[ExprOrCell]) -> None:
    """
    Require all expressions to have distinct values.
    
    Example:
        require_distinct_exprs([baker, cooper, fletcher, miller, smith])
    """
    cells = [e._cell if isinstance(e, Expr) else e for e in exprs]
    require_distinct_cells(cells)


def const(value: Any, name: Optional[str] = None) -> Expr:
    """
    Create a constant expression.
    
    Example:
        five = const(5, 'five')
    """
    cell = Cell(name=name, context=f"const:{value}")
    constant(value, cell)
    return Expr(cell)


# =============================================================================
# Query functions
# =============================================================================

def cell(name: Optional[str] = None) -> Expr:
    """
    Create a named, initially-empty cell wrapped in an Expr.

    This is the expression-layer analog of ``Cell(name=name)``.
    Use it when you need a cell whose value will be constrained
    by bidirectional constraint methods rather than by amb().

    Example::

        x = cell('x')
        y = cell('y')
        z = cell('z')
        x.constrain_sum(y, z)    # z = x + y, in all directions
        z.cell.add_content(10)
        x.cell.add_content(3)
        # y now has content 7
    """
    return Expr(Cell(name=name))


def cond(predicate: ExprOrCell, if_true: ExprOrCell, if_false: ExprOrCell) -> Expr:
    """
    Conditional routing: route if_true or if_false to a result cell.

    Translation::

        cond(p, t, f)
            ≡  r = Cell()
               conditional(p.cell, t.cell, f.cell, r)
               Expr(r)

    This is unidirectional: p, t, f are inputs; the returned Expr is output.
    The value flows from whichever branch the predicate selects.

    Args:
        predicate: Boolean-valued Expr (e.g. result of .gt(), .eq(), &, |).
        if_true:   Value to route when predicate is True.
        if_false:  Value to route when predicate is False.

    Example::

        # larger = max(x, y)
        larger = cond(x.gt(y), x, y)
    """
    def _cell_of(e: ExprOrCell) -> Cell:
        if isinstance(e, Expr):
            return e._cell
        return e

    p_cell = _cell_of(predicate)
    t_cell = _cell_of(if_true)
    f_cell = _cell_of(if_false)
    result = Cell(context=f"cond({p_cell.describe()},{t_cell.describe()},{f_cell.describe()})")
    cell_conditional(p_cell, t_cell, f_cell, result)
    return Expr(result)


def lazy(*triggers: ExprOrCell) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """
    Decorator: defer network construction until at least one trigger has content.

    Wraps ``compound_propagator`` so complex or recursive sub-networks can be
    expressed in expression style.  The decorated function is called *at most
    once*, and only after one of the trigger expressions receives content.

    Translation::

        @lazy(time)
        def fall_network():
            ...

        ≡  compound_propagator([time.cell], fall_network)

    Args:
        *triggers: Expr or Cell objects that gate construction.

    Returns:
        A decorator that registers the build function with compound_propagator
        and returns it unchanged.

    Example::

        from propagator.expression import cell, const, lazy
        from propagator.intervals import make_interval

        time   = cell('time')
        height = cell('height')

        @lazy(time)
        def fall_network():
            g        = const(make_interval(9.789, 9.832))
            one_half = const(make_interval(0.5, 0.5))
            t2       = cell('t_squared')
            gt2      = cell('g_t_squared')
            time.constrain_square(t2)
            g.constrain_product(t2, gt2)
            one_half.constrain_product(gt2, height)
    """
    trigger_cells: List[Cell] = [e._cell if isinstance(e, Expr) else e for e in triggers]

    def decorator(build_fn: Callable[[], None]) -> Callable[[], None]:
        compound_propagator(trigger_cells, build_fn)
        return build_fn

    return decorator


def query(expr: ExprOrCell) -> Any:
    """
    Query the current value of an expression.

    Returns the TMS-queried value if using TMS, otherwise the raw content.
    """
    from .nothing import nothing_p
    from .tms import tms_query, tms_p

    c = expr._cell if isinstance(expr, Expr) else expr
    content = c.content

    if nothing_p(content):
        return None
    if tms_p(content):
        return tms_query(content)
    return content
