"""
The Nothing Sentinel for propagator networks.

This is a foundational module with NO internal dependencies - it can be
imported by any other module without causing circular imports.

In Scheme: (define nothing #(*the-nothing*))

Using a dedicated Nothing sentinel instead of Python's None provides:

1. SEMANTIC CLARITY: "nothing" means "no information yet" in propagator 
   semantics. This is distinct from None which could mean "the value is None"
   or "missing/error". A dedicated sentinel makes the propagator concept
   explicit in the type system.

2. TYPE SAFETY: Functions can distinguish between "cell has no information"
   (nothing) and "function returned None" (an actual None value). This is
   especially important when propagators compute optional values.

3. CONSISTENCY WITH SCHEME: The original implementation uses a distinct
   nothing object. This makes the Python code more directly translatable.

4. CLEARER ERROR MESSAGES: Seeing "Nothing" in tracebacks is more informative
   than seeing "None" when debugging propagator networks.
"""


class _Nothing:
    """
    Singleton class representing "no information" in a cell.
    
    This is semantically distinct from None - it means the cell has not
    yet received any information, not that it contains a null value.
    
    Scheme equivalent:
        (define nothing #(*the-nothing*))
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __repr__(self):
        return "Nothing"
    
    def __bool__(self):
        # Nothing is falsy, similar to None, for convenience in conditionals
        return False
    
    def __eq__(self, other):
        # Only equal to itself (identity comparison)
        return self is other
    
    def __hash__(self):
        return hash("Nothing")


# The single instance of nothing
nothing = _Nothing()

 
def nothing_p(x) -> bool:
    """
    Check if a value is nothing (no information).

    Scheme equivalent:
        (define (nothing? thing) (eq? thing nothing))

    This is a strict identity check against the dedicated sentinel only.
    Python's None is NOT nothing -- it is an ordinary, storable value, same
    as any other. A cell can legitimately hold None as its content (e.g.
    `cell.add_content(None)`), and that is distinguishable from a cell that
    has received no information at all (whose content is the `nothing`
    object).

    Plain Python functions can still use `return None` to mean "I have
    nothing to contribute" without knowing about this sentinel: functions
    lifted via `lift_to_cell_contents`/`function_to_propagator_constructor`
    (see cell.py) coerce a wrapped function's own None return into nothing
    at that boundary. Only a None that reaches a cell some other way (e.g.
    `constant(None, cell)`, or `cell.add_content(None)` directly) is stored
    as real data.
    """
    return x is nothing
