"""
Cell module for the propagators library.
"""
from typing import List, Callable, Union

# nothing.py has zero internal dependencies, so it's always safe to import
# at module level (unlike merge.py, which cell.py still imports lazily
# inside add_content() to avoid a genuine cycle: merge.py registers
# handlers that reference types defined in modules that import cell.py).
from .nothing import nothing, nothing_p

# Import scheduler functions - these handle queuing instead of immediate execution
from .scheduler import (
    alert_propagators_and_maybe_run,
    alert_all_propagators as _scheduler_alert_all,
    all_propagators,
    run,
    initialize_scheduler,
    is_scheduler_running,
)


# Global registry of propagators for conservative invalidation
# Note: This is now managed by the scheduler, but we keep a reference here
# for backward compatibility with code that accesses _all_propagators directly
def _get_all_propagators() -> List[Callable]:
    """Get all registered propagators from the scheduler."""
    return all_propagators()

# For backward compatibility
_all_propagators: List[Callable] = []  # Deprecated - use all_propagators() instead


def alert_propagators(propagators: Union[Callable, List[Callable]]) -> None:
    """
    Alert one or more propagators to execute.
    
    This schedules propagators for execution via the scheduler.
    If auto-run is enabled and we're not already inside run(),
    the scheduler will execute them.
    """
    alert_propagators_and_maybe_run(propagators)


def alert_all_propagators() -> None:
    """
    Alert all registered propagators.
    
    This schedules all known propagators for execution.
    """
    _scheduler_alert_all()
    # Run if not already running
    if not is_scheduler_running():
        run()

class Cell:
    """
    A basic cell class for propagation simulations.
    
    Cells can track debugging context:
        name: Human-readable name (e.g., "baker")
        role: What role this cell plays (e.g., "predicate", "intermediate", "output")
        parent: The output cell this is helping compute (for intermediates)
        context: Additional context about the cell's purpose
    """
    def __init__(self, name: str = None, role: str = None, parent: 'Cell' = None, context: str = None):
        # Initialize neighbors as an actual empty list (not a type annotation)
        self.neighbors: List[Callable] = []
        self.content = nothing  # The dedicated sentinel, not None -- None is a storable value
        self.name = name  # Optional human-readable name for debugging
        self.role = role  # Role: "predicate", "intermediate", "output", etc.
        self.parent = parent  # Parent output cell (for tracking lineage)
        self.context = context  # Additional context string

    def describe(self) -> str:
        """
        Return a human-readable description of this cell.
        
        Tries multiple strategies to provide meaningful context:
        1. Use explicit name if set
        2. Use parent name + role if available
        3. Use context string if provided
        4. Fall back to hash-based ID
        """
        if self.name:
            return self.name
        
        # Try to derive name from parent
        if self.parent is not None:
            parent_name = self.parent.name if self.parent.name else f"Cell@{id(self.parent) % 10000}"
            if self.role:
                return f"{self.role}({parent_name})"
            return f"for:{parent_name}"
        
        # Use context if provided
        if self.context:
            return self.context
        
        # Fall back to ID
        return f"Cell@{id(self) % 10000}"

    def __repr__(self) -> str:
        """Return string representation of the cell."""
        desc = self.describe()
        if self.name:
            return f"Cell({desc}={self.content})"
        return f"Cell({desc}, content={self.content})"
    
    def new_neighbor(self, new_neighbor) -> None:
        if new_neighbor not in self.neighbors:
            # list.insert() mutates in place and returns None, don't assign it!
            self.neighbors.insert(0, new_neighbor)
            alert_propagators(new_neighbor)

    def add_content(self, increment) -> None:
        # Import merge/contradictory_p from merge.py only when needed
        from .merge import merge, contradictory_p

        if nothing_p(increment):
            return "ok"  # Return symbol of ok - nothing adds no information
        elif nothing_p(self.content):
            self.content = increment
            for neighbor in self.neighbors:
                alert_propagators(neighbor)
        else:
            # Merge with existing content using generic merge
            answer = merge(self.content, increment)
            
            if answer == self.content:
                return "ok"  # No change, so no need to alert neighbors
            elif contradictory_p(answer):
                raise Exception("Ack! Inconsistency!")
            else:
                # Update content and alert neighbors
                self.content = answer
                alert_propagators(self.neighbors)
    
    # not necessary? This is message passing style, not necessary with oop
    def me(self,message):
        if message == 'new-neighbor!':
            return self.new_neighbor  # Return the function
        elif message == 'add-content':
            return self.add_content
        elif message == 'content':
            return self.content  # Return the value
        else:
            raise Exception("Unknown message", message)


def listify(x):
    return x if isinstance(x, list) else [x]

def lift_to_cell_contents(f: Callable) -> Callable:
    """
    Lifts a regular function to work with cell contents.

    This higher-order function takes a normal function and returns a new
    function that handles the 'nothing' sentinel automatically:
    - If any argument is nothing, returns nothing (short-circuits without
      calling f).
    - Otherwise, applies the original function to the arguments. If f
      itself returns a bare None -- the ordinary Python idiom for "I have
      nothing to contribute" -- that None is coerced to nothing here, so
      plain functions can keep using `return None` to mean "no update"
      without importing or knowing about the sentinel. A None returned by
      the CALLER explicitly (via cell.add_content(None) directly, not
      through this lifting) is unaffected by this function and is stored
      as real data -- this is the one deliberate seam between "None as an
      ordinary Python idiom" and "None as data."

    Args:
        f: A function to lift to work with cell contents

    Returns:
        A new function that handles nothing values and applies f to non-nothing arguments

    Example:
        >>> add = lift_to_cell_contents(lambda x, y: x + y)
        >>> add(5, 10)  # Returns 15
        >>> add(5, nothing)  # Returns nothing (short-circuited, f never called)
    """
    def lifted(*args):
        # If any argument is nothing, return nothing without calling f.
        if any(nothing_p(arg) for arg in args):
            return nothing
        # Otherwise, apply the original function to all arguments. A plain
        # function's own `return None` (meaning "nothing to contribute")
        # becomes the sentinel here, not a stored None.
        result = f(*args)
        return nothing if result is None else result

    return lifted


def lift_to_provenance_aware_cell_contents(f: Callable) -> Callable:
    """
    Lift a constructor-like function through nothing, Supported, and TMS values.

    This is for structured value constructors that should preserve provenance
    from their inputs, similar to how generic arithmetic operators already do.
    When any argument is TMS, the function is evaluated through
    full_tms_unpacking; when any argument is Supported, it is evaluated through
    supported_unpacking with automatic coercion of flat arguments.
    """
    def lifted(*args):
        if any(nothing_p(arg) for arg in args):
            return nothing

        from .supported_values import coercing, supported_p, supported_unpacking, to_supported
        from .tms import full_tms_unpacking, tms_p, to_tms

        if any(tms_p(arg) for arg in args):
            return coercing(to_tms, full_tms_unpacking(f))(*args)

        if any(supported_p(arg) for arg in args):
            return coercing(to_supported, supported_unpacking(f))(*args)

        result = f(*args)
        return nothing if result is None else result

    return lifted

def propagator(neighbors: List[Cell], to_do):
    """
    Creates a propagator that connects neighbor cells to a computation.
    
    This arranges for the thunk to_do to be run at least once, and asks each
    cell in the neighbors argument to have to_do rerun if that cell's content changes.
    
    Args:
        neighbors: A cell or list of cells whose changes should trigger to_do
        to_do: A function (thunk) to run when any neighbor changes
    """
    for cell in listify(neighbors):
        cell.new_neighbor(to_do)

    # Schedule the propagator - the scheduler will track it
    alert_propagators(to_do)


def function_to_propagator_constructor(f: Callable) -> Callable:
    """
    Converts a regular function into a propagator constructor.
    
    This permits constructing propagators from primitive functions.
    The returned constructor takes cells as arguments, where the LAST cell
    is the output and all preceding cells are inputs.
    
    Args:
        f: A regular function to convert into a propagator constructor
        
    Returns:
        A propagator constructor function that takes (*inputs, output) cells
        
    Example:
        >>> adder = function_to_propagator_constructor(lambda a, b: a + b)
        >>> adder(cell_a, cell_b, cell_sum)  # sum = a + b
    """
    def propagator_constructor(*cells):
        # Split cells: last one is output, rest are inputs
        output = cells[-1]  # (car (last-pair cells))
        inputs = cells[:-1]  # (except-last-pair cells)
        
        # Lift the function to handle None values
        lifted_f = lift_to_cell_contents(f)
        
        # Create the propagator computation
        def to_do():
            # Get content from each input cell: (map content inputs)
            input_values = [cell.content for cell in inputs]
            # Apply the lifted function: (apply lifted-f ...)
            result = lifted_f(*input_values)
            # Add result to output cell
            output.add_content(result)
        
        # Register propagator with input cells only (output isn't a neighbor!)
        propagator(list(inputs), to_do)
    
    return propagator_constructor


def provenance_aware_function_to_propagator_constructor(f: Callable) -> Callable:
    """
    Convert a constructor-like function into a provenance-aware propagator.

    This is the structured-value analogue of function_to_propagator_constructor:
    it preserves Supported/TMS provenance from inputs automatically, rather than
    requiring per-example wrapper plumbing.
    """
    def propagator_constructor(*cells):
        output = cells[-1]
        inputs = cells[:-1]

        lifted_f = lift_to_provenance_aware_cell_contents(f)

        def to_do():
            input_values = [cell.content for cell in inputs]
            result = lifted_f(*input_values)
            output.add_content(result)

        propagator(list(inputs), to_do)

    return propagator_constructor


def compound_propagator(neighbors, to_build: Callable) -> None:
    """
    Compound propagator with lazy construction.
    
    A compound propagator is implemented with a procedure that will construct 
    the propagator's body on demand. We take care that it is constructed only 
    if some neighbor actually has a value, and that it is constructed only once.
    
    Args:
        neighbors: A cell or list of cells that trigger construction
        to_build: A procedure that constructs the propagator's body
        
    Scheme equivalent:
        (define (compound-propagator neighbors to-build)
          (let ((done? #f)
                (neighbors (listify neighbors)))
            (define (test)
              (if done?
                  'ok
                  (if (every nothing? (map content neighbors))
                      'ok
                      (begin
                        (set! done? #t)
                        (to-build)))))
            (propagator neighbors test)))
    """
    # Normalize neighbors to a list
    neighbor_list = listify(neighbors)
    
    # Mutable state flag - tracks whether we've built the propagator body
    done = False
    
    def test():
        nonlocal done
        
        if done:  # Already constructed
            return 'ok'
        
        # Check if every neighbor has nothing
        # (every nothing? (map content neighbors))
        if all(nothing_p(cell.content) for cell in neighbor_list):
            return 'ok'  # Don't build yet, no data available
        else:
            # At least one neighbor has content - build now, but only once
            done = True
            to_build()
    
    # Register the test function with neighbors
    propagator(neighbor_list, test)