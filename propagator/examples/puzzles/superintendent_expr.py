"""
Superintendent Puzzle - Expression-Oriented Version

Compare this to superintendent_puzzle.py to see how much cleaner the
expression-oriented syntax is.

This is the "frontend" that the Radul/Sussman paper mentions in the footnote:
"Even a relatively straightforward expression-oriented frontend could let us
write something far more pleasant."

The key insight: expressions like `(smith - fletcher).abs().eq(1)` implicitly
create all the intermediate cells and wire up the propagators.
"""

from propagator.expression import amb, require, abhor, require_distinct_exprs, query
from propagator.tms import get_contradictions, number_of_calls_to_fail


def multiple_dwelling():
    """
    Solve the multiple dwelling puzzle using expression syntax.
    
    Compare to superintendent_puzzle.py - this version is much more readable!
    
    The puzzle:
    - Baker, Cooper, Fletcher, Miller, and Smith live on different floors (1-5)
    - Baker doesn't live on floor 5
    - Cooper doesn't live on floor 1
    - Fletcher doesn't live on floor 5 or 1
    - Miller lives on a higher floor than Cooper
    - Smith and Fletcher don't live on adjacent floors
    - Fletcher and Cooper don't live on adjacent floors
    """
    # Create the ambiguous choices - each person can be on any floor
    baker = amb(1, 2, 3, 4, 5, name='baker')
    cooper = amb(1, 2, 3, 4, 5, name='cooper')
    fletcher = amb(1, 2, 3, 4, 5, name='fletcher')
    miller = amb(1, 2, 3, 4, 5, name='miller')
    smith = amb(1, 2, 3, 4, 5, name='smith')
    
    # All on different floors
    require_distinct_exprs([baker, cooper, fletcher, miller, smith])
    
    # Specific constraints (the clues)
    abhor(baker.eq(5))           # Baker doesn't live on 5
    abhor(cooper.eq(1))          # Cooper doesn't live on 1
    abhor(fletcher.eq(5))        # Fletcher doesn't live on 5
    abhor(fletcher.eq(1))        # Fletcher doesn't live on 1
    
    require(miller.gt(cooper))   # Miller > Cooper
    
    # Smith and Fletcher not adjacent: |smith - fletcher| != 1
    abhor((smith - fletcher).abs().eq(1))
    
    # Fletcher and Cooper not adjacent: |fletcher - cooper| != 1
    abhor((fletcher - cooper).abs().eq(1))
    
    return [baker, cooper, fletcher, miller, smith]


if __name__ == "__main__":
    from propagator import run, initialize_scheduler
    
    initialize_scheduler()
    residents = multiple_dwelling()
    run()
    
    print("=" * 60)
    print("SUPERINTENDENT PUZZLE - EXPRESSION SYNTAX")
    print("=" * 60)
    
    names = ['baker', 'cooper', 'fletcher', 'miller', 'smith']
    for name, expr in zip(names, residents):
        val = query(expr)
        floor = val.value if hasattr(val, 'value') else val
        print(f"  {name}: floor {floor}")
    
    print()
    print(f"Contradictions encountered: {len(get_contradictions())}")
    print(f"Calls to fail: {number_of_calls_to_fail}")
