
from propagator.cell import Cell
from propagator.primitives import constant, eq, gt, subtractor, absolute_value
from propagator.guessing_machine import abhor, one_of, require, require_distinct
from propagator.tms import tms_query, number_of_calls_to_fail, get_contradictions

def multiple_dwelling():
    baker = Cell('baker')
    cooper = Cell('cooper')
    fletcher = Cell('fletcher')
    miller = Cell('miller')
    smith = Cell('smith')

    floors = [1,2,3,4,5]

    #constrain cells to be one of the floors
    one_of(floors, baker)
    one_of(floors, fletcher)
    one_of(floors, smith)
    one_of(floors, cooper)
    one_of(floors, miller)       

    #constrain cells to be distinct
    require_distinct([baker, fletcher, smith, cooper, miller])

    # Specific constraints from the puzzle, ie clues. Abhor forbids certain cases
    b_eq_5 = Cell('b_eq_5')
    f_eq_5 = Cell('f_eq_5')
    m_gt_c = Cell('m_gt_c')
    fc = Cell('fc')
    five = Cell('five')
    as_f = Cell('as_f')
    af_c = Cell('af_c')
    c_eq_1 = Cell('c_eq_1')
    f_eq_1 = Cell('f_eq_1')
    sf = Cell('sf')
    one = Cell('one')
    s_f = Cell('s_f')
    f_c = Cell('f_c')
    constant(1, one)
    constant(5, five)

    eq(five, baker, b_eq_5); abhor(b_eq_5)
    eq(one, cooper, c_eq_1); abhor(c_eq_1)
    eq(five, fletcher, f_eq_5); abhor(f_eq_5)
    eq(one, fletcher, f_eq_1); abhor(f_eq_1)

    gt(miller, cooper, m_gt_c); require(m_gt_c)
    subtractor(smith, fletcher, s_f)
    absolute_value(s_f, as_f)
    eq(one, as_f, sf); abhor(sf)

    subtractor(fletcher, cooper, f_c)
    absolute_value(f_c, af_c)
    eq(one, af_c, fc); abhor(fc)
    return [baker, fletcher, smith, cooper, miller]

if __name__ == "__main__":
    from propagator import run, initialize_scheduler
    
    initialize_scheduler()
    residents = multiple_dwelling()
    run()
    
    values = [tms_query(cell.content) for cell in residents]
    names = ['baker', 'fletcher', 'smith', 'cooper', 'miller']

    print("=" * 50)
    print("SUPERINTENDENT PUZZLE RESULTS")
    print("=" * 50)
    
    for name, val in zip(names, values):
        floor = val.value if hasattr(val, 'value') else val
        print(f"  {name}: floor {floor}")
    
    print(f"\nContradictions encountered: {len(get_contradictions())}")
    print(f"Calls to fail: {number_of_calls_to_fail}")
    print(f"\nContradictions: {get_contradictions()}")
    for i, ng in enumerate(get_contradictions()):
        print(f"  nogood_{i}: {ng}")
