from propagator import (
    Cell,
    compound_propagator,
    constant,
    make_interval,
    product,
    quadratic,
)
from propagator.supported_values import supported
from propagator.tms import TmsContradiction, bring_in, kick_out, make_tms, tms_query


def similar_triangles_height(s_base, h_base, s, h):
    def similar_triangles_compute():
        ratio = Cell()
        product(s_base, ratio, h_base)          # ratio = h_base / s_base (via multidirectional constraint)
        product(s, ratio, h)                     # h = s * ratio (via multidirectional constraint)

    compound_propagator([s_base, h_base, s], similar_triangles_compute)

def fall_duration(time, height):

    def fall_duration_compute():
        g = Cell()
        one_half = Cell()
        t_squared = Cell()
        g_times_t_squared = Cell()

        constant(make_interval(9.789, 9.832), g)                      # g = 9.8 m/s^2
        constant(make_interval(0.5, 0.5), one_half)                   # constant 1/2
        quadratic(time, t_squared)                                    # t^2 (bidirectional via constraint)
        product(g, t_squared, g_times_t_squared)                      # g * t^2 (bidirectional via constraint)
        product(one_half, g_times_t_squared, height)                  # h = 1/2 * g * t^2 (bidirectional via constraint)

    compound_propagator([time], fall_duration_compute)

def testing_fall_duration():
    barometer_height = Cell()
    barometer_shadow = Cell()
    building_height = Cell()
    building_shadow = Cell()
    similar_triangles_height(barometer_shadow, barometer_height, building_shadow, building_height)

    building_shadow.add_content(make_tms(supported(make_interval(54.9, 55.1), ['shadows'])))
    barometer_height.add_content(make_tms(supported(make_interval(0.3, 0.32), ['shadows'])))
    barometer_shadow.add_content(make_tms(supported(make_interval(0.36, 0.37), ['shadows'])))
    print(building_height.content)

    print("Nothing much changes while there is only one source of information,")

    fall_time = Cell()
    fall_duration(fall_time, building_height)
    fall_time.add_content(make_tms(supported(make_interval(2.9, 3.1), ['fall-time']))) 
    print(building_height.content)
    print("but when we add the second experiment, the TMS remembers the deductions made in the first.")

    
    print("With an implicit global worldview, querying a TMS requires no additional input and produces the most informative value supported by premises in the current worldview:")
    print(tms_query (building_height.content))


    print("We also provide a means to remove premises from the current worldview, which in this case causes the TMS query to fall back to the lessinformative inference that can be made using only the shadows experiment:")
    kick_out ('fall-time')
    print(tms_query (building_height.content))

    print("Likewise, we can ask the system for the best answer it can give if we trust the fall-time experiment but not the shadows experiment,")
    bring_in ('fall-time')
    kick_out ('shadows')
    print(tms_query (building_height.content))

    print("which may involve some additional computation not needed heretofore, whose results are reflected in the full TMS we can observe at the end.")
    print(building_height.content)


    print("which may involve some additional computation not needed heretofore, whose results are reflected in the full TMS we can observe at the end.")
    building_height.add_content(make_tms(supported(45,['superintendent'])))
    print(building_height.content)

    print("though indeed if we trust it, it provides the best estimate we have.")
    print(tms_query (building_height.content))

    print("(and restoring our faith in the shadows experiment has no effect on the accuracy of this answer).")
    bring_in ('shadows')
    print(tms_query (building_height.content))

    print("If we now turn our attention to the height of the barometers we have been dropping and giving away, we notice that as before, in addition to the originally supplied measurements, the system has made a variety of deductions about it, based on reasoning backwards from our estimates of the height of the building and the other measurements in the shadows experiment.")
    print(barometer_height.content)

    print("If we should ask for the best estimate of the height of the barometer, we observe the same problem we noticed in the previous section, namely that the system produces a spurious dependency on the fall-time experiment, whose findings are actually redundant for answering this question.")
    print(tms_query (barometer_height.content))


    print("We can verify the irrelevance of the fall-time measurements by disbelieving them and observing that the answer remains the same, but with more accurate dependencies.")
    kick_out ('fall-time')
    print(tms_query (barometer_height.content))

    print("What is more, having been asked to make that deduction, the truth maintenance system remembers it, and produces the better answer thereafter, even if we subsequently restore our faith in the fall-time experiment,")
    bring_in ('fall-time')
    print(tms_query (barometer_height.content))

    print(barometer_height.content)


    # TODO: How to deal with contradiction in terms of propagating them through the network?
    try:
        building_height.add_content(supported(make_interval(46.,50.), ['pressure'])) # Contradiction with superintendent
    except TmsContradiction as e:
        print("Caught TMS Contradiction as expected:", e)

    #print (tms_query (building_height.content))

    print (tms_query (barometer_height.content))

    kick_out ('superintendent')
    print (tms_query (building_height.content))

    print(tms_query (barometer_height.content))

    bring_in ('superintendent')
    kick_out ('pressure')
    #TODO: Problem: Superintindendent set value created an interval instead of keeping it as a single value
    print (tms_query (building_height.content))
    print (tms_query (barometer_height.content))
    

if __name__ == "__main__":

    testing_fall_duration()
   