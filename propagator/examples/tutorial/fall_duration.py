from propagator import (
    Cell,
    compound_propagator,
    constant,
    make_interval,
    product,
    quadratic,
)


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

    compound_propagator([time], fall_duration_compute)# Duration of fall in seconds


def testing_fall_duration():

    fall_time = Cell()
    building_height = Cell()

    fall_duration(fall_time, building_height)

    fall_time.add_content(make_interval(2.9, 3.1))
    print(building_height.content)     # Duration of fall in seconds


    baromter_height = Cell()
    barometer_shadow = Cell()
    building_height = Cell()
    building_shadow = Cell()
    similar_triangles_height(barometer_shadow, baromter_height, building_shadow, building_height)

    building_shadow.add_content(make_interval(54.9, 55.1))
    baromter_height.add_content(make_interval(0.3, 0.32))
    barometer_shadow.add_content(make_interval(0.36, 0.37))
    print(building_height.content)

    fall_time = Cell()
    fall_duration(fall_time, building_height)
    fall_time.add_content(make_interval(2.9, 3.1))
    print(building_height.content)

    fall_time = Cell()
    building_height = Cell()
    fall_duration(fall_time, building_height)

    fall_time.add_content(make_interval(2.9, 3.1))
    print(building_height.content)

    building_height.add_content(45)
    print(fall_time.content)

if __name__ == "__main__":
    testing_fall_duration()    