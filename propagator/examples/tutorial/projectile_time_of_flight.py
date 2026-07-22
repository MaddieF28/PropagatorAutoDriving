from propagator import (
    Cell,
    compound_propagator,
    constant,
    difference,
    make_interval,
    product,
)


def projectile_time_of_flight(initial_velocity, flight_time):
    """Build a projectile landing-time network for same-height launch/landing.

        Constraint encoded:
            g * t = 2 * v0

        This is the non-zero time-of-flight branch derived from
            0 = v0*t - (1/2)*g*t^2
        for same-height launch and landing.
    """

    def projectile_time_of_flight_compute():
        g = Cell()
        two = Cell()
        zero = Cell()

        g_times_t = Cell()
        two_v0 = Cell()

        constant(9.81, g)
        constant(2.0, two)
        constant(0.0, zero)

        # Time-of-flight relation for same-height launch/landing.
        product(g, flight_time, g_times_t)
        product(two, initial_velocity, two_v0)
        difference(g_times_t, two_v0, zero)

    compound_propagator([initial_velocity, flight_time], projectile_time_of_flight_compute)


def projectile_time_of_flight_interval(initial_velocity, flight_time):
    """Build an interval network variant with uncertain gravitational acceleration.

    Constraint encoded:
        g * t = 2 * v0

    Uses Earth's local gravity interval as in other tutorial examples.
    """

    def projectile_time_of_flight_interval_compute():
        g = Cell()
        two = Cell()
        zero = Cell()

        g_times_t = Cell()
        two_v0 = Cell()

        constant(make_interval(9.789, 9.832), g)
        constant(2.0, two)
        constant(0.0, zero)

        product(g, flight_time, g_times_t)
        product(two, initial_velocity, two_v0)
        difference(g_times_t, two_v0, zero)

    compound_propagator([initial_velocity, flight_time], projectile_time_of_flight_interval_compute)


def testing_projectile_time_of_flight():
    # Query 1: known time, infer unknown initial velocity
    initial_velocity = Cell()
    flight_time = Cell()
    projectile_time_of_flight(initial_velocity, flight_time)

    flight_time.add_content(3.0)
    print(initial_velocity.content)  # 14.715 m/s

    # Query 2: known initial velocity, infer time of flight
    initial_velocity = Cell()
    flight_time = Cell()
    projectile_time_of_flight(initial_velocity, flight_time)

    initial_velocity.add_content(19.62)
    print(flight_time.content)  # 4.0 s


def testing_projectile_time_of_flight_interval():
    # Query 1: known time interval, infer initial velocity interval
    initial_velocity = Cell()
    flight_time = Cell()
    projectile_time_of_flight_interval(initial_velocity, flight_time)

    flight_time.add_content(make_interval(2.9, 3.1))
    print(initial_velocity.content)  # Approximately Interval(14.19, 15.24)

    # Query 2: known initial-velocity interval, infer time interval
    initial_velocity = Cell()
    flight_time = Cell()
    projectile_time_of_flight_interval(initial_velocity, flight_time)

    initial_velocity.add_content(make_interval(19.5, 19.8))
    print(flight_time.content)  # Approximately Interval(3.97, 4.05)


if __name__ == "__main__":
    testing_projectile_time_of_flight()
    testing_projectile_time_of_flight_interval()
