from propagator import (
    Cell,
    compound_propagator,
    switch,
    conditional,
    constant,
    divider,
    adder,
    subtractor,
    multiplier,
    absolute_value,
    less_than,
    inverter,
)


def heron_step(x,g,h):
    def heron_compute():  # No parameters! Closes over x, g, h
        x_div_g = Cell()
        g_sum_x_div_g = Cell()
        two = Cell()

        divider(x, g, x_div_g)               # x / g
        adder(g, x_div_g, g_sum_x_div_g)     # g + x/g
        constant(2, two)                      # constant 2
        divider(g_sum_x_div_g, two, h)        # (g + x/g) / 2
    compound_propagator([x, g], heron_compute)


def good_enough(g, x, done):
    def good_enough_compute():
        g_squared = Cell()
        eps = Cell()
        x_minus_g_squared = Cell()
        abs_x_minus_g_squared = Cell()

        constant(0.00000001, eps)                                    # small constant
        multiplier(g, g, g_squared)                                  # g^2
        subtractor(x, g_squared, x_minus_g_squared)                  # x - g^2
        absolute_value(x_minus_g_squared, abs_x_minus_g_squared)     # |x - g^2|
        less_than(abs_x_minus_g_squared, eps, done)                  # done = (|x - g^2| < eps)

    compound_propagator([g, x], good_enough_compute)



def sqrt_iter(x, g, answer):
    def sqrt_iter_compute():
        done = Cell()
        not_done = Cell()
        x_if_not_done = Cell()
        g_if_not_done = Cell()
        new_g = Cell()

        good_enough(g, x, done)                  # Check if good enough
        switch(done, g, answer)                   # If done, output g as answer
        inverter(done, not_done)                 # not_done = not done
        switch(not_done, x, x_if_not_done)     # If not done, pass x
        switch(not_done, g, g_if_not_done)     # If not done, pass g
        heron_step(x_if_not_done, g_if_not_done, new_g)  #
        sqrt_iter(x_if_not_done, new_g, answer)  # Recur with new guess

        next_guess = Cell()
        heron_step(x, g, next_guess)

    compound_propagator([x, g], sqrt_iter_compute)

def sqrt_network(x, answer):
    def sqrt_network_compute():
        one = Cell()
        constant(1, one)
        sqrt_iter(x, one, answer) 

    compound_propagator([x],sqrt_network_compute)

def testing_sqrt_iter():
    x = Cell()
    answer = Cell()

    sqrt_network(x, answer)
    x.add_content(2)
    print(answer.content)  # Improved guess for sqrt(2)

if __name__ == "__main__":
    testing_sqrt_iter()