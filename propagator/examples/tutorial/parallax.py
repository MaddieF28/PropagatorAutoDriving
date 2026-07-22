"""Parallax distance example translated from the Scheme propagator text.

The core relation is:
    tan(theta) * distance = AU_in_parsecs

where theta is in radians and distance is in parsecs.
"""

from __future__ import annotations

import math

from propagator import (
    Cell,
    constant,
    function_to_propagator_constructor,
    initialize_scheduler,
    kick_out,
    make_interval,
    make_tms,
    product,
    run,
    tms_query,
    bring_in,
)
from propagator.intervals import Interval
from propagator.merge import make_generic_operator
from propagator.supported_values import supported, supported_p, supported_unpacking
from propagator.tms import full_tms_unpacking, tms_p


# 1 arcsecond in radians.
ARCSECOND_RADIANS = math.pi / (180.0 * 3600.0)

# 1 AU expressed in parsecs, from the parsec definition.
AU_IN_PARSECS = math.tan(ARCSECOND_RADIANS)


def _crosses_tan_asymptote(low: float, high: float) -> bool:
    """Return True if [low, high] crosses a tan asymptote pi/2 + k*pi."""
    k_min = math.ceil((low - math.pi / 2.0) / math.pi)
    k_max = math.floor((high - math.pi / 2.0) / math.pi)
    return k_min <= k_max


def _tan_interval(x: Interval) -> Interval:
    """Compute tan over an interval for monotone segments only."""
    if _crosses_tan_asymptote(x.low, x.high):
        raise ValueError(
            "tan interval crosses an asymptote; split interval support is not implemented"
        )
    return make_interval(math.tan(x.low), math.tan(x.high))


def _atan_interval(x: Interval) -> Interval:
    """Compute atan over an interval (always monotone increasing)."""
    return make_interval(math.atan(x.low), math.atan(x.high))


def _make_tan_constraint() -> tuple:
    """Build bidirectional tan/atan propagators with interval/TMS support."""
    generic_tan = make_generic_operator(1, "tan", lambda x: math.tan(x))
    generic_atan = make_generic_operator(1, "atan", lambda x: math.atan(x))

    # Interval operations
    generic_tan.assign_operation(_tan_interval, lambda x: isinstance(x, Interval))
    generic_atan.assign_operation(_atan_interval, lambda x: isinstance(x, Interval))

    # Supported and TMS lifting
    generic_tan.assign_operation(supported_unpacking(generic_tan), supported_p)
    generic_tan.assign_operation(full_tms_unpacking(generic_tan), tms_p)
    generic_atan.assign_operation(supported_unpacking(generic_atan), supported_p)
    generic_atan.assign_operation(full_tms_unpacking(generic_atan), tms_p)

    tan_forward = function_to_propagator_constructor(generic_tan)
    atan_backward = function_to_propagator_constructor(generic_atan)
    return tan_forward, atan_backward


TAN_FORWARD, ATAN_BACKWARD = _make_tan_constraint()


def tan_constraint(theta: Cell, t: Cell) -> None:
    """Impose t = tan(theta), bidirectionally via tan and atan."""
    TAN_FORWARD(theta, t)
    ATAN_BACKWARD(t, theta)


def parallax_distance_constraint(parallax: Cell, distance: Cell) -> None:
    """Impose tan(parallax) * distance = AU_IN_PARSECS."""
    t = Cell("t")
    au = Cell("AU_in_parsecs")
    constant(AU_IN_PARSECS, au)
    tan_constraint(parallax, t)
    product(t, distance, au)


def mas_to_radians(milliarcseconds: float) -> float:
    """Convert milliarcseconds to radians."""
    return milliarcseconds * ARCSECOND_RADIANS / 1000.0


def plus_minus_interval(value: float, delta: float) -> Interval:
    """Create [value-delta, value+delta]."""
    return make_interval(value - delta, value + delta)


def tell(cell: Cell, value: Interval, premise: object) -> None:
    """Add a TMS-supported interval measurement to a cell."""
    cell.add_content(make_tms(supported(value, [premise])))


def inquire(cell: Cell, label: str) -> None:
    """Print best current value and full TMS content."""
    print(f"{label}: best={tms_query(cell.content)}")
    print(f"{label}: full={cell.content}")


def parallax() -> dict[str, Cell]:
    """Build the Vega parallax network and return key cells."""
    vega_parallax = Cell("Vega-parallax")
    vega_distance = Cell("Vega-parallax-distance")
    parallax_distance_constraint(vega_parallax, vega_distance)
    return {
        "vega_parallax": vega_parallax,
        "vega_distance": vega_distance,
    }


def demo() -> None:
    """Run the same storyline as the Scheme text with TMS premises."""
    network = parallax()
    vega_parallax = network["vega_parallax"]
    vega_distance = network["vega_distance"]

    struve_1837 = "FGWvonStruve1837"
    russell_1982 = "JRussell-etal1982"
    gatewood_1995 = "Gatewood-deJonge1995"
    vanleeuwen_2007 = "FvanLeeuwen2007Nov"

    print("\\n== Struve 1837 ==")
    tell(
        vega_parallax,
        plus_minus_interval(mas_to_radians(125.0), mas_to_radians(50.0)),
        struve_1837,
    )
    run()
    inquire(vega_distance, "distance")
    

    print("\\n== Russell et al. 1982 ==")
    tell(
        vega_parallax,
        plus_minus_interval(mas_to_radians(124.3), mas_to_radians(4.9)),
        russell_1982,
    )
    run()
    inquire(vega_distance, "distance")

    print("\\n== Gatewood and de Jonge 1995 (in tension) ==")
    tell(
        vega_parallax,
        plus_minus_interval(mas_to_radians(131.0), mas_to_radians(0.77)),
        gatewood_1995,
    )
    run()
    inquire(vega_parallax, "parallax")
    inquire(vega_distance, "distance")

    print("\\n== Retract Gatewood 1995 ==")
    kick_out(gatewood_1995)
    run()
    inquire(vega_distance, "distance")

    print("\\n== Add Van Leeuwen 2007 (Hipparcos) ==")
    tell(
        vega_parallax,
        plus_minus_interval(mas_to_radians(130.23), mas_to_radians(0.36)),
        vanleeuwen_2007,
    )
    run()
    inquire(vega_parallax, "parallax")

    print("\\n== Prefer Hipparcos over Russell ==")
    kick_out(russell_1982)
    run()
    inquire(vega_distance, "distance")

    print("\\n== Re-introduce Gatewood and intersect with Hipparcos ==")
    bring_in(gatewood_1995)
    run()
    inquire(vega_distance, "distance")


# TODO(EIRNf): Scheme's assert!/retract! interface tracks premises by symbol with a
# dedicated API. Here we approximate with kick_out/bring_in and premise objects.
# TODO(EIRNf): Scheme inquire prints explanation provenance in a richer "because"
# trace. This example only prints tms_query() and full TMS content.
# TODO(EIRNf): Interval tan is currently limited to monotone segments and raises
# if an interval crosses tan asymptotes. A full implementation should split ranges.
# TODO(EIRNf): Unit-aware values (AU, parsec, radians) are not type-checked. A
# dimensional analysis layer would prevent accidental unit mismatches.


if __name__ == "__main__":
    initialize_scheduler()
    demo()