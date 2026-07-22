"""Pytest coverage for number/interval coercion and mixed arithmetic flows.

These tests exercise the generic operator coercion hooks and ensure
propagators can mix interval and scalar information in one network.
"""

from decimal import Decimal
from fractions import Fraction

from propagator import (
	Cell,
	adder,
	generic_add,
	generic_div,
	generic_mul,
	generic_sub,
	make_interval,
	multiplier,
	to_interval,
)


def test_to_interval_coercion():
	"""Motivation: scalar values must be liftable into interval arithmetic."""
	interval = make_interval(2, 4)
	assert to_interval(5) == make_interval(5, 5)
	assert to_interval(interval) == interval


def test_generic_ops_with_mixed_types():
	"""Motivation: mixed scalar/interval inputs should compute consistently."""
	interval = make_interval(2, 4)
	assert generic_mul(interval, interval) == make_interval(4, 16)
	assert generic_mul(interval, 3) == make_interval(6, 12)
	assert generic_mul(3, interval) == make_interval(6, 12)
	assert generic_mul(2, 5) == 10

	i1 = make_interval(10, 20)
	assert generic_div(i1, interval) == make_interval(2.5, 10.0)
	assert generic_div(i1, 2) == make_interval(5.0, 10.0)
	assert generic_div(10, interval) == make_interval(2.5, 5.0)

	assert generic_add(interval, interval) == make_interval(4, 8)
	assert generic_add(interval, 5) == make_interval(7, 9)
	assert generic_add(5, interval) == make_interval(7, 9)

	assert generic_sub(interval, interval) == make_interval(-2, 2)
	assert generic_sub(interval, 1) == make_interval(1, 3)
	assert generic_sub(10, interval) == make_interval(6, 8)


def test_mixed_type_propagators():
	"""Motivation: propagator networks should preserve mixed-type semantics."""
	a = Cell()
	b = Cell()
	c = Cell()
	multiplier(a, b, c)
	a.add_content(make_interval(2, 3))
	b.add_content(5)
	assert c.content == make_interval(10, 15)

	d = Cell()
	e = Cell()
	f = Cell()
	adder(d, e, f)
	d.add_content(make_interval(1, 2))
	e.add_content(10)
	assert f.content == make_interval(11, 12)


def test_generic_ops_with_fraction_and_decimal_scalars():
	"""Motivation: interval arithmetic should support broader ordered scalar types."""
	f_interval = make_interval(Fraction(1, 2), Fraction(3, 2))
	assert generic_add(f_interval, Fraction(1, 2)) == make_interval(Fraction(1, 1), Fraction(2, 1))
	assert generic_mul(f_interval, Fraction(2, 1)) == make_interval(Fraction(1, 1), Fraction(3, 1))

	d_interval = make_interval(Decimal("2"), Decimal("8"))
	assert generic_div(d_interval, Decimal("2")) == make_interval(Decimal("1"), Decimal("4"))
