"""Pytest coverage for the generic operator dispatch and simple propagators.

These are fast, end-to-end smoke tests that ensure the generic operator
registry is wired and that function-to-propagator constructors run.
"""

from propagator import (
	Cell,
	adder,
	eq,
	generic_abs,
	generic_add,
	generic_and,
	generic_div,
	generic_eq,
	generic_gte,
	generic_gt,
	generic_lt,
	generic_lte,
	generic_mul,
	generic_not,
	generic_or,
	generic_sqrt,
	generic_square,
	generic_sub,
	lt,
	squarer,
	sqrter,
)


def test_generic_arithmetic_ops():
	"""Motivation: arithmetic operators are the backbone of numeric constraints."""
	assert generic_add(5, 3) == 8
	assert generic_sub(10, 4) == 6
	assert generic_mul(6, 7) == 42
	assert generic_div(20, 4) == 5
	assert generic_abs(-5) == 5
	assert generic_square(4) == 16
	assert generic_sqrt(16) == 4


def test_generic_comparison_ops():
	"""Motivation: comparisons are used for constraint guards and searches."""
	assert generic_eq(5, 5) is True
	assert generic_eq(5, 3) is False
	assert generic_lt(3, 5) is True
	assert generic_gt(5, 3) is True
	assert generic_lte(3, 3) is True
	assert generic_gte(5, 5) is True


def test_generic_boolean_ops():
	"""Motivation: boolean combinators are needed for conditional logic."""
	assert generic_not(True) is False
	assert generic_and(True, False) is False
	assert generic_or(True, False) is True


def test_basic_propagator_constructors():
	"""Motivation: ensure unidirectional propagators compute outputs."""
	a = Cell()
	b = Cell()
	c = Cell()
	adder(a, b, c)
	a.add_content(10)
	b.add_content(20)
	assert c.content == 30

	d = Cell()
	e = Cell()
	squarer(d, e)
	d.add_content(5)
	assert e.content == 25

	f = Cell()
	g = Cell()
	sqrter(f, g)
	f.add_content(25)
	assert g.content == 5

	h = Cell()
	i = Cell()
	j = Cell()
	eq(h, i, j)
	h.add_content(5)
	i.add_content(5)
	assert j.content is True

	k = Cell()
	l = Cell()
	m = Cell()
	lt(k, l, m)
	k.add_content(3)
	l.add_content(5)
	assert m.content is True
