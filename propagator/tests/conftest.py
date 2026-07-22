"""Pytest configuration for propagator tests.

Provides shared fixtures such as scheduler reset that several tests require.
"""

import pytest

from propagator import initialize_scheduler


@pytest.fixture(autouse=True)
def reset_scheduler_before_each_test():
    """Ensure a clean scheduler for every test."""
    initialize_scheduler()
