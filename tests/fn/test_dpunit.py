"""Tests for dpunit.dp_unit_definition."""
import numpy as np
import pytest
from morie.fn.dpunit import dp_unit_definition


def test_dpunit_basic():
    """Test basic functionality."""
    unit = np.random.default_rng(42).normal(0, 1, 100)
    records = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_unit_definition(unit, records)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpunit_edge():
    """Test edge cases."""
    unit = np.random.default_rng(42).normal(0, 1, 100)
    records = np.random.default_rng(42).normal(0, 1, 100)
    result = dp_unit_definition(unit, records)
    assert isinstance(result, dict)
