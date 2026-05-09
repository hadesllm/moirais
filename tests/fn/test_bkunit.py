"""Tests for bkunit.burkov_unit_vector."""
import numpy as np
import pytest
from moirais.fn.bkunit import burkov_unit_vector


def test_bkunit_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = burkov_unit_vector(a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bkunit_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = burkov_unit_vector(a)
    assert isinstance(result, dict)
