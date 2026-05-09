"""Tests for otplan.ot_plan_to_map."""
import numpy as np
import pytest
from moirais.fn.otplan import ot_plan_to_map


def test_otplan_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_plan_to_map(T, Y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otplan_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_plan_to_map(T, Y)
    assert isinstance(result, dict)
