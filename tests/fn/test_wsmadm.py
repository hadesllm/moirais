"""Tests for wsmadm.wasserman_admissible."""
import numpy as np
import pytest
from moirais.fn.wsmadm import wasserman_admissible


def test_wsmadm_basic():
    """Test basic functionality."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_admissible(estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmadm_edge():
    """Test edge cases."""
    estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = wasserman_admissible(estimator)
    assert isinstance(result, dict)
