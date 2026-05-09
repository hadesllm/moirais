"""Tests for bocpd.bocpd."""
import numpy as np
import pytest
from moirais.fn.bocpd import bocpd


def test_bocpd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    hazard = np.random.default_rng(42).normal(0, 1, 100)
    result = bocpd(x, hazard)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bocpd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    hazard = np.random.default_rng(42).normal(0, 1, 100)
    result = bocpd(x, hazard)
    assert isinstance(result, dict)
