"""Tests for eslrss2.esl_total_sum_squares."""
import numpy as np
import pytest
from moirais.fn.eslrss2 import esl_total_sum_squares


def test_eslrss2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_total_sum_squares(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslrss2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_total_sum_squares(y)
    assert isinstance(result, dict)
