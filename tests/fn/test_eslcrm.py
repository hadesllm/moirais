"""Tests for eslcrm.esl_cross_entropy."""
import numpy as np
import pytest
from moirais.fn.eslcrm import esl_cross_entropy


def test_eslcrm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = esl_cross_entropy(y, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslcrm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = esl_cross_entropy(y, p)
    assert isinstance(result, dict)
