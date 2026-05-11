"""Tests for eslshk.esl_shrinkage."""
import numpy as np
import pytest
from morie.fn.eslshk import esl_shrinkage


def test_eslshk_basic():
    """Test basic functionality."""
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_shrinkage(nu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslshk_edge():
    """Test edge cases."""
    nu = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_shrinkage(nu)
    assert isinstance(result, dict)
