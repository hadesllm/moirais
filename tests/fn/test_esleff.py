"""Tests for esleff.esl_effective_dof."""
import numpy as np
import pytest
from morie.fn.esleff import esl_effective_dof


def test_esleff_basic():
    """Test basic functionality."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_effective_dof(S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esleff_edge():
    """Test edge cases."""
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = esl_effective_dof(S)
    assert isinstance(result, dict)
