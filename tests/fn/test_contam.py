"""Tests for contam.epsilon_contamination."""
import numpy as np
import pytest
from morie.fn.contam import epsilon_contamination


def test_contam_basic():
    """Test basic functionality."""
    epsilon = 1e-6
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_contamination(epsilon, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_contam_edge():
    """Test edge cases."""
    epsilon = 1e-6
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = epsilon_contamination(epsilon, H)
    assert isinstance(result, dict)
