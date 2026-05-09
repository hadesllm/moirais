"""Tests for hml1r.geron_l1_regularization."""
import numpy as np
import pytest
from moirais.fn.hml1r import geron_l1_regularization


def test_hml1r_basic():
    """Test basic functionality."""
    theta = 0.0
    alpha = 0.05
    result = geron_l1_regularization(theta, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hml1r_edge():
    """Test edge cases."""
    theta = 0.0
    alpha = 0.05
    result = geron_l1_regularization(theta, alpha)
    assert isinstance(result, dict)
