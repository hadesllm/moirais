"""Tests for hml2r.geron_l2_regularization."""
import numpy as np
import pytest
from moirais.fn.hml2r import geron_l2_regularization


def test_hml2r_basic():
    """Test basic functionality."""
    theta = 0.0
    alpha = 0.05
    result = geron_l2_regularization(theta, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hml2r_edge():
    """Test edge cases."""
    theta = 0.0
    alpha = 0.05
    result = geron_l2_regularization(theta, alpha)
    assert isinstance(result, dict)
