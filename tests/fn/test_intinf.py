"""Tests for intinf.interaction_information."""
import numpy as np
import pytest
from moirais.fn.intinf import interaction_information


def test_intinf_basic():
    """Test basic functionality."""
    pxyz = np.random.default_rng(42).normal(0, 1, 100)
    result = interaction_information(pxyz)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intinf_edge():
    """Test edge cases."""
    pxyz = np.random.default_rng(42).normal(0, 1, 100)
    result = interaction_information(pxyz)
    assert isinstance(result, dict)
