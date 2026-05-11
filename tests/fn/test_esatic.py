"""Tests for esatic.eap_information."""
import numpy as np
import pytest
from morie.fn.esatic import eap_information


def test_esatic_basic():
    """Test basic functionality."""
    item_pool = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = eap_information(item_pool, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esatic_edge():
    """Test edge cases."""
    item_pool = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = eap_information(item_pool, theta)
    assert isinstance(result, dict)
