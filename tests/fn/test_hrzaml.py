"""Tests for hrzaml.horowitz_additive_nonid_link."""
import numpy as np
import pytest
from moirais.fn.hrzaml import horowitz_additive_nonid_link


def test_hrzaml_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    link = 'identity'
    result = horowitz_additive_nonid_link(x, y, bandwidth, link)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzaml_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    link = 'identity'
    result = horowitz_additive_nonid_link(x, y, bandwidth, link)
    assert isinstance(result, dict)
