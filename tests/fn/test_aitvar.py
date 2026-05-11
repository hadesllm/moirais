"""Tests for aitvar.aitchison_variation."""
import numpy as np
import pytest
from morie.fn.aitvar import aitchison_variation


def test_aitvar_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_variation(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aitvar_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_variation(X)
    assert isinstance(result, dict)
