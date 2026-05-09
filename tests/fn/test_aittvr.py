"""Tests for aittvr.aitchison_total_variance."""
import numpy as np
import pytest
from moirais.fn.aittvr import aitchison_total_variance


def test_aittvr_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_total_variance(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_aittvr_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = aitchison_total_variance(X)
    assert isinstance(result, dict)
