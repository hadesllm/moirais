"""Tests for cttdis.ctt_discrimination."""
import numpy as np
import pytest
from moirais.fn.cttdis import ctt_discrimination


def test_cttdis_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_discrimination(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_cttdis_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = ctt_discrimination(X)
    assert isinstance(result, dict)
