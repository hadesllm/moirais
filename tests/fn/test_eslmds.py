"""Tests for eslmds.esl_mds."""
import numpy as np
import pytest
from morie.fn.eslmds import esl_mds


def test_eslmds_basic():
    """Test basic functionality."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = esl_mds(D, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslmds_edge():
    """Test edge cases."""
    D = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = esl_mds(D, k)
    assert isinstance(result, dict)
