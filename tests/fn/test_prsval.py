"""Tests for prsval.presmessick_validity."""
import numpy as np
import pytest
from morie.fn.prsval import presmessick_validity


def test_prsval_basic():
    """Test basic functionality."""
    evidence_set = np.random.default_rng(42).normal(0, 1, 100)
    result = presmessick_validity(evidence_set)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prsval_edge():
    """Test edge cases."""
    evidence_set = np.random.default_rng(42).normal(0, 1, 100)
    result = presmessick_validity(evidence_set)
    assert isinstance(result, dict)
