"""Tests for alibi.alibi."""
import numpy as np
import pytest
from moirais.fn.alibi import alibi


def test_alibi_basic():
    """Test basic functionality."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    slopes = np.random.default_rng(42).normal(0, 1, 100)
    result = alibi(scores, slopes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alibi_edge():
    """Test edge cases."""
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    slopes = np.random.default_rng(42).normal(0, 1, 100)
    result = alibi(scores, slopes)
    assert isinstance(result, dict)
