"""Tests for esllsc.esl_lda_disc."""
import numpy as np
import pytest
from morie.fn.esllsc import esl_lda_disc


def test_esllsc_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_lda_disc(X, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_esllsc_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = esl_lda_disc(X, y)
    assert isinstance(result, dict)
