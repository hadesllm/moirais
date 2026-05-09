"""Tests for eslsft.esl_softmax."""
import numpy as np
import pytest
from moirais.fn.eslsft import esl_softmax


def test_eslsft_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = esl_softmax(T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eslsft_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = esl_softmax(T)
    assert isinstance(result, dict)
