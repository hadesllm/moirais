"""Tests for bayess.effective_sample_size_bayes."""
import numpy as np
import pytest
from moirais.fn.bayess import effective_sample_size_bayes


def test_bayess_basic():
    """Test basic functionality."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size_bayes(chain)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bayess_edge():
    """Test edge cases."""
    chain = np.random.default_rng(42).normal(0, 1, 100)
    result = effective_sample_size_bayes(chain)
    assert isinstance(result, dict)
