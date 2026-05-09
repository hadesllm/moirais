"""Tests for pplxm.perplexity_metric."""
import numpy as np
import pytest
from moirais.fn.pplxm import perplexity_metric


def test_pplxm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = perplexity_metric(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_pplxm_edge():
    """Test edge cases."""
    result = perplexity_metric(np.array([42.0]))
    assert result['n'] == 1
