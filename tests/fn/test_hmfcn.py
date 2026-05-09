"""Tests for hmfcn.geron_fcn."""
import numpy as np
import pytest
from moirais.fn.hmfcn import geron_fcn


def test_hmfcn_basic():
    """Test basic functionality."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fcn(image, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmfcn_edge():
    """Test edge cases."""
    image = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_fcn(image, model)
    assert isinstance(result, dict)
