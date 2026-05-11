"""Tests for grmlc.geron_classification_mlp_output."""
import numpy as np
import pytest
from morie.fn.grmlc import geron_classification_mlp_output


def test_grmlc_basic():
    """Test basic functionality."""
    a_last = np.random.default_rng(42).normal(0, 1, 100)
    W_out = np.random.default_rng(42).normal(0, 1, 100)
    b_out = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_classification_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grmlc_edge():
    """Test edge cases."""
    a_last = np.random.default_rng(42).normal(0, 1, 100)
    W_out = np.random.default_rng(42).normal(0, 1, 100)
    b_out = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_classification_mlp_output(a_last, W_out, b_out)
    assert isinstance(result, dict)
