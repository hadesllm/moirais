"""Tests for causdiddc.causal_did_de_chaisemartin."""
import numpy as np
import pytest
from morie.fn.causdiddc import causal_did_de_chaisemartin


def test_causdiddc_basic():
    """Test basic functionality."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    D_panel = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_de_chaisemartin(Y_panel, D_panel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causdiddc_edge():
    """Test edge cases."""
    Y_panel = np.random.default_rng(42).normal(0, 1, 100)
    D_panel = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_did_de_chaisemartin(Y_panel, D_panel)
    assert isinstance(result, dict)
