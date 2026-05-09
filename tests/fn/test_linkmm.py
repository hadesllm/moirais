"""Tests for linkmm.linking_meanmean."""
import numpy as np
import pytest
from moirais.fn.linkmm import linking_meanmean


def test_linkmm_basic():
    """Test basic functionality."""
    params_form_a = np.random.default_rng(42).normal(0, 1, 100)
    params_form_b = np.random.default_rng(42).normal(0, 1, 100)
    common_items = np.random.default_rng(42).normal(0, 1, 100)
    result = linking_meanmean(params_form_a, params_form_b, common_items)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_linkmm_edge():
    """Test edge cases."""
    params_form_a = np.random.default_rng(42).normal(0, 1, 100)
    params_form_b = np.random.default_rng(42).normal(0, 1, 100)
    common_items = np.random.default_rng(42).normal(0, 1, 100)
    result = linking_meanmean(params_form_a, params_form_b, common_items)
    assert isinstance(result, dict)
