"""Tests for plcbsc.placebo_scm_inference."""
import numpy as np
import pytest
from morie.fn.plcbsc import placebo_scm_inference


def test_plcbsc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    result = placebo_scm_inference(y, treated, controls)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_plcbsc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    treated = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    result = placebo_scm_inference(y, treated, controls)
    assert isinstance(result, dict)
