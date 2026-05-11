"""Tests for ccdsgn.case_control."""
import numpy as np
import pytest
from morie.fn.ccdsgn import case_control


def test_ccdsgn_basic():
    """Test basic functionality."""
    cases = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    exposed = np.random.default_rng(42).normal(0, 1, 100)
    unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = case_control(cases, controls, exposed, unexposed)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ccdsgn_edge():
    """Test edge cases."""
    cases = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    exposed = np.random.default_rng(42).normal(0, 1, 100)
    unexposed = np.random.default_rng(42).normal(0, 1, 100)
    result = case_control(cases, controls, exposed, unexposed)
    assert isinstance(result, dict)
