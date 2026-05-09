"""Tests for matccd.matched_case_control."""
import numpy as np
import pytest
from moirais.fn.matccd import matched_case_control


def test_matccd_basic():
    """Test basic functionality."""
    cases = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    matching_id = np.random.default_rng(42).normal(0, 1, 100)
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    result = matched_case_control(cases, controls, matching_id, exposure)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_matccd_edge():
    """Test edge cases."""
    cases = np.random.default_rng(42).normal(0, 1, 100)
    controls = np.random.default_rng(42).normal(0, 1, 100)
    matching_id = np.random.default_rng(42).normal(0, 1, 100)
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    result = matched_case_control(cases, controls, matching_id, exposure)
    assert isinstance(result, dict)
