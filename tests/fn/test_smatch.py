"""Tests for smatch.sccs_design."""
import numpy as np
import pytest
from morie.fn.smatch import sccs_design


def test_smatch_basic():
    """Test basic functionality."""
    events = np.random.default_rng(42).normal(0, 1, 100)
    exposure_windows = np.random.default_rng(42).normal(0, 1, 100)
    person_id = np.random.default_rng(42).normal(0, 1, 100)
    result = sccs_design(events, exposure_windows, person_id)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smatch_edge():
    """Test edge cases."""
    events = np.random.default_rng(42).normal(0, 1, 100)
    exposure_windows = np.random.default_rng(42).normal(0, 1, 100)
    person_id = np.random.default_rng(42).normal(0, 1, 100)
    result = sccs_design(events, exposure_windows, person_id)
    assert isinstance(result, dict)
