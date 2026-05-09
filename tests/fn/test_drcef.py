"""Tests for drcef.dr_callaway_event_study."""
import numpy as np
import pytest
from moirais.fn.drcef import dr_callaway_event_study


def test_drcef_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_callaway_event_study(y, D, unit, time, cohort)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drcef_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_callaway_event_study(y, D, unit, time, cohort)
    assert isinstance(result, dict)
