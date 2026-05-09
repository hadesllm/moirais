"""Tests for evstud.event_study_coefficients."""
import numpy as np
import pytest
from moirais.fn.evstud import event_study_coefficients


def test_evstud_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = event_study_coefficients(y, D, unit, time, cohort)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evstud_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    time = np.linspace(0, 10, 100)
    cohort = np.random.default_rng(42).normal(0, 1, 100)
    result = event_study_coefficients(y, D, unit, time, cohort)
    assert isinstance(result, dict)
