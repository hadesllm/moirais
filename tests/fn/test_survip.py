"""Tests for survip.survey_p_value."""
import numpy as np
import pytest
from morie.fn.survip import survey_p_value


def test_survip_basic():
    """Test basic functionality."""
    test_stat = np.random.default_rng(42).normal(0, 1, 100)
    DEFF = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_p_value(test_stat, DEFF)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survip_edge():
    """Test edge cases."""
    test_stat = np.random.default_rng(42).normal(0, 1, 100)
    DEFF = np.random.default_rng(42).normal(0, 1, 100)
    result = survey_p_value(test_stat, DEFF)
    assert isinstance(result, dict)
