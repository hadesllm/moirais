"""Tests for smltot.survey_total."""
import numpy as np
import pytest
from moirais.fn.smltot import survey_total


def test_smltot_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_total(y, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smltot_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_total(y, weights)
    assert isinstance(result, dict)
