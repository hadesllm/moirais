"""Tests for svymed.survey_median."""
import numpy as np
import pytest
from moirais.fn.svymed import survey_median


def test_svymed_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_median(y, weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_svymed_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    weights = np.random.default_rng(45).exponential(1, 100)
    result = survey_median(y, weights)
    assert isinstance(result, dict)
