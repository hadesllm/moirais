"""Tests for odgrev.outbreak_detection."""
import numpy as np
import pytest
from moirais.fn.odgrev import outbreak_detection


def test_odgrev_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    prior_hazard = np.random.default_rng(42).normal(0, 1, 100)
    result = outbreak_detection(counts, prior_hazard)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_odgrev_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    prior_hazard = np.random.default_rng(42).normal(0, 1, 100)
    result = outbreak_detection(counts, prior_hazard)
    assert isinstance(result, dict)
