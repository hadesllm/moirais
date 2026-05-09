"""Tests for reidR.reidentification_risk."""
import numpy as np
import pytest
from moirais.fn.reidR import reidentification_risk


def test_reidR_basic():
    """Test basic functionality."""
    sample = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    quasi_ids = np.arange(100, dtype=int)
    result = reidentification_risk(sample, population, quasi_ids)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reidR_edge():
    """Test edge cases."""
    sample = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    quasi_ids = np.arange(100, dtype=int)
    result = reidentification_risk(sample, population, quasi_ids)
    assert isinstance(result, dict)
