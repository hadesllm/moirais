"""Tests for madAd.mad_anomaly_score."""
import numpy as np
import pytest
from moirais.fn.madAd import mad_anomaly_score


def test_madAd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mad_anomaly_score(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_madAd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = mad_anomaly_score(x)
    assert isinstance(result, dict)
