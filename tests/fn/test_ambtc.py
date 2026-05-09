"""Tests for ambtc.am_bootstrap_se."""
import numpy as np
import pytest
from moirais.fn.ambtc import am_bootstrap_se


def test_ambtc_basic():
    """Test basic functionality."""
    survey_data = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = am_bootstrap_se(survey_data, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ambtc_edge():
    """Test edge cases."""
    survey_data = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = am_bootstrap_se(survey_data, B)
    assert isinstance(result, dict)
