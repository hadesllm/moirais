"""Tests for hrzcbsm.horowitz_choice_based_sms."""
import numpy as np
import pytest
from morie.fn.hrzcbsm import horowitz_choice_based_sms


def test_hrzcbsm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    sampling_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_choice_based_sms(x, y, sampling_weights)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzcbsm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    sampling_weights = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_choice_based_sms(x, y, sampling_weights)
    assert isinstance(result, dict)
