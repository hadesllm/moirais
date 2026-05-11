"""Tests for mtr2sx.sex_specific_mr."""
import numpy as np
import pytest
from morie.fn.mtr2sx import sex_specific_mr


def test_mtr2sx_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    instrument = np.random.default_rng(42).normal(0, 1, 100)
    sex = np.random.default_rng(42).normal(0, 1, 100)
    result = sex_specific_mr(y, exposure, instrument, sex)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtr2sx_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    instrument = np.random.default_rng(42).normal(0, 1, 100)
    sex = np.random.default_rng(42).normal(0, 1, 100)
    result = sex_specific_mr(y, exposure, instrument, sex)
    assert isinstance(result, dict)
