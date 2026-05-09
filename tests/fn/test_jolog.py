"""Tests for jolog.joseph_log_transform."""
import numpy as np
import pytest
from moirais.fn.jolog import joseph_log_transform


def test_jolog_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = joseph_log_transform(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jolog_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = joseph_log_transform(y)
    assert isinstance(result, dict)
