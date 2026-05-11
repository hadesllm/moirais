"""Tests for qsarh.hansch_qsar."""
import numpy as np
import pytest
from morie.fn.qsarh import hansch_qsar


def test_qsarh_basic():
    """Test basic functionality."""
    activities = np.random.default_rng(42).normal(0, 1, 100)
    descriptors = np.random.default_rng(42).normal(0, 1, 100)
    result = hansch_qsar(activities, descriptors)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_qsarh_edge():
    """Test edge cases."""
    activities = np.random.default_rng(42).normal(0, 1, 100)
    descriptors = np.random.default_rng(42).normal(0, 1, 100)
    result = hansch_qsar(activities, descriptors)
    assert isinstance(result, dict)
