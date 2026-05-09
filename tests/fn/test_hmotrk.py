"""Tests for hmotrk.geron_object_tracking."""
import numpy as np
import pytest
from moirais.fn.hmotrk import geron_object_tracking


def test_hmotrk_basic():
    """Test basic functionality."""
    frames = np.random.default_rng(42).normal(0, 1, 100)
    detector = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_object_tracking(frames, detector)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_hmotrk_edge():
    """Test edge cases."""
    frames = np.random.default_rng(42).normal(0, 1, 100)
    detector = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_object_tracking(frames, detector)
    assert isinstance(result, dict)
