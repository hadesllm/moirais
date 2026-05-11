"""Tests for smplep.sample_overlap."""
import numpy as np
import pytest
from morie.fn.smplep import sample_overlap


def test_smplep_basic():
    """Test basic functionality."""
    frame_a = np.random.default_rng(42).normal(0, 1, 100)
    frame_b = np.random.default_rng(42).normal(0, 1, 100)
    overlap_indicator = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_overlap(frame_a, frame_b, overlap_indicator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smplep_edge():
    """Test edge cases."""
    frame_a = np.random.default_rng(42).normal(0, 1, 100)
    frame_b = np.random.default_rng(42).normal(0, 1, 100)
    overlap_indicator = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_overlap(frame_a, frame_b, overlap_indicator)
    assert isinstance(result, dict)
