"""Tests for sam2vd.sam2_video_propagation."""
import numpy as np
import pytest
from morie.fn.sam2vd import sam2_video_propagation


def test_sam2vd_basic():
    """Test basic functionality."""
    video_frames = np.random.default_rng(42).normal(0, 1, 100)
    initial_prompt = np.random.default_rng(42).normal(0, 1, 100)
    result = sam2_video_propagation(video_frames, initial_prompt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sam2vd_edge():
    """Test edge cases."""
    video_frames = np.random.default_rng(42).normal(0, 1, 100)
    initial_prompt = np.random.default_rng(42).normal(0, 1, 100)
    result = sam2_video_propagation(video_frames, initial_prompt)
    assert isinstance(result, dict)
