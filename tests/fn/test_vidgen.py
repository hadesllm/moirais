"""Tests for vidgen.video_diffusion."""
import numpy as np
import pytest
from moirais.fn.vidgen import video_diffusion


def test_vidgen_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    conditions = np.random.default_rng(42).normal(0, 1, 100)
    n_frames = np.random.default_rng(42).normal(0, 1, 100)
    result = video_diffusion(t, conditions, n_frames)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vidgen_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    conditions = np.random.default_rng(42).normal(0, 1, 100)
    n_frames = np.random.default_rng(42).normal(0, 1, 100)
    result = video_diffusion(t, conditions, n_frames)
    assert isinstance(result, dict)
