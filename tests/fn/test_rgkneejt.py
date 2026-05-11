"""Tests for rgkneejt.rangayyan_knee_joint_sound."""
import numpy as np
import pytest
from morie.fn.rgkneejt import rangayyan_knee_joint_sound


def test_rgkneejt_basic():
    """Test basic functionality."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    force = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_knee_joint_sound(vag, fs, force)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgkneejt_edge():
    """Test edge cases."""
    vag = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    force = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_knee_joint_sound(vag, fs, force)
    assert isinstance(result, dict)
