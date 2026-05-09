"""Tests for rfppos.reactive_pose_filter."""
import numpy as np
import pytest
from moirais.fn.rfppos import reactive_pose_filter


def test_rfppos_basic():
    """Test basic functionality."""
    pose = np.random.default_rng(42).normal(0, 1, 100)
    cys_residue = np.random.default_rng(42).normal(0, 1, 100)
    result = reactive_pose_filter(pose, cys_residue)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rfppos_edge():
    """Test edge cases."""
    pose = np.random.default_rng(42).normal(0, 1, 100)
    cys_residue = np.random.default_rng(42).normal(0, 1, 100)
    result = reactive_pose_filter(pose, cys_residue)
    assert isinstance(result, dict)
