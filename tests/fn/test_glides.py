"""Tests for glides.glide_score_proxy."""
import numpy as np
import pytest
from morie.fn.glides import glide_score_proxy


def test_glides_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand_pose = np.random.default_rng(42).normal(0, 1, 100)
    result = glide_score_proxy(receptor, ligand_pose)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_glides_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand_pose = np.random.default_rng(42).normal(0, 1, 100)
    result = glide_score_proxy(receptor, ligand_pose)
    assert isinstance(result, dict)
