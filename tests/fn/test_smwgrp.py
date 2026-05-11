"""Tests for smwgrp.small_worldness."""
import numpy as np
import pytest
from morie.fn.smwgrp import small_worldness


def test_smwgrp_basic():
    """Test basic functionality."""
    G = np.eye(10)
    random_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = small_worldness(G, random_baseline)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smwgrp_edge():
    """Test edge cases."""
    G = np.eye(10)
    random_baseline = np.random.default_rng(42).normal(0, 1, 100)
    result = small_worldness(G, random_baseline)
    assert isinstance(result, dict)
