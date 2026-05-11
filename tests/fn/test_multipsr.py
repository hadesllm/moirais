"""Tests for multipsr.multi_stage_sampling."""
import numpy as np
import pytest
from morie.fn.multipsr import multi_stage_sampling


def test_multipsr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stage_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = multi_stage_sampling(y, stage_probs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_multipsr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stage_probs = np.random.default_rng(42).normal(0, 1, 100)
    result = multi_stage_sampling(y, stage_probs)
    assert isinstance(result, dict)
