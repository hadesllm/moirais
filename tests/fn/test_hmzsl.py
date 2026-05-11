"""Tests for hmzsl.geron_zero_shot."""
import numpy as np
import pytest
from morie.fn.hmzsl import geron_zero_shot


def test_hmzsl_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_zero_shot(model, prompt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmzsl_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_zero_shot(model, prompt)
    assert isinstance(result, dict)
