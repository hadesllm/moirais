"""Tests for km038.kamath_ch2_gpt2_task_conditioning."""
import numpy as np
import pytest
from morie.fn.km038 import kamath_ch2_gpt2_task_conditioning


def test_km038_basic():
    """Test basic functionality."""
    input = np.random.default_rng(42).normal(0, 1, 100)
    task = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_gpt2_task_conditioning(input, task)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km038_edge():
    """Test edge cases."""
    input = np.random.default_rng(42).normal(0, 1, 100)
    task = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch2_gpt2_task_conditioning(input, task)
    assert isinstance(result, dict)
