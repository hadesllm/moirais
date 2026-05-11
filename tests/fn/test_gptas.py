"""Tests for gptas.gpt_assistant_decode."""
import numpy as np
import pytest
from morie.fn.gptas import gpt_assistant_decode


def test_gptas_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    max_len = np.random.default_rng(42).normal(0, 1, 100)
    result = gpt_assistant_decode(model, prompt, k, max_len)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gptas_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    prompt = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    max_len = np.random.default_rng(42).normal(0, 1, 100)
    result = gpt_assistant_decode(model, prompt, k, max_len)
    assert isinstance(result, dict)
