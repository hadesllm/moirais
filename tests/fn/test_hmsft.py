"""Tests for hmsft.geron_sft."""
import numpy as np
import pytest
from moirais.fn.hmsft import geron_sft


def test_hmsft_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    instruction_data = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sft(model, instruction_data, epochs, lr)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmsft_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    instruction_data = np.random.default_rng(42).normal(0, 1, 100)
    epochs = np.random.default_rng(42).normal(0, 1, 100)
    lr = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_sft(model, instruction_data, epochs, lr)
    assert isinstance(result, dict)
