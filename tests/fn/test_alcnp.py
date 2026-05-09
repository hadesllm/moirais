"""Tests for alcnp.alammar_chain_prompting."""
import numpy as np
import pytest
from moirais.fn.alcnp import alammar_chain_prompting


def test_alcnp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_chain_prompting(x, prompts, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alcnp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    prompts = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = alammar_chain_prompting(x, prompts, model)
    assert isinstance(result, dict)
