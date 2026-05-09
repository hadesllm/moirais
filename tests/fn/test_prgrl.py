"""Tests for prgrl.prog_rl."""
import numpy as np
import pytest
from moirais.fn.prgrl import prog_rl


def test_prgrl_basic():
    """Test basic functionality."""
    env_factory = np.random.default_rng(42).normal(0, 1, 100)
    schedule = np.random.default_rng(42).normal(0, 1, 100)
    result = prog_rl(env_factory, schedule)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_prgrl_edge():
    """Test edge cases."""
    env_factory = np.random.default_rng(42).normal(0, 1, 100)
    schedule = np.random.default_rng(42).normal(0, 1, 100)
    result = prog_rl(env_factory, schedule)
    assert isinstance(result, dict)
