"""Tests for jntfr.joint_frailty."""
import numpy as np
import pytest
from moirais.fn.jntfr import joint_frailty


def test_jntfr_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    terminal = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_frailty(time, event, terminal, cluster)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jntfr_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    terminal = np.random.default_rng(42).normal(0, 1, 100)
    cluster = np.random.default_rng(42).normal(0, 1, 100)
    result = joint_frailty(time, event, terminal, cluster)
    assert isinstance(result, dict)
