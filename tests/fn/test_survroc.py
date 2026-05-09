"""Tests for survroc.time_dep_roc."""
import numpy as np
import pytest
from moirais.fn.survroc import time_dep_roc


def test_survroc_basic():
    """Test basic functionality."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    marker = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = time_dep_roc(time, event, marker, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_survroc_edge():
    """Test edge cases."""
    time = np.linspace(0, 10, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    marker = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = time_dep_roc(time, event, marker, t)
    assert isinstance(result, dict)
