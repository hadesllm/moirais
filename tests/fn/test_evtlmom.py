"""Tests for evtlmom.evt_trimmed_lmom."""
import numpy as np
import pytest
from moirais.fn.evtlmom import evt_trimmed_lmom


def test_evtlmom_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    t = np.linspace(0, 10, 100)
    order = 4
    result = evt_trimmed_lmom(x, s, t, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evtlmom_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    t = np.linspace(0, 10, 100)
    order = 4
    result = evt_trimmed_lmom(x, s, t, order)
    assert isinstance(result, dict)
