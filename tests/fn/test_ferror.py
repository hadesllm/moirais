"""Tests for ferror.fader_renewable."""
import numpy as np
import pytest
from morie.fn.ferror import fader_renewable


def test_ferror_basic():
    """Test basic functionality."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    Rt = np.random.default_rng(42).normal(0, 1, 100)
    gen_int = np.random.default_rng(42).normal(0, 1, 100)
    result = fader_renewable(incidence, Rt, gen_int)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ferror_edge():
    """Test edge cases."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    Rt = np.random.default_rng(42).normal(0, 1, 100)
    gen_int = np.random.default_rng(42).normal(0, 1, 100)
    result = fader_renewable(incidence, Rt, gen_int)
    assert isinstance(result, dict)
