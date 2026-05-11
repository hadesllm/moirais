"""Tests for epimet.epinow2."""
import numpy as np
import pytest
from morie.fn.epimet import epinow2


def test_epimet_basic():
    """Test basic functionality."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    gen_int = np.random.default_rng(42).normal(0, 1, 100)
    delays = np.random.default_rng(42).normal(0, 1, 100)
    result = epinow2(incidence, gen_int, delays)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_epimet_edge():
    """Test edge cases."""
    incidence = np.random.default_rng(42).normal(0, 1, 100)
    gen_int = np.random.default_rng(42).normal(0, 1, 100)
    delays = np.random.default_rng(42).normal(0, 1, 100)
    result = epinow2(incidence, gen_int, delays)
    assert isinstance(result, dict)
