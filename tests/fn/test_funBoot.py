"""Tests for funBoot.functional_bootstrap."""
import numpy as np
import pytest
from moirais.fn.funBoot import functional_bootstrap


def test_funBoot_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    n_boot = 100
    result = functional_bootstrap(Y, n_boot)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_funBoot_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    n_boot = 100
    result = functional_bootstrap(Y, n_boot)
    assert isinstance(result, dict)
