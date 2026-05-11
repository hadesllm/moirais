"""Tests for fwlfwd.fully_corrective_fw."""
import numpy as np
import pytest
from morie.fn.fwlfwd import fully_corrective_fw


def test_fwlfwd_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    domain = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = fully_corrective_fw(f, grad_f, domain, x0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fwlfwd_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    grad_f = np.random.default_rng(42).normal(0, 1, 100)
    domain = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    result = fully_corrective_fw(f, grad_f, domain, x0)
    assert isinstance(result, dict)
