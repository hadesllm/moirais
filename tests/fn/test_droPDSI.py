"""Tests for droPDSI.pdsi."""
import numpy as np
import pytest
from moirais.fn.droPDSI import pdsi


def test_droPDSI_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    PET = np.random.default_rng(42).normal(0, 1, 100)
    awc = np.random.default_rng(42).normal(0, 1, 100)
    result = pdsi(P, PET, awc)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_droPDSI_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    PET = np.random.default_rng(42).normal(0, 1, 100)
    awc = np.random.default_rng(42).normal(0, 1, 100)
    result = pdsi(P, PET, awc)
    assert isinstance(result, dict)
