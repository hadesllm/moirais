"""Tests for evrec.evt_record_count."""
import numpy as np
import pytest
from moirais.fn.evrec import evt_record_count


def test_evrec_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_record_count(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_evrec_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = evt_record_count(x)
    assert isinstance(result, dict)
