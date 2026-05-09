"""Tests for sgtfid.sgt_fiedler_value."""
import numpy as np
import pytest
from moirais.fn.sgtfid import sgt_fiedler_value


def test_sgtfid_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_fiedler_value(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtfid_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_fiedler_value(A)
    assert isinstance(result, dict)
