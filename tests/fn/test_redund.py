"""Tests for redund.redundancy."""
import numpy as np
import pytest
from morie.fn.redund import redundancy


def test_redund_basic():
    """Test basic functionality."""
    p = 5
    result = redundancy(p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_redund_edge():
    """Test edge cases."""
    p = 5
    result = redundancy(p)
    assert isinstance(result, dict)
