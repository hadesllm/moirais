"""Tests for erdosg.erdos_renyi_gnp."""
import numpy as np
import pytest
from moirais.fn.erdosg import erdos_renyi_gnp


def test_erdosg_basic():
    """Test basic functionality."""
    n = 100
    p = 5
    result = erdos_renyi_gnp(n, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_erdosg_edge():
    """Test edge cases."""
    n = 100
    p = 5
    result = erdos_renyi_gnp(n, p)
    assert isinstance(result, dict)
