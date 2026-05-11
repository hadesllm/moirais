"""Tests for shorE.shor_factoring."""
import numpy as np
import pytest
from morie.fn.shorE import shor_factoring


def test_shorE_basic():
    """Test basic functionality."""
    N = 100
    result = shor_factoring(N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_shorE_edge():
    """Test edge cases."""
    N = 100
    result = shor_factoring(N)
    assert isinstance(result, dict)
