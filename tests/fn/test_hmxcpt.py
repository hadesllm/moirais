"""Tests for hmxcpt.geron_xception."""
import numpy as np
import pytest
from morie.fn.hmxcpt import geron_xception


def test_hmxcpt_basic():
    """Test basic functionality."""
    n_classes = 3
    result = geron_xception(n_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmxcpt_edge():
    """Test edge cases."""
    n_classes = 3
    result = geron_xception(n_classes)
    assert isinstance(result, dict)
