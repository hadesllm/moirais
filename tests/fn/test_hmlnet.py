"""Tests for hmlnet.geron_lenet5."""
import numpy as np
import pytest
from moirais.fn.hmlnet import geron_lenet5


def test_hmlnet_basic():
    """Test basic functionality."""
    n_classes = 3
    result = geron_lenet5(n_classes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmlnet_edge():
    """Test edge cases."""
    n_classes = 3
    result = geron_lenet5(n_classes)
    assert isinstance(result, dict)
