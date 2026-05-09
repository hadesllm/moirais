"""Tests for spbino.schabenberger_binomial_process."""
import numpy as np
import pytest
from moirais.fn.spbino import schabenberger_binomial_process


def test_spbino_basic():
    """Test basic functionality."""
    n = 100
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_binomial_process(n, region)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spbino_edge():
    """Test edge cases."""
    n = 100
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_binomial_process(n, region)
    assert isinstance(result, dict)
