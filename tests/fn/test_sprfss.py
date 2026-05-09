"""Tests for sprfss.schabenberger_random_field_stationarity."""
import numpy as np
import pytest
from moirais.fn.sprfss import schabenberger_random_field_stationarity


def test_sprfss_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_random_field_stationarity(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_sprfss_edge():
    """Test edge cases."""
    result = schabenberger_random_field_stationarity(np.array([42.0]))
    assert result['n'] == 1
