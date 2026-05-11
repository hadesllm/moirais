"""Tests for sppois.schabenberger_poisson_process."""
import numpy as np
import pytest
from morie.fn.sppois import schabenberger_poisson_process


def test_sppois_basic():
    """Test basic functionality."""
    lam = 0.1
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_poisson_process(lam, region)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sppois_edge():
    """Test edge cases."""
    lam = 0.1
    region = (0.0, 1.0, 0.0, 1.0)
    result = schabenberger_poisson_process(lam, region)
    assert isinstance(result, dict)
