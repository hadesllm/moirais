"""Tests for moirais.fn.carbic."""
import numpy as np
import pytest
from moirais.fn.carbic import carbic


class TestCarbic:
    def test_basic(self):
        ll=-55.0; k=3; n=40
        result = carbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-55.0; k=3; n=40
        result = carbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-55.0; k=3; n=40
        result = carbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
