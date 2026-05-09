"""Tests for moirais.fn.scpbic."""
import numpy as np
import pytest
from moirais.fn.scpbic import scpbic


class TestScpbic:
    def test_basic(self):
        ll=-70.0; k=5; n=40
        result = scpbic(ll, k, n)
        assert result is not None

    def test_returns_spatial_result(self):
        ll=-70.0; k=5; n=40
        result = scpbic(ll, k, n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll=-70.0; k=5; n=40
        result = scpbic(ll, k, n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
