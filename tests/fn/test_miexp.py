"""Tests for moirais.fn.miexp."""
import numpy as np
import pytest
from moirais.fn.miexp import miexp


class TestMiexp:
    def test_basic(self):
        n=20
        result = miexp(n)
        assert result is not None

    def test_returns_spatial_result(self):
        n=20
        result = miexp(n)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        n=20
        result = miexp(n)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
