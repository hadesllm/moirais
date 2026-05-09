"""Tests for moirais.fn.carvr."""
import numpy as np
import pytest
from moirais.fn.carvr import carvr


class TestCarvr:
    def test_basic(self):
        var_sp=0.6; var_un=0.4
        result = carvr(var_sp, var_un)
        assert result is not None

    def test_returns_spatial_result(self):
        var_sp=0.6; var_un=0.4
        result = carvr(var_sp, var_un)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        var_sp=0.6; var_un=0.4
        result = carvr(var_sp, var_un)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
