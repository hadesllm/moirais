"""Tests for morie.fn.sdmspil."""
import numpy as np
import pytest
from morie.fn.sdmspil import sdmspil


class TestSdmspil:
    def test_basic(self):
        indirect=0.3; total=0.8
        result = sdmspil(indirect, total)
        assert result is not None

    def test_returns_spatial_result(self):
        indirect=0.3; total=0.8
        result = sdmspil(indirect, total)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        indirect=0.3; total=0.8
        result = sdmspil(indirect, total)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
