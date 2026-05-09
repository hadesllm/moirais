"""Tests for moirais.fn.scnblrt."""
import numpy as np
import pytest
from moirais.fn.scnblrt import scnblrt


class TestScnblrt:
    def test_basic(self):
        ll_nb=-68.0; ll_pois=-72.0; df=1
        result = scnblrt(ll_nb, ll_pois, df)
        assert result is not None

    def test_returns_spatial_result(self):
        ll_nb=-68.0; ll_pois=-72.0; df=1
        result = scnblrt(ll_nb, ll_pois, df)
        assert hasattr(result, "statistic")

    def test_statistic_numeric(self):
        ll_nb=-68.0; ll_pois=-72.0; df=1
        result = scnblrt(ll_nb, ll_pois, df)
        assert result.statistic is not None
        assert not (result.statistic != result.statistic and result.statistic != float("nan"))
