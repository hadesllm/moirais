"""Tests for moirais.fn.siurgn — SIU by region."""

import pytest
import pandas as pd
from moirais.fn.siurgn import siu_by_region
from moirais.fn._containers import DescriptiveResult


class TestSiuByRegion:
    def test_basic(self):
        df = pd.DataFrame({"region": ["Toronto"] * 10 + ["Eastern"] * 5})
        r = siu_by_region(df)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["counts"]["Toronto"] == 10

    def test_missing_col(self):
        with pytest.raises(ValueError):
            siu_by_region(pd.DataFrame({"x": [1]}))
