"""Tests for moirais.fn.catw -- Catenary curve fitting."""

import numpy as np
import pandas as pd
from moirais.fn.catw import catenary_fit, catw
from moirais.fn._containers import DescriptiveResult


class TestCatw:
    def test_alias(self):
        assert catw is catenary_fit

    def test_fit_catenary(self):
        x = np.linspace(-5, 5, 50)
        a_true = 2.0
        y = a_true * np.cosh(x / a_true)
        df = pd.DataFrame({"x": x, "y": y})
        result = catenary_fit(df, x="x", y="y")
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - a_true) < 0.5
        assert result.extra["r_squared"] > 0.99
