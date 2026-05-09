"""Tests for moirais.fn.swmpt -- Biomass estimation."""

import numpy as np
import pandas as pd
from moirais.fn.swmpt import biomass_estimate, swmpt
from moirais.fn._containers import DescriptiveResult


class TestSwmpt:
    def test_alias(self):
        assert swmpt is biomass_estimate

    def test_basic(self):
        df = pd.DataFrame({"dbh": [10, 20, 30], "height": [5, 10, 15]})
        result = biomass_estimate(df)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_carbon(self):
        df = pd.DataFrame({"dbh": [25.0], "height": [12.0]})
        result = biomass_estimate(df)
        assert abs(result.extra["total_carbon_kg"] - result.value * 0.47) < 0.01
