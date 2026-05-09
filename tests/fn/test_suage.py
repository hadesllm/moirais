"""Tests for moirais.fn.suage -- age-specific substance use."""

import pytest
import pandas as pd
from moirais.fn.suage import substance_by_age


class TestSubstanceByAge:
    def test_basic(self):
        df = pd.DataFrame({
            "substance_use": [1, 0, 1, 0, 1, 1],
            "age_group": ["18-24", "18-24", "25-34", "25-34", "35-44", "35-44"],
        })
        res = substance_by_age(df)
        assert res.name == "substance_by_age"
        assert res.extra["n_groups"] == 3

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError):
            substance_by_age(df)
