"""Tests for morie.fn.sugen -- gender-specific substance use."""

import pytest
import pandas as pd
from morie.fn.sugen import substance_by_gender


class TestSubstanceByGender:
    def test_basic(self):
        df = pd.DataFrame({
            "substance_use": [1, 0, 1, 0, 1, 0],
            "gender": ["M", "M", "F", "F", "M", "F"],
        })
        res = substance_by_gender(df)
        assert res.name == "substance_by_gender"
        assert res.extra["n_groups"] == 2

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            substance_by_gender(df)
