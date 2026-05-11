"""Tests for morie.fn.odm_c — OTIS demo cross-tab."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.odm_c import otis_demo_cross
from morie.fn._containers import DescriptiveResult


class TestOtisDemoCross:

    def test_returns_descriptive(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"region": rng.choice(["E", "W"], 20),
                           "age_group": rng.choice(["Y", "O"], 20),
                           "gender": rng.choice(["M", "F"], 20),
                           "person_id": range(20)})
        result = otis_demo_cross(df)
        assert isinstance(result, DescriptiveResult)
        assert isinstance(result.value, pd.DataFrame)

    def test_counts_positive(self):
        df = pd.DataFrame({"region": ["E", "W", "E", "W"], "age_group": ["Y", "O", "Y", "O"],
                           "gender": ["M", "F", "M", "F"], "person_id": [1, 2, 3, 4]})
        result = otis_demo_cross(df)
        assert (result.value >= 0).all().all()
