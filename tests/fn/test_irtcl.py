"""Tests for morie.fn.irtcl — IRT calibration."""

import numpy as np
import pandas as pd
from morie.fn.irtcl import irt_calibrate


class TestIrtCalibrate:
    def test_returns_dict(self, mapq_binary_df):
        result = irt_calibrate(mapq_binary_df, model="1PL")
        assert isinstance(result, dict)
        assert "item_params" in result
        assert "theta" in result
        assert "model" in result

    def test_item_count(self, mapq_binary_df):
        result = irt_calibrate(mapq_binary_df, model="2PL")
        assert len(result["item_params"]) == 10

    def test_theta_length(self, mapq_binary_df):
        result = irt_calibrate(mapq_binary_df)
        assert len(result["theta"]) == len(mapq_binary_df)

    def test_1pl_a_equals_one(self, mapq_binary_df):
        result = irt_calibrate(mapq_binary_df, model="1PL")
        for item, params in result["item_params"].items():
            assert abs(params["a"] - 1.0) < 1e-10

    def test_se_theta_available(self, mapq_binary_df):
        result = irt_calibrate(mapq_binary_df)
        assert result["se_theta"] is not None
        assert len(result["se_theta"]) == len(mapq_binary_df)

    def test_ndarray(self, rng):
        data = rng.binomial(1, 0.6, size=(100, 5)).astype(float)
        result = irt_calibrate(data, model="1PL")
        assert len(result["item_params"]) == 5
