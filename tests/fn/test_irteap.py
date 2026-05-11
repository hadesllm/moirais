"""Tests for morie.fn.irteap — EAP theta estimation."""

import numpy as np
import pytest
from morie.fn.irteap import irt_eap_theta


class TestIrtEapTheta:
    def test_returns_dict(self):
        responses = np.array([1, 1, 0, 0, 1])
        params = {f"i{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        result = irt_eap_theta(responses, params)
        assert isinstance(result, dict)
        assert "theta" in result
        assert "se" in result

    def test_high_scorer_positive(self):
        responses = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        params = {f"i{j}": {"a": 1.0, "b": float(j * 0.5 - 2)} for j in range(10)}
        result = irt_eap_theta(responses, params)
        assert result["theta"] > 0

    def test_se_positive(self):
        responses = np.array([1, 0, 1, 0])
        params = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        result = irt_eap_theta(responses, params)
        assert result["se"] > 0

    def test_prior_shifts_estimate(self):
        responses = np.array([1, 0, 1, 0, 1])
        params = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(5)}
        r1 = irt_eap_theta(responses, params, prior_mean=0.0)
        r2 = irt_eap_theta(responses, params, prior_mean=2.0)
        # Positive prior should shift estimate upward
        assert r2["theta"] > r1["theta"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            irt_eap_theta(np.array([1, 0]), {"i1": {"a": 1.0, "b": 0.0}})
