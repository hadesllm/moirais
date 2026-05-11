"""Tests for morie.fn.irtml — MLE theta estimation."""

import numpy as np
import pytest
from morie.fn.irtml import irt_mle_theta


class TestIrtMleTheta:
    def test_returns_dict(self):
        responses = np.array([1, 1, 1, 0, 0])
        params = {f"i{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        result = irt_mle_theta(responses, params)
        assert isinstance(result, dict)
        assert "theta" in result
        assert "se" in result
        assert "converged" in result

    def test_high_scorer_positive_theta(self):
        responses = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        params = {f"i{j}": {"a": 1.0, "b": float(j * 0.5 - 2)} for j in range(10)}
        result = irt_mle_theta(responses, params)
        assert result["theta"] > 0

    def test_low_scorer_negative_theta(self):
        responses = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
        params = {f"i{j}": {"a": 1.0, "b": float(j * 0.5 - 2)} for j in range(10)}
        result = irt_mle_theta(responses, params)
        assert result["theta"] < 0

    def test_se_positive(self):
        responses = np.array([1, 0, 1, 0, 1])
        params = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(5)}
        result = irt_mle_theta(responses, params)
        assert result["se"] > 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            irt_mle_theta(np.array([1, 0]), {"i1": {"a": 1.0, "b": 0.0}})

    def test_converges(self):
        responses = np.array([1, 1, 0, 0, 1])
        params = {f"i{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        result = irt_mle_theta(responses, params)
        assert result["converged"]
