"""Tests for moirais.fn.irtli — IRT log-likelihood."""

import numpy as np
import pytest
from moirais.fn.irtli import irt_likelihood


class TestIrtLikelihood:
    def test_returns_float(self):
        responses = np.array([1, 0, 1, 0])
        params = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        result = irt_likelihood(responses, params, theta=0.0)
        assert isinstance(result, float)

    def test_negative(self):
        responses = np.array([1, 0, 1, 0])
        params = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        result = irt_likelihood(responses, params, theta=0.0)
        assert result < 0  # log-likelihood always negative

    def test_max_at_correct_theta(self):
        # Perfect responses at easy items -> max likelihood at high theta
        responses = np.array([1, 1, 1, 1, 1])
        params = {f"i{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        ll_high = irt_likelihood(responses, params, theta=3.0)
        ll_low = irt_likelihood(responses, params, theta=-3.0)
        assert ll_high > ll_low

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            irt_likelihood(np.array([1, 0]), {"i1": {"a": 1.0, "b": 0.0}}, theta=0.0)

    def test_handles_nan(self):
        responses = np.array([1, np.nan, 1, 0])
        params = {f"i{j}": {"a": 1.0, "b": 0.0} for j in range(4)}
        result = irt_likelihood(responses, params, theta=0.0)
        assert np.isfinite(result)
