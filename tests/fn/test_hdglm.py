"""Tests for hdglm (Hodges-Lehmann estimator)."""

import numpy as np
import pytest
from moirais.fn.hdglm import hdglm


class TestHdglm:
    """Hodges-Lehmann point estimator."""

    def test_hdglm_symmetric(self):
        """Symmetric data should have estimate near center."""
        x = np.array([1, 2, 3, 4, 5])
        result = hdglm(x)
        # Estimate should be near 3
        assert 2.5 < result["estimate"] < 3.5

    def test_hdglm_robust_to_outliers(self):
        """Should be robust to extreme outliers."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5, 1000])
        result_x = hdglm(x)
        result_y = hdglm(y)
        # Estimate should not change much with extreme outlier
        assert abs(result_x["estimate"] - result_y["estimate"]) < 50

    def test_hdglm_ci_ordered(self):
        """CI lower bound should be ≤ upper bound."""
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        result = hdglm(x)
        assert result["lower_ci"] <= result["upper_ci"]

    def test_hdglm_returns_dict(self):
        """Return type should be dict with required keys."""
        x = np.array([1, 2, 3, 4, 5])
        result = hdglm(x)
        required_keys = {"estimate", "walsh_count", "lower_ci", "upper_ci"}
        assert set(result.keys()) == required_keys

    def test_hdglm_small_sample_error(self):
        """Sample size < 2 should raise error."""
        with pytest.raises(ValueError):
            hdglm(np.array([1]))

    def test_hdglm_estimate_in_ci(self):
        """Estimate should fall within its own CI."""
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        result = hdglm(x)
        assert result["lower_ci"] <= result["estimate"] <= result["upper_ci"]
