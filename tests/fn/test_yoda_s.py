"""Out of chaos, comes order. — Friedrich Nietzsche"""

import numpy as np
from moirais.fn.yoda_s import sensitivity_analysis, yoda_s
from moirais.fn._containers import DescriptiveResult


class TestYodaS:
    def test_alias(self):
        assert yoda_s is sensitivity_analysis

    def test_strong_ate_high_gamma(self):
        """Strong ATE (ate=5, se=0.5): should tolerate high gamma."""
        result = sensitivity_analysis(5.0, 0.5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "Rosenbaum sensitivity"
        # With z=10, even gamma=3 should not overturn
        assert result.extra["critical_gamma"] >= 3.0
        assert result.extra["table"][0]["significant_5pct"] is True

    def test_weak_ate_low_gamma(self):
        """Weak ATE (ate=0.5, se=0.4): small gamma overturns."""
        result = sensitivity_analysis(0.5, 0.4)
        # z = 1.25, should be overturned at modest gamma
        assert result.extra["critical_gamma"] < 3.0

    def test_table_length(self):
        result = sensitivity_analysis(3.0, 1.0, n_gamma=20)
        assert len(result.extra["table"]) == 20
