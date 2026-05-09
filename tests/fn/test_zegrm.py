"""Tests for moirais.fn.zegrm -- Gravity migration model"""

import numpy as np
import pytest

from moirais.fn.zegrm import gravity_migration


class TestGravityMigration:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gravity_migration(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gravity_migration(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
