"""Tests for moirais.fn.zemir -- Migration flow model"""

import numpy as np
import pytest

from moirais.fn.zemir import migration_flow


class TestMigrationFlow:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = migration_flow(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = migration_flow(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
