"""Tests for morie.fn.zxdps -- Dirichlet process spatial"""

import numpy as np
import pytest

from morie.fn.zxdps import dirichlet_proc_sp


class TestDirichletProcSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dirichlet_proc_sp(data)
        assert result.value is not None

    def test_output_type(self):
        result = dirichlet_proc_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
