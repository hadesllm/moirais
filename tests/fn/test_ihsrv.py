"""Tests for moirais.fn.ihsrv -- indigenous service access."""

import pytest
from moirais.fn.ihsrv import indigenous_service_access


class TestIndigenousServiceAccess:
    def test_basic(self):
        res = indigenous_service_access(access_indigenous=0.6, access_general=0.8)
        assert res.estimate == pytest.approx(0.75)
        assert res.extra["gap"] == pytest.approx(0.2)

    def test_equal(self):
        res = indigenous_service_access(0.8, 0.8)
        assert res.estimate == pytest.approx(1.0)
