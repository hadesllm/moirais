"""Test xmss_sign."""
from morie.fn._containers import CryptoResult
from morie.fn.xmss import xmss, xmss_sign


class TestXmssSign:
    def test_basic(self):
        result = xmss_sign(b"msg", tree_height=2)
        assert isinstance(result, CryptoResult)
        assert result.success is True
        assert result.algorithm == "XMSS"
        assert result.operation == "sign"

    def test_extra_fields(self):
        result = xmss_sign(b"msg", tree_height=2)
        assert "signature" in result.extra
        assert "pk" in result.extra
        assert result.extra["tree_height"] == 2

    def test_alias(self):
        assert xmss is xmss_sign
