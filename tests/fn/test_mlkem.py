"""Tests for mlkem — ML-KEM-768 post-quantum key encapsulation."""
from morie.fn.mlkem import mlkem768_keygen, mlkem768_encaps, mlkem768_decaps
from morie.fn._containers import CryptoResult


def test_mlkem_keygen():
    result = mlkem768_keygen()
    assert isinstance(result, CryptoResult)
    assert result.algorithm == "ML-KEM-768"
    assert result.operation == "keygen"
    assert result.success is True
    assert len(result.extra["pk"]) > 0
    assert len(result.extra["sk"]) > 0


def test_mlkem_roundtrip():
    kg = mlkem768_keygen()
    pk, sk = kg.extra["pk"], kg.extra["sk"]
    enc = mlkem768_encaps(pk)
    assert enc.success is True
    ct = enc.extra["ct"]
    ss_enc = enc.extra["shared_secret"]
    dec = mlkem768_decaps(sk, ct)
    assert dec.success is True
    ss_dec = dec.extra["shared_secret"]
    assert len(ss_enc) == 32
    assert len(ss_dec) == 32
