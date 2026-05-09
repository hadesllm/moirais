"""Tests for moirais.crypto.keystore — Encrypted key pair storage."""

import os

import pytest

from moirais.crypto.keystore import create_keystore, list_keys, load_keypair, store_keypair


class TestKeystore:

    def test_create_and_list(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("hunter2", path=ks)
        assert list_keys("hunter2", path=ks) == []

    def test_store_and_load_roundtrip(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("pass123", path=ks)
        pk = os.urandom(1184)
        sk = os.urandom(2400)
        store_keypair("alice", pk, sk, "pass123", path=ks)
        loaded_pk, loaded_sk = load_keypair("alice", "pass123", path=ks)
        assert loaded_pk == pk
        assert loaded_sk == sk

    def test_list_keys_returns_names(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("pw", path=ks)
        store_keypair("key-a", b"\x01" * 32, b"\x02" * 64, "pw", path=ks)
        store_keypair("key-b", b"\x03" * 32, b"\x04" * 64, "pw", path=ks)
        names = list_keys("pw", path=ks)
        assert sorted(names) == ["key-a", "key-b"]

    def test_wrong_password_fails(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("correct", path=ks)
        store_keypair("test", b"\x00" * 32, b"\xff" * 64, "correct", path=ks)
        with pytest.raises(ValueError, match="tag mismatch"):
            load_keypair("test", "wrong", path=ks)

    def test_missing_key_raises(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("pw", path=ks)
        with pytest.raises(KeyError, match="not found"):
            load_keypair("nonexistent", "pw", path=ks)

    def test_duplicate_keystore_raises(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("pw", path=ks)
        with pytest.raises(FileExistsError):
            create_keystore("pw", path=ks)

    def test_missing_keystore_raises(self, tmp_path):
        ks = str(tmp_path / "nope.json")
        with pytest.raises(FileNotFoundError):
            list_keys("pw", path=ks)

    def test_file_permissions(self, tmp_path):
        ks = str(tmp_path / "keystore.json")
        create_keystore("pw", path=ks)
        import stat
        mode = os.stat(ks).st_mode
        assert mode & stat.S_IRWXG == 0
        assert mode & stat.S_IRWXO == 0
