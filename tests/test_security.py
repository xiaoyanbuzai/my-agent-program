"""鉴权与 URL 抓取安全测试。"""

from __future__ import annotations

import socket

import pytest
from jose import JWTError

import search_online
from auth import create_token, decode_token, hash_password, verify_password
from search_online import FetchError, validate_public_url


def test_password_hash_roundtrip():
    hashed = hash_password("correct horse battery")
    assert verify_password("correct horse battery", hashed)
    assert not verify_password("wrong password", hashed)


def test_password_over_bcrypt_limit_rejected():
    with pytest.raises(ValueError):
        hash_password("a" * 73)
    assert not verify_password("a" * 73, hash_password("short-pass"))


def test_token_roundtrip_and_tamper():
    token = create_token(42)
    assert decode_token(token) == 42

    with pytest.raises(JWTError):
        decode_token(token + "x")


def _fake_dns(monkeypatch, address: str):
    def fake_getaddrinfo(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 443))]

    monkeypatch.setattr(search_online.socket, "getaddrinfo", fake_getaddrinfo)


def test_validate_public_url_rejects_local_and_private(monkeypatch):
    _fake_dns(monkeypatch, "127.0.0.1")
    with pytest.raises(FetchError):
        validate_public_url("http://localhost:8000/")
    with pytest.raises(FetchError):
        validate_public_url("http://example.com/")

    _fake_dns(monkeypatch, "192.168.1.10")
    with pytest.raises(FetchError):
        validate_public_url("https://example.com/")

    _fake_dns(monkeypatch, "93.184.216.34")
    assert validate_public_url("https://example.com/paper") == "https://example.com/paper"


def test_validate_public_url_rejects_bad_schemes_and_ports(monkeypatch):
    _fake_dns(monkeypatch, "93.184.216.34")
    for url in (
        "file:///C:/secret.txt",
        "ftp://example.com/a.pdf",
        "https://user:pass@example.com/",
        "https://example.com:8080/",
    ):
        with pytest.raises(FetchError):
            validate_public_url(url)


def test_validate_public_url_allowlist(monkeypatch):
    monkeypatch.setenv("FETCH_ALLOWED_HOSTS", "arxiv.org")
    from config import get_settings

    get_settings.cache_clear()
    try:
        assert validate_public_url("https://arxiv.org/abs/1") == "https://arxiv.org/abs/1"
        with pytest.raises(FetchError):
            validate_public_url("https://example.com/")
    finally:
        monkeypatch.delenv("FETCH_ALLOWED_HOSTS", raising=False)
        get_settings.cache_clear()
