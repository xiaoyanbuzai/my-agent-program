"""鉴权与 URL 抓取安全测试。"""

from __future__ import annotations

import socket
import gzip

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


@pytest.mark.parametrize(
    "url",
    [
        "https://arxiv.org/pdf/2502.21321",
        "https://arxiv.org/pdf/2502.21321v1",
        "https://arxiv.org/abs/2502.21321",
        "https://example.com/paper.pdf?download=1",
    ],
)
def test_fetch_page_text_routes_pdf_urls_to_binary_download(url, monkeypatch):
    monkeypatch.setattr(search_online, "validate_public_url", lambda value: value)

    def unexpected_open(*args, **kwargs):
        raise AssertionError("PDF URLs must not be sent to the HTML reader")

    monkeypatch.setattr(search_online, "_open", unexpected_open)

    assert search_online.fetch_page_text(url) == "__IS_PDF__"


def test_download_binary_converts_arxiv_abs_url_to_pdf_endpoint(monkeypatch):
    monkeypatch.setattr(search_online, "validate_public_url", lambda value: value)
    requested_urls = []

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return b"%PDF-1.4"

    def fake_open(request, _timeout):
        requested_urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(search_online, "_open", fake_open)

    assert search_online.download_binary("https://arxiv.org/abs/2502.21321") == b"%PDF-1.4"
    assert requested_urls == ["https://arxiv.org/pdf/2502.21321"]


def test_fetch_page_text_falls_back_for_doi_when_jina_times_out(monkeypatch):
    monkeypatch.setattr(search_online, "validate_public_url", lambda value: value)
    requested_urls = []

    class FakeResponse:
        headers = {}

        def __init__(self, payload: bytes):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return self.payload[:_limit]

    def fake_open(request, _timeout):
        requested_urls.append(request.full_url)
        if request.full_url.startswith("https://r.jina.ai/"):
            raise search_online.FetchError("network request failed or timeout")
        return FakeResponse(
            b"<html><head><title>AAAI 2026 Paper</title>"
            b"<meta name=\"description\" content=\"A useful abstract\">"
            b"</head><body><p>Body text is here.</p></body></html>"
        )

    monkeypatch.setattr(search_online, "_open", fake_open)

    text = search_online.fetch_page_text("https://doi.org/10.1609/aaai.v40i25.39205")
    assert "AAAI 2026 Paper" in text
    assert "Body text is here" in text
    assert requested_urls[0].startswith("https://r.jina.ai/")
    assert any(url.startswith("https://doi.org/") for url in requested_urls[1:])


def test_gzip_response_decompression_respects_size_limit():
    compressed = gzip.compress(b"x" * 1024)

    with pytest.raises(FetchError, match="解压后的响应体超过"):
        search_online._decode_payload(compressed, max_bytes=100)


def test_gzip_response_decompression_succeeds_within_size_limit():
    content = b"paper abstract"
    compressed = gzip.compress(content)

    assert search_online._decode_payload(compressed, max_bytes=100) == content
