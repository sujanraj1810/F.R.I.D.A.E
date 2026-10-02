import pytest

from tools.web_tools import _validate_public_url


def test_web_guardrail_rejects_local_addresses():
    with pytest.raises(ValueError):
        _validate_public_url("http://localhost:8000/")

    with pytest.raises(ValueError):
        _validate_public_url("http://127.0.0.1/")


def test_web_guardrail_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        _validate_public_url("file:///etc/passwd")
