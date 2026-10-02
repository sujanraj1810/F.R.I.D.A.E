# tools/web_tools.py

from __future__ import annotations

import io
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    FETCH_TIMEOUT,
    MAX_FETCH_BYTES,
    MAX_FETCH_REDIRECTS,
)
from observability import log_event


USER_AGENT = "FRIDAE/2.0"


def _is_public_ip(value: str) -> bool:
    """Return True only for globally routable IP addresses."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False

    return address.is_global


def _resolve_public_ips(hostname: str) -> list[str]:
    """Resolve a hostname and require every resolved address to be public."""
    try:
        results = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: {hostname}") from exc

    addresses = sorted(
        {
            result[4][0]
            for result in results
            if result[4]
        }
    )

    if not addresses:
        raise ValueError(f"Hostname resolved to no addresses: {hostname}")

    if not all(_is_public_ip(address) for address in addresses):
        raise ValueError("URL resolves to a non-public IP address")

    return addresses


def _validate_public_url(url: str) -> str:
    """Validate a URL before making a server-side HTTP request."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")

    if not parsed.hostname:
        raise ValueError("URL must contain a hostname")

    hostname = parsed.hostname.strip().lower()

    if hostname in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed")

    try:
        _resolve_public_ips(hostname)
    except ValueError:
        raise

    return url


def _response_size(response: requests.Response) -> int:
    """Read Content-Length when available, otherwise return zero."""
    value = response.headers.get("Content-Length", "").strip()

    if not value:
        return 0

    try:
        return max(0, int(value))
    except ValueError:
        return 0


def _read_response_bytes(response: requests.Response) -> bytes:
    """Read a streamed response while enforcing the byte limit."""
    declared_size = _response_size(response)

    if declared_size > MAX_FETCH_BYTES:
        raise ValueError("Response exceeds the maximum fetch size")

    chunks: list[bytes] = []
    total = 0

    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue

        total += len(chunk)

        if total > MAX_FETCH_BYTES:
            raise ValueError("Response exceeds the maximum fetch size")

        chunks.append(chunk)

    return b"".join(chunks)


def _guarded_get(url: str) -> tuple[str, requests.Response]:
    """GET a public URL while validating every redirect hop."""
    current_url = _validate_public_url(url)

    for hop in range(MAX_FETCH_REDIRECTS + 1):
        response = requests.get(
            current_url,
            headers={"User-Agent": USER_AGENT},
            timeout=FETCH_TIMEOUT,
            stream=True,
            allow_redirects=False,
        )

        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")

            if not location:
                response.close()
                raise ValueError("Redirect response has no Location header")

            next_url = urljoin(current_url, location)
            response.close()

            if hop >= MAX_FETCH_REDIRECTS:
                raise ValueError("Too many redirects")

            current_url = _validate_public_url(next_url)
            continue

        return current_url, response

    raise ValueError("Too many redirects")


def fetch_url(url: str) -> str:
    """Fetch a public URL and return its decoded text content."""
    log_event("web_fetch", url=url)

    response: requests.Response | None = None

    try:
        final_url, response = _guarded_get(url)
        body = _read_response_bytes(response)

        encoding = response.encoding or "utf-8"
        text = body.decode(encoding, errors="replace")

        log_event(
            "web_fetch_complete",
            url=url,
            final_url=final_url,
            bytes=len(body),
        )

        return text
    except Exception as exc:
        log_event(
            "web_fetch_error",
            url=url,
            error=type(exc).__name__,
        )
        raise
    finally:
        if response is not None:
            response.close()


def fetch_soup(url: str) -> BeautifulSoup:
    """Fetch a public HTML page and return a BeautifulSoup document."""
    text = fetch_url(url)
    return BeautifulSoup(text, "html.parser")


def fetch_csv(url: str, **kwargs: Any) -> pd.DataFrame:
    """Fetch a public CSV resource into a DataFrame."""
    text = fetch_url(url)
    return pd.read_csv(io.StringIO(text), **kwargs)


def fetch_excel(url: str, **kwargs: Any) -> pd.DataFrame:
    """Fetch a public Excel resource into a DataFrame."""
    response: requests.Response | None = None

    try:
        final_url, response = _guarded_get(url)
        body = _read_response_bytes(response)

        log_event(
            "web_fetch_complete",
            url=url,
            final_url=final_url,
            bytes=len(body),
        )

        return pd.read_excel(io.BytesIO(body), **kwargs)
    except Exception as exc:
        log_event(
            "web_fetch_error",
            url=url,
            error=type(exc).__name__,
        )
        raise
    finally:
        if response is not None:
            response.close()


def fetch_table(url: str, **kwargs: Any) -> list[pd.DataFrame]:
    """Fetch HTML tables from a public page."""
    text = fetch_url(url)

    # pandas HTML parsing requires an HTML parser dependency. The caller
    # receives the parsed tables rather than raw HTML.
    return pd.read_html(io.StringIO(text), **kwargs)


def fetch_json(url: str, **kwargs: Any) -> Any:
    """Fetch JSON text from a public URL and parse it with pandas/json tooling."""
    text = fetch_url(url)

    import json

    return json.loads(text, **kwargs)
