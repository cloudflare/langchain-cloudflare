"""Stub xxhash module for Pyodide compatibility.

xxhash is a C extension that doesn't have a Pyodide wheel.
This stub provides the minimal API needed by langgraph
using hashlib as a fallback.

Note: This stub allows create_agent to work in Cloudflare Python Workers,
but hashing may have slightly different characteristics than native xxhash.
"""

import hashlib
from typing import Union

# MARK: - Hash Functions


def xxh3_128_hexdigest(data: Union[str, bytes]) -> str:
    """Compute XXH3 128-bit hash and return as hex string.

    Uses MD5 as fallback since it also produces 128-bit hashes.
    The hash values will differ from native xxhash, but this is
    acceptable for ID generation purposes in langgraph.

    Args:
        data: Input data (str will be encoded to UTF-8)

    Returns:
        32-character hex string (128 bits)
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()


def xxh3_64_hexdigest(data: Union[str, bytes]) -> str:
    """Compute XXH3 64-bit hash and return as hex string.

    Uses first 16 characters of MD5 as fallback (64 bits).

    Args:
        data: Input data (str will be encoded to UTF-8)

    Returns:
        16-character hex string (64 bits)
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.md5(data).hexdigest()[:16]


# MARK: - Hash Classes


class xxh3_128:
    """XXH3 128-bit hash object stub."""

    def __init__(self, data: Union[str, bytes, None] = None):
        self._hasher = hashlib.md5()
        if data is not None:
            self.update(data)

    def update(self, data: Union[str, bytes]) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()

    def digest(self) -> bytes:
        return self._hasher.digest()

    def intdigest(self) -> int:
        return int.from_bytes(self._hasher.digest(), "big")


class xxh3_64:
    """XXH3 64-bit hash object stub."""

    def __init__(self, data: Union[str, bytes, None] = None):
        self._hasher = hashlib.md5()
        if data is not None:
            self.update(data)

    def update(self, data: Union[str, bytes]) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._hasher.update(data)

    def hexdigest(self) -> str:
        return self._hasher.hexdigest()[:16]

    def digest(self) -> bytes:
        return self._hasher.digest()[:8]

    def intdigest(self) -> int:
        return int.from_bytes(self._hasher.digest()[:8], "big")


# MARK: - Version

__version__ = "3.5.0"  # Stub version
VERSION = (3, 5, 0)
XXHASH_VERSION = "0.8.2"  # Native xxhash version we're mimicking
