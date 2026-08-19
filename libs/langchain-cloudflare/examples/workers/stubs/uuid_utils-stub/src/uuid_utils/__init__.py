"""Stub uuid_utils module for Pyodide compatibility.

uuid_utils is a Rust extension with no Pyodide wheel. `langchain_core` only
uses `uuid_utils.compat.uuid7` (see langchain_core/utils/uuid.py), so this
stub implements just that surface -- a pure-Python RFC 9562 UUIDv7 generator
returning a standard-library `uuid.UUID`.
"""

from . import compat

__all__ = ["compat"]
