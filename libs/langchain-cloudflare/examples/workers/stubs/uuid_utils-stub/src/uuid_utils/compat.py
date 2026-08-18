"""Stub of ``uuid_utils.compat`` for Pyodide compatibility.

In the native package this module mirrors the top-level API but returns stdlib
``uuid.UUID`` objects instead of the Rust ``UUID`` type. This stub already
returns stdlib UUIDs everywhere, so it simply re-exports them.
"""

from uuid_utils import UUID, uuid4, uuid7

__all__ = ["UUID", "uuid4", "uuid7"]
