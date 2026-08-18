"""Stub uuid_utils module for Pyodide compatibility.

uuid-utils is a compiled Rust extension with no Pyodide/WebAssembly wheel, so
it cannot be installed into the Worker bundle. langsmith (pulled in transitively
by langchain-core) imports ``uuid_utils.compat.uuid7`` from its internal
``_uuid`` helper, which is enough to abort Worker startup at import time.

This stub reimplements the small surface langsmith actually touches in pure
Python, following RFC 9562 for the UUIDv7 layout.

Note: the native package generates the counter bits with a monotonic
per-millisecond counter. This stub uses random bits instead, so UUIDs created
within the same millisecond are not guaranteed to sort against each other.
Ordering across different milliseconds -- which is what the timestamp prefix is
actually for -- still holds.
"""

import os
import time
import uuid
from typing import Optional

# MARK: - UUID Generation


def uuid7(timestamp: Optional[int] = None, nanos: Optional[int] = None) -> uuid.UUID:
    """Generate a UUIDv7 (time-ordered, RFC 9562 section 5.7).

    Layout:
        [0-5]  48 bits  unix_ts_ms
        [6]     4 bits  version (0111) + 4 bits rand_a
        [7]     8 bits  rand_a (continued)
        [8]     2 bits  variant (10) + 6 bits rand_b
        [9-15] 56 bits  rand_b (continued)

    Args:
        timestamp: Optional Unix timestamp in whole seconds. Defaults to now.
        nanos: Optional nanosecond remainder to add to ``timestamp``.

    Returns:
        A stdlib ``uuid.UUID`` with version 7, matching what
        ``uuid_utils.compat`` returns (the compat module converts the native
        Rust UUID into a stdlib UUID).
    """
    if timestamp is None:
        timestamp_ms = time.time_ns() // 1_000_000
    else:
        timestamp_ms = timestamp * 1_000
        if nanos is not None:
            timestamp_ms += nanos // 1_000_000

    b = bytearray(os.urandom(16))

    # 48-bit big-endian millisecond timestamp.
    b[0:6] = (timestamp_ms & 0xFFFF_FFFF_FFFF).to_bytes(6, "big")

    # Version 7 in the high nibble of byte 6, preserving random low nibble.
    b[6] = 0x70 | (b[6] & 0x0F)

    # RFC 4122 variant (10xx) in the high bits of byte 8.
    b[8] = 0x80 | (b[8] & 0x3F)

    return uuid.UUID(bytes=bytes(b))


def uuid4() -> uuid.UUID:
    """Generate a random UUIDv4, delegating to the standard library."""
    return uuid.uuid4()


UUID = uuid.UUID

__all__ = ["UUID", "uuid4", "uuid7"]
