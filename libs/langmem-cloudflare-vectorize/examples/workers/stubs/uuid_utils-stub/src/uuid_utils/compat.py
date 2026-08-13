"""Pure-Python RFC 9562 UUIDv7 generator, standing in for `uuid_utils.compat`.

`langchain_core.utils.uuid.uuid7` calls this with either no arguments (use
the current time) or `timestamp`/`nanos` (Unix seconds + remaining
nanoseconds), matching the real `uuid_utils.compat.uuid7` signature, and
expects a standard-library `uuid.UUID` back.
"""

import os
import time
from typing import Optional
from uuid import UUID

_VERSION_BITS = 0x7 << 12  # version nibble (7) in the high bits of the 3rd group
_VARIANT_BITS = 0b10 << 62  # RFC 4122 variant in the high bits of the 4th group


def uuid7(timestamp: Optional[int] = None, nanos: Optional[int] = None) -> UUID:
    """Generate a UUIDv7: 48-bit millisecond timestamp + random bits."""
    if timestamp is None:
        nanoseconds = time.time_ns()
    else:
        nanoseconds = timestamp * 1_000_000_000 + (nanos or 0)
    unix_ts_ms = nanoseconds // 1_000_000

    rand = os.urandom(10)  # 80 bits of randomness

    ts_bytes = (unix_ts_ms & 0xFFFFFFFFFFFF).to_bytes(6, "big")

    rand_a = int.from_bytes(rand[0:2], "big") & 0x0FFF
    ver_rand_bytes = (_VERSION_BITS | rand_a).to_bytes(2, "big")

    rand_b = int.from_bytes(rand[2:10], "big") & ((1 << 62) - 1)
    variant_rand_bytes = (_VARIANT_BITS | rand_b).to_bytes(8, "big")

    return UUID(bytes=ts_bytes + ver_rand_bytes + variant_rand_bytes)


__all__ = ["uuid7"]
