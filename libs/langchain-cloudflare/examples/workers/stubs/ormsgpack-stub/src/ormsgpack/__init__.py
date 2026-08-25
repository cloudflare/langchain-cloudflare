"""Stub ormsgpack module for Pyodide compatibility.

ormsgpack is a C extension that doesn't have a Pyodide wheel.
This stub provides the minimal API needed by langgraph-checkpoint
using msgpack-python as a fallback (which is pure Python).

Note: This stub allows langchain>=1.0.0 and create_agent to work
in Cloudflare Python Workers, but checkpointing may have reduced
performance compared to native ormsgpack.
"""

import json
from typing import Any

# Try to use msgpack if available, otherwise fall back to JSON
try:
    import msgpack

    def packb(obj: Any, **kwargs) -> bytes:
        """Serialize object to MessagePack bytes using msgpack."""
        return msgpack.packb(obj, use_bin_type=True)

    def unpackb(data: bytes, **kwargs) -> Any:
        """Deserialize MessagePack bytes to object using msgpack."""
        return msgpack.unpackb(data, raw=False)

except ImportError:
    # Fall back to JSON if msgpack not available
    def packb(obj: Any, **kwargs) -> bytes:  # type: ignore[misc]
        """Serialize object to JSON bytes (fallback)."""
        return json.dumps(obj).encode("utf-8")

    def unpackb(data: bytes, **kwargs) -> Any:  # type: ignore[misc]
        """Deserialize JSON bytes to object (fallback)."""
        return json.loads(data.decode("utf-8"))


# Export options that ormsgpack defines (referenced by langgraph-checkpoint).
# Values match the real ormsgpack 1.12.x package so bitwise-OR'd option
# combinations behave identically to constant lookups; the stub's packb/unpackb
# ignore them (JSON/msgpack fallback has no native support for most of these).
OPT_NAIVE_UTC = 1
OPT_NON_STR_KEYS = 2
OPT_OMIT_MICROSECONDS = 4
OPT_PASSTHROUGH_BIG_INT = 8
OPT_PASSTHROUGH_DATACLASS = 16
OPT_PASSTHROUGH_DATETIME = 32
OPT_PASSTHROUGH_SUBCLASS = 64
OPT_PASSTHROUGH_TUPLE = 128
OPT_SERIALIZE_NUMPY = 256
OPT_SERIALIZE_PYDANTIC = 512
OPT_SORT_KEYS = 1024
OPT_UTC_Z = 2048
OPT_PASSTHROUGH_UUID = 4096
OPT_PASSTHROUGH_ENUM = 8192
OPT_DATETIME_AS_TIMESTAMP_EXT = 16384
OPT_REPLACE_SURROGATES = 32768

__version__ = "1.12.1"  # Stub version matching requirement
