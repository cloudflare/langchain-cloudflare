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
    def packb(obj: Any, **kwargs) -> bytes:
        """Serialize object to JSON bytes (fallback)."""
        return json.dumps(obj).encode("utf-8")

    def unpackb(data: bytes, **kwargs) -> Any:
        """Deserialize JSON bytes to object (fallback)."""
        return json.loads(data.decode("utf-8"))


# Export options that ormsgpack defines (referenced by langgraph-checkpoint)
OPT_SERIALIZE_NUMPY = 1
OPT_SERIALIZE_DATACLASS = 2
OPT_SERIALIZE_UUID = 4
OPT_UTC_Z = 8
OPT_NAIVE_UTC = 16
OPT_OMIT_MICROSECONDS = 32
OPT_PASSTHROUGH_BIG_INT = 64
OPT_PASSTHROUGH_DATACLASS = 128
OPT_PASSTHROUGH_DATETIME = 256
OPT_PASSTHROUGH_SUBCLASS = 512
OPT_NON_STR_KEYS = 1024
OPT_PASSTHROUGH_ENUM = 2048
OPT_PASSTHROUGH_UUID = 4096

__version__ = "1.10.0"  # Stub version matching requirement
