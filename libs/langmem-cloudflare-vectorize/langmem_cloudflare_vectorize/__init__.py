from importlib import metadata

from langmem_cloudflare_vectorize.langmem_vectorize_basestore import (
    CloudflareVectorizeBaseStore,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "CloudflareVectorizeBaseStore",
    "__version__",
]
