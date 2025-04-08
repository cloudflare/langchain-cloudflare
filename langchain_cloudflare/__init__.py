from importlib import metadata

from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_cloudflare.vectorstores import CloudflareVectorize

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCloudflareWorkersAI",
    "CloudflareVectorize",
    "CloudflareWorkersAIEmbeddings",
    "__version__",
]
