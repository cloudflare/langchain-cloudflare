from importlib import metadata

from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI
from langchain_cloudflare.document_loaders import CloudflareWorkersAILoader
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_cloudflare.retrievers import CloudflareWorkersAIRetriever
from langchain_cloudflare.toolkits import CloudflareWorkersAIToolkit
from langchain_cloudflare.tools import CloudflareWorkersAITool
from langchain_cloudflare.vectorstores import CloudflareWorkersAIVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCloudflareWorkersAI",
    "CloudflareWorkersAIVectorStore",
    "CloudflareWorkersAIEmbeddings",
    "CloudflareWorkersAILoader",
    "CloudflareWorkersAIRetriever",
    "CloudflareWorkersAIToolkit",
    "CloudflareWorkersAITool",
    "__version__",
]
