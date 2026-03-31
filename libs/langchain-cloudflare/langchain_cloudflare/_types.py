# MARK: - Imports
from typing import Any, Dict, List, Literal, Optional

from typing_extensions import NotRequired, TypedDict


# MARK: - Headers
class Headers(TypedDict):
    """Headers for Cloudflare API requests."""

    Authorization: str


VectorizeHeaders = TypedDict(
    "VectorizeHeaders",
    {
        "Authorization": str,
        "Content-Type": str,
    },
)


# MARK: - Binding Query Options
class BindingQueryOptions(TypedDict):
    """Options for querying a Cloudflare Vectorize binding."""

    topK: int
    filter: NotRequired[Dict[str, Any]]
    namespace: NotRequired[str]
    returnMetadata: NotRequired[str]
    returnValues: NotRequired[Literal[True]]


# MARK: - Vectorize Dict Result Type
class VectorizedDict(TypedDict):
    """Result type for vectorized data."""

    id: str
    text: str
    values: List[float]
    namespace: NotRequired[Optional[str]]
    metadata: NotRequired[Optional[Dict[str, Any]]]
