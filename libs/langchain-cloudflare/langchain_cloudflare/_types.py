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


# MARK: - AI Search Types
class AISearchScoringDetails(TypedDict):
    """Per-chunk scoring breakdown returned by AI Search."""

    vector_score: NotRequired[float]
    vector_rank: NotRequired[int]
    keyword_score: NotRequired[float]
    keyword_rank: NotRequired[int]
    reranking_score: NotRequired[float]
    fusion_method: NotRequired[str]


class AISearchItem(TypedDict):
    """Source item metadata for an AI Search chunk."""

    key: str
    metadata: NotRequired[Optional[Dict[str, Any]]]
    timestamp: NotRequired[Optional[float]]


class AISearchChunk(TypedDict):
    """A single content chunk returned by the AI Search ``/search`` endpoint."""

    id: str
    score: float
    text: str
    type: str
    item: NotRequired[AISearchItem]
    scoring_details: NotRequired[AISearchScoringDetails]
