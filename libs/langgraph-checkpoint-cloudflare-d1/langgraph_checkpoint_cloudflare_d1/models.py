from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ErrorSource(BaseModel):
    """Source information for an error."""

    pointer: str = ""


class Error(BaseModel):
    """Cloudflare D1 API error model."""

    code: int = 0
    message: str = ""
    documentation_url: str = ""
    source: ErrorSource = Field(default_factory=ErrorSource)


class Timings(BaseModel):
    """SQL execution timing information."""

    sql_duration_ms: float = 0.0

    class Config:
        """Pydantic configuration."""

        extra = "ignore"


class Meta(BaseModel):
    """Metadata for D1 query results."""

    served_by: str = ""
    served_by_region: str = ""
    served_by_primary: bool = True
    timings: Timings = Field(default_factory=Timings)
    duration: float = 0.0
    changes: int = 0
    last_row_id: int = 0
    changed_db: bool = False
    size_after: int = 0
    rows_read: int = 0
    rows_written: int = 0

    class Config:
        """Pydantic configuration."""

        extra = "ignore"


class D1QueryResult(BaseModel):
    """Represents a single query result from Cloudflare D1."""

    results: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = True
    meta: Meta = Field(default_factory=Meta)

    class Config:
        """Pydantic configuration."""

        extra = "ignore"


class D1Response(BaseModel):
    """Cloudflare D1 API response model."""

    result: List[D1QueryResult] = Field(default_factory=list)
    errors: List[Error] = Field(default_factory=list)
    messages: List[str] = Field(default_factory=list)
    success: bool = False

    class Config:
        """Pydantic configuration."""

        extra = "ignore"

    def get_rows(self) -> List[Dict[str, Any]]:
        """Helper method to get the results from the first query."""
        if (
            not self.result
            or not isinstance(self.result, list)
            or len(self.result) == 0
        ):
            return []
        return self.result[0].results
