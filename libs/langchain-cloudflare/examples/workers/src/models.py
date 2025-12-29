"""Pydantic models for structured output."""

from typing import List, Optional

from pydantic import BaseModel, Field

# MARK: - Entity Models


class Entity(BaseModel):
    """An entity mentioned in the text."""

    name: str = Field(description="Name of the entity")
    ticker: Optional[str] = Field(
        default=None, description="Stock ticker if applicable"
    )
    role: str = Field(description="Role of the entity")


# MARK: - Announcement Models


class Announcement(BaseModel):
    """An announcement extracted from text."""

    type: str = Field(
        description="Type: partnership, investment, regulatory, milestone, event, m&a"
    )
    context: str = Field(description="Brief context of the announcement")
    entities: List[Entity] = Field(
        default_factory=list, description="Entities involved"
    )


class Data(BaseModel):
    """Extracted announcements from text."""

    announcements: List[Announcement] = Field(default_factory=list)
