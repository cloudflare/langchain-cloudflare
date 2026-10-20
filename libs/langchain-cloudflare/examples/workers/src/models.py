"""Pydantic models for structured output."""

from datetime import date as Date
from enum import Enum
from typing import List, Literal, Optional

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


# MARK: - Native Structured Output
class NativeInvoice(BaseModel):
    """Invoice facts extracted from the source."""

    invoice_number: str = Field(description="Invoice number")
    due_date: Date = Field(description="Issue date plus the net payment days")
    total: float = Field(description="Amount due including shipping, in USD")
    contacts: List[str] = Field(description="Full names of the contact people")
    po_number: Optional[str] = Field(default=None, description="Purchase order number")


class SupportTeam(str, Enum):
    billing = "billing"
    technical = "technical"


class NativeSupportDecision(BaseModel):
    """Support-request classification."""

    urgent: bool = Field(description="Is this support request urgent?")
    team: SupportTeam = Field(description="Which team should handle this request?")


class NativeInvoiceReview(BaseModel):
    """Invoice facts and decisions about the same source."""

    invoice: NativeInvoice
    paid: bool = Field(
        description="Does the source explicitly say payment is completed?"
    )
    department: Literal["billing", "technical"] = Field(
        description="Which department should handle this invoice?"
    )


class NativeImageFacts(BaseModel):
    """Facts printed in the attached image."""

    ticker: str = Field(description="Ticker printed under TICKER")
    timeframe: str = Field(description="Timeframe printed under TIMEFRAME")
    date: Date = Field(description="Date printed under DATE")
