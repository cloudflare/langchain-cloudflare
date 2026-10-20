"""Compose Clairic and Clef for one primary announcement from source lines.

Each nonempty input line is a context candidate. This focused example reuses
native model calls and the existing nested Pydantic output schema; it does not
translate arbitrary Pydantic schemas or discover groups of multiple events.
"""

# MARK: - IMPORTS
from langchain_cloudflare import (
    CloudflareWorkersAIDecisionModel,
    CloudflareWorkersAIExtractionModel,
)

if __package__:
    from .models import Announcement, Data, Entity
else:
    # Wrangler loads sibling source modules directly; pytest uses the package.
    from models import Announcement, Data, Entity

# MARK: - DECISION DEFINITIONS
ELIGIBILITY = (
    "Eligible announcements are new partnerships, investments, regulatory "
    "approvals, milestones, events, or mergers/acquisitions. Routine "
    "financial results alone are not an eligible announcement."
)
ANNOUNCEMENT_TYPES = {
    "partnership": "Companies agree to collaborate or jointly develop",
    "investment": "Equity funding or a capital investment",
    "regulatory": "An approval or license granted by a regulator",
    "milestone": "A substantial internal accomplishment",
    "event": "Hosting or participating in an event",
    "m&a": "Acquiring, merging, or selling a business",
    "none": "No eligible announcement, including routine earnings",
}


# MARK: - SOURCE EVIDENCE
def group_list_spans(
    *, text: str, values: list[str], spans: list[list[int]]
) -> dict[str, list[list[int]]]:
    """Anchor every occurrence by exact substring, preserving value order.

    Native list values and occurrence spans need not have equal lengths or
    matching order. Reject spans outside the source or absent from the values,
    and values without any matching occurrence. Do not infer or repair names.
    """
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError("List values must be strings")
    if not isinstance(spans, list):
        raise ValueError("List spans must be a list")
    grouped: dict[str, list[list[int]]] = {value: [] for value in values}
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(type(offset) is int for offset in span)
        ):
            raise ValueError("Each source span must contain two integer offsets")
        start, end = span
        if not 0 <= start < end <= len(text):
            raise ValueError("Source span is outside the input text")
        value = text[start:end]
        if value not in grouped:
            raise ValueError("Source span does not match any list value")
        grouped[value].append(span)
    if any(not occurrences for occurrences in grouped.values()):
        raise ValueError("List value has no matching source span")
    return grouped


# MARK: - COMBINED EXAMPLE
async def extract_announcements(
    *,
    text: str,
    extraction_model: CloudflareWorkersAIExtractionModel,
    decision_model: CloudflareWorkersAIDecisionModel,
) -> dict:
    """Extract candidates, decide their roles, and validate nested results.

    Callers supply REST- or binding-backed native model instances. The
    returned ``stages`` preserve the provider responses, including rejected
    candidate spans; ``extracted`` contains the validated ``Data`` structure.
    Only exact duplicate candidate strings are removed in code. Clef makes
    all semantic inclusion/role decisions, and context is its chosen source
    line verbatim. No expected fixture names or classification labels are
    supplied as answers.
    """
    # Clairic supplies candidate mentions, including possible false positives.
    candidates_result = await extraction_model.aextract(
        state=text,
        fields={
            "companies": {
                "type": "list",
                "items": "text",
                "instructions": "Names of the companies mentioned, in source order",
            }
        },
    )
    mentions = candidates_result["answers"]["companies"]
    occurrences = group_list_spans(
        text=text, values=mentions["value"], spans=mentions["spans"]
    )
    candidates = list(occurrences)
    if len(candidates) > 61:
        raise ValueError("This example supports at most 61 candidate mentions")

    # Candidate sentences come from the input, not the test's expected answer.
    contexts = {
        f"S{i}": line for i, line in enumerate(text.splitlines()) if line.strip()
    }
    questions = {
        "is_announcement": {
            "type": "noul",
            "instructions": (
                f"Does this release contain an eligible announcement? {ELIGIBILITY}"
            ),
        },
        "announcement_type": {
            "type": "choice",
            "instructions": f"Classify the primary announcement. {ELIGIBILITY}",
            "criteria": ANNOUNCEMENT_TYPES,
        },
        "context": {
            "type": "choice",
            "instructions": (
                "Select the source sentence stating the primary eligible "
                f"announcement, rather than supporting detail. {ELIGIBILITY} "
                "Choose none when there is no eligible announcement."
            ),
            "criteria": {**contexts, "none": "No eligible announcement context"},
        },
    }
    for i, name in enumerate(candidates):
        questions[f"role_{i}"] = {
            "type": "choice",
            "instructions": (
                f"What role does the exact source span {name!r} play in the "
                f"eligible announcement? {ELIGIBILITY} Choose None for exchanges, "
                "ticker symbols, shortened or possessive aliases, and spans that "
                "are not full company names directly involved in the announcement."
            ),
            "criteria": {
                "Partner": "A full company name participating in the agreement",
                "Company": (
                    "The full company name whose internal milestone is announced"
                ),
                "None": (
                    "Not a full participant company name, or no eligible announcement"
                ),
            },
        }
    decision_result = await decision_model.aevaluate(state=text, questions=questions)
    answers = decision_result["answers"]
    stages = {"candidates": candidates_result, "decisions": decision_result}
    if (
        answers["is_announcement"]["noul"] <= 0.5
        or answers["announcement_type"]["choice"] == "none"
        or answers["context"]["choice"] == "none"
    ):
        return {"extracted": Data().model_dump(), "stages": stages}

    participants = [
        (name, answers[f"role_{i}"]["choice"])
        for i, name in enumerate(candidates)
        if answers[f"role_{i}"]["choice"] != "None"
    ]
    # Tickers are facts extracted for model-selected names, never inferred.
    ticker_values = {}
    if participants:
        ticker_result = await extraction_model.aextract(
            state=text,
            fields={
                f"ticker_{i}": {
                    "type": "text",
                    "instructions": (
                        f"Stock ticker explicitly written for {name}; return "
                        "null if absent. Do not infer it from prior knowledge."
                    ),
                }
                for i, (name, _) in enumerate(participants)
            },
        )
        stages["tickers"] = ticker_result
        ticker_values = ticker_result["answers"]
    result = Data(
        announcements=[
            Announcement(
                type=answers["announcement_type"]["choice"],
                context=contexts[answers["context"]["choice"]],
                entities=[
                    Entity(
                        name=name,
                        role=role,
                        ticker=ticker_values[f"ticker_{i}"]["value"],
                    )
                    for i, (name, role) in enumerate(participants)
                ],
            )
        ]
    )
    return {"extracted": result.model_dump(), "stages": stages}
