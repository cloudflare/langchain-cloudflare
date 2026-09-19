"""Shared Workers AI request options for the REST paths.

The binding counterpart lives in :func:`langchain_cloudflare.bindings.
create_binding_run_options`, which builds the third argument to
``env.AI.run()``. Over REST the same options travel in the request body
instead, so both paths need their own construction step.
"""

# MARK: - Imports
from __future__ import annotations

from typing import Any, Dict, Optional

# MARK: - Request Options


def apply_reject_if_busy(
    payload: Dict[str, Any],
    reject_if_busy: Optional[bool],
) -> Dict[str, Any]:
    """Add ``options.rejectIfBusy`` to a REST request body.

    Cloudflare accepts a top-level ``options`` object on both the native
    ``/ai/run/{model}`` endpoint and the OpenAI-compatible
    ``/ai/v1/chat/completions`` endpoint, so the same shape works for either.

    A caller may already have supplied an ``options`` dict of their own (for
    example through ``model_kwargs``); their other keys are preserved and only
    ``rejectIfBusy`` is set, so the explicit field wins for that one key
    without discarding the rest.

    Args:
        payload: The request body being built. Not mutated.
        reject_if_busy: When True, request fail-fast behavior. When False or
            None, the body is returned without an ``options`` key added.

    Returns:
        The payload, with ``options.rejectIfBusy`` set when requested.
    """
    if not reject_if_busy:
        return payload

    existing = payload.get("options")
    options = dict(existing) if isinstance(existing, dict) else {}
    options["rejectIfBusy"] = True

    return {**payload, "options": options}


__all__ = ["apply_reject_if_busy"]
