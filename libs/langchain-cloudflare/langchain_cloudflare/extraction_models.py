"""Cloudflare Workers AI extraction models (Clairic)."""

# MARK: - IMPORTS
from typing import Any, Dict, List, Literal, Optional, Union

from ._native_models import WorkersAINativeModel

# MARK: - MODELS
EXTRACTION_MODELS = ("@cf/cloudflare/clairic",)


# MARK: - EXTRACTION MODEL
class CloudflareWorkersAIExtractionModel(WorkersAINativeModel):
    """Extract typed values through Clairic's native state/fields API.

    Fields use Cloudflare's ``text``, ``number``, ``date``, ``time``,
    ``datetime``, ``duration``, ``list``, ``estimate`` or ``forecast`` specs.
    Results preserve native answers, source spans, uncertainty, truncation
    and token usage. Missing single-value answers may be ``None``; lists
    may be empty. A valid type does not guarantee a correct answer.

    REST usage::

        model = CloudflareWorkersAIExtractionModel()
        result = model.extract(
            state="Invoice INV-2291 totals $1080.",
            fields={"total": {"type": "number", "unit": "USD"}},
        )

    Python Worker usage::

        model = CloudflareWorkersAIExtractionModel(binding=self.env.AI)
        result = await model.aextract(state=state, fields=fields)

    REST credentials default to ``CF_ACCOUNT_ID`` and ``CF_AI_API_TOKEN``.
    Worker bindings authenticate through the existing runtime.
    """

    model_name: Literal["@cf/cloudflare/clairic"] = "@cf/cloudflare/clairic"

    # MARK: - PAYLOAD
    def _prepare_payload(
        self,
        state: str,
        fields: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]],
        reference_date: Optional[str],
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": "clairic",
            "state": state,
            "fields": fields,
        }
        if images is not None:
            payload["images"] = images
        if reference_date is not None:
            payload["reference_date"] = reference_date
        if options is not None:
            payload["options"] = options
        return payload

    # MARK: - EXTRACTION
    def extract(
        self,
        *,
        state: str = "",
        fields: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]] = None,
        reference_date: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract typed values through synchronous REST.

        ``fields`` contains native Clairic field specs. Text read from
        ``state`` includes source spans. Image-derived text can instead have
        ``source="image"``, a null span and ``spell_confidence``. These native
        fields are preserved unchanged. Optional images must be PNG/JPEG/WebP
        data URLs or ``{content_type, base64}`` objects, not remote URLs.
        ``reference_date`` anchors dates without a year. Native ``options``
        can request quantiles or returned state text. Use :meth:`aextract`
        with a Worker binding.
        """
        if self.binding is not None:
            raise ValueError("Use aextract() with a Workers AI binding")
        payload = self._prepare_payload(state, fields, images, reference_date, options)
        return self._run(payload)

    async def aextract(
        self,
        *,
        state: str = "",
        fields: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]] = None,
        reference_date: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract the same native values through async REST or ``env.AI.run``.

        Model options remain in the payload; gateway and reject-if-busy
        settings use the binding's existing third-argument run options.
        """
        payload = self._prepare_payload(state, fields, images, reference_date, options)
        return await self._arun(payload)
