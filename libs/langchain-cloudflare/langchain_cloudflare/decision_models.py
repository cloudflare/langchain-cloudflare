"""Cloudflare Workers AI decision models (Clef, Clef Flash, and Clef Omni)."""

# MARK: - IMPORTS
from typing import Any, Dict, List, Literal, Optional, Union

from ._native_models import WorkersAINativeModel

# MARK: - MODELS
DECISION_MODELS = (
    "@cf/cloudflare/clef",
    "@cf/cloudflare/clef-flash",
    "@cf/cloudflare/clef-omni",
)


# MARK: - DECISION MODEL
class CloudflareWorkersAIDecisionModel(WorkersAINativeModel):
    """Evaluate state against Clef's typed questions and return probabilities.

    Questions use Cloudflare's native ``noul`` (yes/no probability), ``choice``
    (categorical decision), or ``score`` (ordered rubric) schemas. The returned
    dictionary preserves ``model``, ``answers``, and ``usage`` from the API.
    These models use a decision API rather than chat messages or tool calls.

    REST usage::

        model = CloudflareWorkersAIDecisionModel()
        result = model.evaluate(
            state="Checkout has been failing for every customer.",
            questions={
                "urgent": {"type": "noul", "instructions": "Is this urgent?"}
            },
        )

    Python Worker usage::

        model = CloudflareWorkersAIDecisionModel(binding=self.env.AI)
        result = await model.aevaluate(state=state, questions=questions)

    Credentials default to ``CF_ACCOUNT_ID`` and ``CF_AI_API_TOKEN``. A Worker
    binding authenticates through the runtime and needs neither credential.
    """

    model_name: Literal[
        "@cf/cloudflare/clef", "@cf/cloudflare/clef-flash", "@cf/cloudflare/clef-omni"
    ] = "@cf/cloudflare/clef"

    # MARK: - PAYLOAD
    def _prepare_payload(
        self,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]],
        audio: Optional[List[Union[str, Dict[str, str]]]],
        videos: Optional[List[Union[str, Dict[str, str]]]],
    ) -> Dict[str, Any]:
        """Derive the required body selector from the endpoint model ID."""
        if (audio is not None or videos is not None) and self.model_name != (
            "@cf/cloudflare/clef-omni"
        ):
            raise ValueError("Audio and video inputs require @cf/cloudflare/clef-omni")
        payload: Dict[str, Any] = {
            "model": self.model_name.rsplit("/", 1)[-1],
            "state": state,
            "questions": questions,
        }
        for field, media in (("images", images), ("audio", audio), ("videos", videos)):
            if media is not None:
                payload[field] = media
        return payload

    # MARK: - EVALUATION
    def evaluate(
        self,
        *,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]] = None,
        audio: Optional[List[Union[str, Dict[str, str]]]] = None,
        videos: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate text or JSON state through REST.

        ``questions`` maps 1–64 IDs to native question objects. Optional images
        are embedded PNG/JPEG/WebP data URLs or ``{content_type, base64}``
        objects; remote URLs are unsupported. Clef Omni also accepts ``audio``
        (WAV/MP3) and ``videos`` (MP4/WebM) in the same embedded formats.
        Use :meth:`aevaluate` in Workers.
        """
        if self.binding is not None:
            raise ValueError("Use aevaluate() with a Workers AI binding")
        payload = self._prepare_payload(state, questions, images, audio, videos)
        return self._run(payload)

    async def aevaluate(
        self,
        *,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Union[str, Dict[str, str]]]] = None,
        audio: Optional[List[Union[str, Dict[str, str]]]] = None,
        videos: Optional[List[Union[str, Dict[str, str]]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the same native input through async REST or ``env.AI.run``.

        Returns the same dictionary as :meth:`evaluate`, including each
        question's probabilities, confidence where applicable, and token usage.
        """
        payload = self._prepare_payload(state, questions, images, audio, videos)
        return await self._arun(payload)
