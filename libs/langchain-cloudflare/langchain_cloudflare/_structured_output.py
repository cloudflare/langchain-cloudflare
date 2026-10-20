"""Compile supported Pydantic schemas into native extraction and decisions."""

# MARK: - IMPORTS
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Type, Union

from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from .decision_models import CloudflareWorkersAIDecisionModel
from .extraction_models import CloudflareWorkersAIExtractionModel


# MARK: - SCHEMA COMPILATION
@dataclass(frozen=True)
class NativeField:
    """Keep native IDs separate from Pydantic property paths and enum values."""

    path: Tuple[str, ...]
    key: str
    spec: Dict[str, Any]
    choices: Optional[Dict[str, Any]] = None


def _compile_schema(
    schema: Union[Type[BaseModel], Dict[str, Any]],
) -> List[NativeField]:
    if isinstance(schema, dict):
        document = schema
        schema_name = document.get("title", "structured output")
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        document = schema.model_json_schema(mode="validation")
        schema_name = schema.__name__
    else:
        raise TypeError("schema must be a Pydantic BaseModel class or JSON schema")
    fields: List[NativeField] = []

    def visit(
        node: Dict[str, Any],
        path: Tuple[str, ...],
        references: Tuple[str, ...] = (),
        descriptions: Tuple[str, ...] = (),
    ) -> None:
        label = ".".join(path) or schema_name

        def unsupported(reason: str) -> NoReturn:
            raise ValueError(f"Unsupported structured-output field {label!r}: {reason}")

        if "$ref" in node:
            reference = node["$ref"]
            if not reference.startswith("#/$defs/") or reference in references:
                unsupported("external or recursive references")
            name = (
                reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
            )
            node = {**document["$defs"][name], **node}
            references = (*references, reference)
        if node.get("description"):
            descriptions = (*descriptions, node["description"])
        nullable = False
        if "anyOf" in node:
            alternatives = node["anyOf"]
            non_null = [item for item in alternatives if item.get("type") != "null"]
            if len(alternatives) != 2 or len(non_null) != 1:
                unsupported("only a single type combined with None is supported")
            nullable = True
            node = {**non_null[0], **{k: v for k, v in node.items() if k != "anyOf"}}
            # Resolve a nullable reference through the same cycle checks.
            if "$ref" in node:
                reference = node["$ref"]
                if not reference.startswith("#/$defs/") or reference in references:
                    unsupported("external or recursive references")
                name = (
                    reference.removeprefix("#/$defs/")
                    .replace("~1", "/")
                    .replace("~0", "~")
                )
                node = {**document["$defs"][name], **node}
                references = (*references, reference)
        if "oneOf" in node or "allOf" in node:
            unsupported("schema composition")
        kind = node.get("type")
        if kind == "object":
            if nullable:
                unsupported("optional objects require an explicit presence decision")
            properties = node.get("properties")
            if not properties or node.get("additionalProperties") not in (None, False):
                unsupported("empty models or dynamic dictionaries")
            for name, child in properties.items():
                visit(child, (*path, name), references, descriptions)
            return
        if not path:
            unsupported("the root must be a model with named fields")
        instructions = f"Field {label}. " + " ".join(descriptions)
        spec: Dict[str, Any] = {"instructions": instructions.strip()}
        choices = None
        if "enum" in node or "const" in node or (kind == "boolean" and nullable):
            values = node.get("enum", [node.get("const")])
            if kind == "boolean" and "enum" not in node and "const" not in node:
                values = [True, False]
            if nullable and None not in values:
                values = [*values, None]
            if len(values) < 2 or not all(
                value is None or isinstance(value, (str, int, float, bool))
                for value in values
            ):
                unsupported("choices require at least two scalar alternatives")
            choices = {f"c{i}": value for i, value in enumerate(values)}
            spec.update(
                type="choice",
                criteria={
                    key: (
                        "No applicable value is stated or determined"
                        if value is None
                        else json.dumps(value, ensure_ascii=False)
                    )
                    for key, value in choices.items()
                },
            )
        elif kind == "boolean":
            spec["type"] = "noul"
        elif kind in ("integer", "number"):
            spec["type"] = "number"
            if kind == "integer":
                spec["decimals"] = 0
        elif kind == "string":
            spec["type"] = {
                "date": "date",
                "time": "time",
                "date-time": "datetime",
            }.get(node.get("format", ""), "text")
        elif kind == "array":
            item = node.get("items", {})
            if (
                item.get("type") != "string"
                or any(
                    key in item for key in ("$ref", "enum", "const", "anyOf", "format")
                )
                or "prefixItems" in node
            ):
                unsupported("only lists of plain strings are supported")
            spec.update(type="list", items="text")
        else:
            unsupported(f"type {kind!r}")
        fields.append(NativeField(path, f"f{len(fields)}", spec, choices))

    visit(document, ())
    return fields


# MARK: - RUNNABLE CONSTRUCTION
def native_structured_output(
    model: Any,
    schema: Union[Type[BaseModel], Dict[str, Any]],
    *,
    extraction_model: Optional[CloudflareWorkersAIExtractionModel],
    decision_model: Optional[CloudflareWorkersAIDecisionModel],
    include_raw: bool,
    boolean_threshold: float,
) -> Runnable[Any, Any]:
    """Build a Runnable using the existing native adapters without new IO paths."""
    if not math.isfinite(boolean_threshold) or not 0 <= boolean_threshold <= 1:
        raise ValueError("boolean_threshold must be a finite number between 0 and 1")
    if extraction_model is not None and not isinstance(
        extraction_model, CloudflareWorkersAIExtractionModel
    ):
        raise TypeError("extraction_model must be CloudflareWorkersAIExtractionModel")
    if decision_model is not None and not isinstance(
        decision_model, CloudflareWorkersAIDecisionModel
    ):
        raise TypeError("decision_model must be CloudflareWorkersAIDecisionModel")
    fields = _compile_schema(schema)
    if isinstance(model, CloudflareWorkersAIExtractionModel):
        if extraction_model is not None:
            raise ValueError("Pass decision_model as the extraction model's companion")
        extraction_model = model
    elif isinstance(model, CloudflareWorkersAIDecisionModel):
        if decision_model is not None:
            raise ValueError("Pass extraction_model as the decision model's companion")
        decision_model = model
    else:
        raise TypeError("Structured output requires a Clairic or Clef model")
    extraction_fields = {
        field.key: field.spec
        for field in fields
        if field.spec["type"] not in ("noul", "choice")
    }
    questions = {
        field.key: field.spec
        for field in fields
        if field.spec["type"] in ("noul", "choice")
    }
    if extraction_fields and extraction_model is None:
        raise ValueError("This schema requires an explicit extraction_model (Clairic)")
    if questions and decision_model is None:
        raise ValueError("This schema requires an explicit decision_model (Clef)")
    if len(questions) > 64:
        raise ValueError("Clef supports at most 64 decision fields per request")

    def payloads(value: Union[str, Dict[str, Any]]) -> Tuple[dict, dict]:
        if isinstance(value, str):
            data: Dict[str, Any] = {"state": value}
        elif isinstance(value, dict):
            data = dict(value)
        else:
            raise TypeError("Input must be text or a native input dictionary")
        unknown = set(data) - {
            "state",
            "images",
            "audio",
            "videos",
            "reference_date",
            "options",
        }
        if unknown:
            raise ValueError(f"Unsupported input keys: {sorted(unknown)}")
        data.setdefault("state", "")
        if extraction_fields and not isinstance(data["state"], str):
            raise TypeError("Clairic structured extraction requires a string state")
        if extraction_fields and any(
            data.get(key) is not None for key in ("audio", "videos")
        ):
            raise ValueError(
                "Audio/video input is supported only for decision-only schemas"
            )
        if not extraction_fields and any(
            key in data for key in ("reference_date", "options")
        ):
            raise ValueError("reference_date/options require extraction fields")
        extraction_input = {
            key: val
            for key, val in data.items()
            if key in ("state", "images", "reference_date", "options")
        }
        decision_input = {
            key: val
            for key, val in data.items()
            if key in ("state", "images", "audio", "videos")
        }
        return extraction_input, decision_input

    def parse(raw: Dict[str, Any]) -> Any:
        values: Dict[str, Any] = {}
        for stage, expected in (
            ("extraction", extraction_fields),
            ("decision", questions),
        ):
            if expected:
                response = raw[stage]
                if response.get("truncated"):
                    raise ValueError(f"{stage} input was truncated")
                if set(response["answers"]) != set(expected):
                    raise ValueError(
                        f"{stage} answer IDs do not match the schema fields"
                    )
        for field in fields:
            kind = field.spec["type"]
            stage = "decision" if kind in ("choice", "noul") else "extraction"
            answer = raw[stage]["answers"][field.key]
            if answer["type"] != kind:
                raise ValueError(f"Wrong native answer type for {'.'.join(field.path)}")
            if kind == "noul":
                probability = answer["noul"]
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, (float, int))
                    or not 0 <= probability <= 1
                ):
                    raise ValueError("Invalid native boolean probability")
                value = probability > boolean_threshold
            elif field.choices is not None:
                value = field.choices[answer["choice"]]
            else:
                value = answer["value"]
            target = values
            for part in field.path[:-1]:
                target = target.setdefault(part, {})
            target[field.path[-1]] = value
        return values if isinstance(schema, dict) else schema.model_validate(values)

    def finish(raw: Dict[str, Any]) -> Any:
        try:
            parsed = parse(raw)
        except Exception as error:
            if not include_raw:
                raise
            return {"raw": raw, "parsed": None, "parsing_error": error}
        if include_raw:
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return parsed

    def invoke(value: Union[str, Dict[str, Any]]) -> Any:
        extraction_input, decision_input = payloads(value)
        # Reject sync binding use before either stage can make a paid REST call.
        if any(
            active and native is not None and native.binding is not None
            for active, native in (
                (extraction_fields, extraction_model),
                (questions, decision_model),
            )
        ):
            raise ValueError("Use ainvoke()/abatch() with Workers AI bindings")
        raw = {}
        if extraction_fields and extraction_model is not None:
            raw["extraction"] = extraction_model.extract(
                fields=extraction_fields, **extraction_input
            )
        if questions and decision_model is not None:
            raw["decision"] = decision_model.evaluate(
                questions=questions, **decision_input
            )
        return finish(raw)

    async def ainvoke(value: Union[str, Dict[str, Any]]) -> Any:
        extraction_input, decision_input = payloads(value)
        raw = {}
        if extraction_fields and extraction_model is not None:
            raw["extraction"] = await extraction_model.aextract(
                fields=extraction_fields, **extraction_input
            )
        if questions and decision_model is not None:
            raw["decision"] = await decision_model.aevaluate(
                questions=questions, **decision_input
            )
        return finish(raw)

    return RunnableLambda(
        invoke, afunc=ainvoke, name="NativeStructuredOutput"
    ).with_types(
        output_type=dict if include_raw or isinstance(schema, dict) else schema
    )
