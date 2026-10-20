"""Adapt LangChain agent schema bindings to the shared native output pipeline."""

# MARK: - IMPORTS
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from ._structured_output import native_structured_output


# MARK: - SCHEMA BINDING
def _schema_from_format(response_format: Any) -> Dict[str, Any]:
    if (
        not isinstance(response_format, dict)
        or response_format.get("type") != "json_schema"
    ):
        raise ValueError(
            "Native agent calls require response_format with a JSON schema"
        )
    spec = response_format.get("json_schema", {})
    if spec.get("strict"):
        raise ValueError(
            "Native models validate locally; provider-side strict decoding "
            "is unavailable"
        )
    schema = spec.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("response_format.json_schema.schema must be an object")
    return schema


def _pipeline(
    model: Any, schema: Dict[str, Any], threshold: float
) -> Runnable[Any, Any]:
    return native_structured_output(
        model,
        schema,
        extraction_model=model.extraction_model,
        decision_model=model.decision_model,
        include_raw=True,
        boolean_threshold=threshold,
    )


def bind_native_tools(
    model: Any, tools: Sequence[Any], *, tool_choice: Optional[str], **kwargs: Any
) -> Runnable[Any, AIMessage]:
    """Accept output declarations, never executable user tools."""
    # LangChain 1.0 always supplies this tool-binding hint for ProviderStrategy.
    # It is not a native inference option or provider-side JSON-schema decoding.
    kwargs.pop("strict", None)
    if "response_format" in kwargs:
        if tools:
            raise ValueError(
                "Native structured agents do not execute application tools"
            )
        schema = _schema_from_format(kwargs["response_format"])
        _pipeline(
            model, schema, kwargs.get("boolean_threshold", model.boolean_threshold)
        )
        return model.bind(**kwargs)  # type: ignore[no-any-return]
    if len(tools) != 1:
        raise ValueError("Native ToolStrategy requires exactly one output schema")
    tool = tools[0]
    declarative = (
        isinstance(tool, StructuredTool)
        and tool.func is None
        and tool.coroutine is None
    ) or (isinstance(tool, type) and issubclass(tool, BaseModel))
    if not declarative:
        raise ValueError(
            "Native models support schema output tools, "
            "not executable application tools"
        )
    function = convert_to_openai_tool(tool)["function"]
    if tool_choice not in (None, "any", "required", "auto", function["name"]):
        raise ValueError("Unsupported native output tool_choice")
    _pipeline(
        model,
        function["parameters"],
        kwargs.get("boolean_threshold", model.boolean_threshold),
    )
    return model.bind(native_output_tool=function, **kwargs)  # type: ignore[no-any-return]


# MARK: - MESSAGE INPUT
def native_chat_request(
    model: Any,
    messages: List[BaseMessage],
    stop: Optional[List[str]],
    kwargs: Dict[str, Any],
) -> Tuple[Runnable[Any, Any], Dict[str, Any], Optional[str]]:
    """Separate instructions from source text and forward embedded images."""
    if stop is not None:
        raise ValueError("Native structured models do not support stop sequences")
    unknown = set(kwargs) - {
        "response_format",
        "native_output_tool",
        "boolean_threshold",
        "reference_date",
        "options",
    }
    if unknown:
        raise ValueError(f"Unsupported native chat arguments: {sorted(unknown)}")
    tool = kwargs.get("native_output_tool")
    schema = (
        tool["parameters"]
        if tool
        else _schema_from_format(kwargs.get("response_format"))
    )
    instructions, sources, images = [], [], []
    output_call_ids: set[str] = set()
    converted = convert_to_openai_messages(messages)
    for message in converted:
        if message.get("tool_calls"):
            calls = message["tool_calls"]
            if not tool or any(
                call["function"]["name"] != tool["name"] for call in calls
            ):
                raise ValueError(
                    "Native agents do not consume application-tool histories"
                )
            output_call_ids.update(call["id"] for call in calls)
            instructions.append("Previous structured answer: " + json.dumps(calls))
            continue
        if message["role"] in ("tool", "function"):
            if message.get("tool_call_id") not in output_call_ids:
                raise ValueError(
                    "Native agents do not consume application-tool histories"
                )
            instructions.append(
                "Structured-output validation feedback: "
                + str(message.get("content", ""))
            )
            continue
        content = message.get("content", "")
        text = []
        for block in (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        ):
            kind = block.get("type")
            if kind == "text":
                text.append(block["text"])
            elif kind == "image_url" and message["role"] == "user":
                image_url = block["image_url"]
                url = image_url["url"] if isinstance(image_url, dict) else image_url
                if (
                    not isinstance(url, str)
                    or not url.startswith("data:image/")
                    or ";base64," not in url
                ):
                    raise ValueError(
                        "Native agent images must be embedded base64 data URLs"
                    )
                images.append(url)
            else:
                raise ValueError(f"Unsupported native chat content block: {kind!r}")
        rendered = "\n".join(text)
        if message["role"] in ("system", "developer"):
            instructions.append(rendered)
        else:
            sources.append((message["role"], rendered))
    if instructions:
        schema = {
            **schema,
            "description": "\n".join([schema.get("description", ""), *instructions]),
        }
    state = (
        sources[0][1]
        if len(sources) == 1
        else "\n\n".join(f"{role}: {text}" for role, text in sources)
    )
    if not state.strip() and not images:
        raise ValueError("Native agent input requires source text or embedded images")
    payload: Dict[str, Any] = {"state": state}
    if images:
        payload["images"] = images
    for key in ("reference_date", "options"):
        if key in kwargs:
            payload[key] = kwargs[key]
    return (
        _pipeline(
            model, schema, kwargs.get("boolean_threshold", model.boolean_threshold)
        ),
        payload,
        tool["name"] if tool else None,
    )


# MARK: - CHAT OUTPUT
def native_chat_result(
    result: Dict[str, Any], model_name: str, tool_name: Optional[str]
) -> ChatResult:
    if result["parsing_error"] is not None:
        raise result["parsing_error"]
    values, raw = result["parsed"], result["raw"]
    input_tokens = sum(
        stage.get("usage", {}).get("input_tokens", 0) for stage in raw.values()
    )
    output_tokens = sum(
        stage.get("usage", {}).get("output_tokens", 0) for stage in raw.values()
    )
    message = AIMessage(
        content="" if tool_name else json.dumps(values),
        tool_calls=[
            {
                "name": tool_name,
                "args": values,
                "id": f"native_{uuid4().hex}",
                "type": "tool_call",
            }
        ]
        if tool_name
        else [],
        response_metadata={"model_name": model_name, "native": raw},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return ChatResult(generations=[ChatGeneration(message=message)])
