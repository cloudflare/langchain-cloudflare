"""Stub sync websocket client -- import-time surface only, never called."""

from typing import Any


def connect(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "websockets is stubbed out for Pyodide compatibility; this Worker "
        "never uses langgraph_sdk's websocket streaming transport."
    )
