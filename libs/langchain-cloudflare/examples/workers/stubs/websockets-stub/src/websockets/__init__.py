"""Stub websockets package for Pyodide compatibility.

websockets has a C accelerator (speedups) with no Pyodide wheel, and this repo
doesn't use it anyway -- it's only pulled in because langgraph_sdk (a
transitive dependency of langgraph.runtime, which langgraph.graph imports)
has a websocket-based streaming transport we never actually invoke. This stub
provides just enough of the import surface for langgraph_sdk.stream.transport
to import cleanly; every `connect()` raises NotImplementedError if ever
actually called.
"""

__version__ = "15.0.1"
