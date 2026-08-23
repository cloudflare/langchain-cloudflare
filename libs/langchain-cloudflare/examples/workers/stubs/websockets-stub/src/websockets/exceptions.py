"""Stub exceptions module -- only the names langgraph_sdk imports."""


class WebSocketException(Exception):
    pass


class ConnectionClosed(WebSocketException):
    pass


class ConnectionClosedError(ConnectionClosed):
    pass


class ConnectionClosedOK(ConnectionClosed):
    pass
