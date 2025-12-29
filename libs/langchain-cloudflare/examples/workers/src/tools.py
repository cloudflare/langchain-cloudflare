"""Tool definitions for the LangChain worker."""

from langchain_core.tools import tool

# MARK: - Weather Tools


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72F"


# MARK: - Stock Tools


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    return f"The stock price of {ticker} is $150.25"


# MARK: - Tool Registry

ALL_TOOLS = [get_weather, get_stock_price]
