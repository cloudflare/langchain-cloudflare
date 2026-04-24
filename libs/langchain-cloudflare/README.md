# langchain-cloudflare

This package contains the LangChain integration with CloudflareWorkersAI

## Installation

```bash
pip install -U langchain-cloudflare
```

And you should configure credentials by setting the following environment variables:

- `CF_ACCOUNT_ID`

AND

- `CF_API_TOKEN` (if using a single token scoped for all services)

OR (if using separately scoped tokens)

- `CF_AI_API_TOKEN` (CloudflareWorkersAI, CloudflareWorkersAIEmbeddings, CloudflareBrowserRunLoader, CloudflareBrowserRunTool)
- `CF_VECTORIZE_API_TOKEN` (CloudflareVectorize)
- `CF_D1_API_TOKEN` (CloudflareVectorize)
- `CF_D1_DATABASE_ID` (CloudflareVectorize)

> **Browser Run** requires the *Browser Rendering – Edit* permission on your API token. See [Browser Run setup](https://developers.cloudflare.com/browser-run/quick-actions/#before-you-begin).

## Chat Models

`ChatCloudflareWorkersAI` class exposes chat models from [CloudflareWorkersAI](https://developers.cloudflare.com/workers-ai/).

```python
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI

llm = ChatCloudflareWorkersAI()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CloudflareWorkersAIEmbeddings` class exposes embeddings from [CloudflareWorkersAI](https://developers.cloudflare.com/workers-ai/).

```python
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings

embeddings = CloudflareWorkersAIEmbeddings(
    model_name="@cf/baai/bge-base-en-v1.5"
)
embeddings.embed_query("What is the meaning of life?")
```

## VectorStores
`CloudflareVectorize` class exposes vectorstores from Cloudflare [Vectorize](https://developers.cloudflare.com/vectorize/).

```python
from langchain_cloudflare.vectorstores import CloudflareVectorize

vst = CloudflareVectorize(
    embedding=embeddings
)
vst.create_index(index_name="my-cool-vectorstore")
```

## Reranker

`CloudflareWorkersAIReranker` reranks documents by relevance using [Workers AI](https://developers.cloudflare.com/workers-ai/).

```python
from langchain_cloudflare import CloudflareWorkersAIReranker

reranker = CloudflareWorkersAIReranker()
results = reranker.rerank(
    query="What is the capital of France?",
    documents=["Paris is the capital of France.", "Berlin is in Germany."],
    top_k=2,
)
```

## Browser Run (Document Loader)

`CloudflareBrowserRunLoader` loads web pages as LangChain `Document` objects using [Cloudflare Browser Run](https://developers.cloudflare.com/browser-run/). It renders JavaScript-heavy pages on Cloudflare's global network and returns clean content via a simple REST API — no local browser required.

```python
from langchain_cloudflare import CloudflareBrowserRunLoader

# Single page -> markdown
loader = CloudflareBrowserRunLoader(
    urls=["https://developers.cloudflare.com/workers-ai/"],
    mode="markdown",
)
docs = loader.load()

# Multi-page crawl -> knowledge base
loader = CloudflareBrowserRunLoader(
    urls=["https://developers.cloudflare.com/cloudflare-one/"],
    mode="crawl",
    crawl_limit=50,
    crawl_depth=2,
)
docs = loader.load()

# Scrape specific elements with CSS selectors
loader = CloudflareBrowserRunLoader(
    urls=["https://example.com/pricing"],
    mode="scrape",
    elements=[{"selector": "h1"}, {"selector": ".plan-card"}],
)
docs = loader.load()  # one Document per matched selector group

# Async support
docs = await loader.aload()
```

Supported modes:

| Mode | Endpoint | Description |
|------|----------|-------------|
| `markdown` | [`/markdown`](https://developers.cloudflare.com/browser-run/quick-actions/markdown-endpoint/) | Clean markdown from any page |
| `crawl` | [`/crawl`](https://developers.cloudflare.com/browser-run/quick-actions/crawl-endpoint/) | Multi-page crawl with async polling |
| `scrape` | [`/scrape`](https://developers.cloudflare.com/browser-run/quick-actions/scrape-endpoint/) | CSS selector-based element extraction |
| `content` | [`/content`](https://developers.cloudflare.com/browser-run/quick-actions/content-endpoint/) | Raw rendered HTML |

## Browser Run (Agent Tool)

`CloudflareBrowserRunTool` gives [LangGraph](https://langchain-ai.github.io/langgraph/) agents the ability to interact with the live web.

```python
from langchain_cloudflare import CloudflareBrowserRunTool

# Read any page as markdown
tool = CloudflareBrowserRunTool(mode="markdown")
content = tool.invoke({"url": "https://example.com"})

# AI-powered structured data extraction
tool = CloudflareBrowserRunTool(
    mode="json",
    json_prompt="Extract the company name, pricing plans, and key features.",
)
data = tool.invoke({"url": "https://www.cloudflare.com/plans/"})
# Returns: {"company_name": "Cloudflare", "pricing_plans": [{"name": "Free", "price": "Free"}, ...]}

# Extract with a JSON schema for strict typing
tool = CloudflareBrowserRunTool(
    mode="json",
    json_response_format={
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "links": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
)

# Discover links on a page
tool = CloudflareBrowserRunTool(mode="links")
links = tool.invoke({"url": "https://example.com"})

# Use multiple tools in a LangGraph agent
from langgraph.prebuilt import ToolNode

tools = [
    CloudflareBrowserRunTool(mode="markdown"),
    CloudflareBrowserRunTool(mode="json", json_prompt="Extract key facts."),
    CloudflareBrowserRunTool(mode="links"),
]
tool_node = ToolNode(tools)  # each tool auto-named: cloudflare_browser_run_markdown, etc.
```

Supported modes:

| Mode | Endpoint | Description |
|------|----------|-------------|
| `markdown` | [`/markdown`](https://developers.cloudflare.com/browser-run/quick-actions/markdown-endpoint/) | Read any webpage as markdown |
| `json` | [`/json`](https://developers.cloudflare.com/browser-run/quick-actions/json-endpoint/) | AI-powered structured data extraction |
| `links` | [`/links`](https://developers.cloudflare.com/browser-run/quick-actions/links-endpoint/) | Discover all links on a page |
| `screenshot` | [`/screenshot`](https://developers.cloudflare.com/browser-run/quick-actions/screenshot-endpoint/) | Capture screenshot (base64 PNG) |
| `pdf` | [`/pdf`](https://developers.cloudflare.com/browser-run/quick-actions/pdf-endpoint/) | Generate PDF (base64) |

### Browser Run in LangGraph Workflows

Both the Loader and Tool integrate with all LangGraph patterns:

**As a custom node in a DAG:**

```python
from typing import TypedDict
from langchain_cloudflare import CloudflareBrowserRunLoader, CloudflareBrowserRunTool
from langgraph.graph import StateGraph, START, END


class ResearchState(TypedDict):
    url: str
    page_content: str
    links: list[str]


def fetch_page(state: ResearchState) -> dict:
    loader = CloudflareBrowserRunLoader(urls=[state["url"]], mode="markdown")
    docs = loader.load()
    return {"page_content": docs[0].page_content}


def extract_links(state: ResearchState) -> dict:
    tool = CloudflareBrowserRunTool(mode="links")
    links = tool.invoke({"url": state["url"]}).strip().split("\n")
    return {"links": links}


graph = StateGraph(ResearchState)
graph.add_node("fetch_page", fetch_page)
graph.add_node("extract_links", extract_links)
graph.add_edge(START, "fetch_page")
graph.add_edge(START, "extract_links")  # runs in parallel
graph.add_edge("fetch_page", END)
graph.add_edge("extract_links", END)
app = graph.compile()

result = app.invoke({"url": "https://example.com", "page_content": "", "links": []})
```

**As tools in a supervisor pattern:**

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

tools = [
    CloudflareBrowserRunTool(mode="markdown"),
    CloudflareBrowserRunTool(mode="json", json_prompt="Extract key facts."),
    CloudflareBrowserRunTool(mode="links"),
]
tool_node = ToolNode(tools)

graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_fn)  # your LLM-based supervisor
graph.add_node("browser_tools", tool_node)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", tools_condition)
graph.add_edge("browser_tools", "supervisor")
app = graph.compile()
```

**In a research loop with cycles:**

```python
def should_continue(state) -> str:
    if state["iteration"] >= 3 or not state["urls_to_visit"]:
        return "done"
    return "continue"

graph = StateGraph(ResearchState)
graph.add_node("discover", discover_links_node)
graph.add_node("fetch", fetch_page_node)
graph.add_edge(START, "discover")
graph.add_edge("discover", "fetch")
graph.add_conditional_edges("fetch", should_continue, {
    "continue": "discover",
    "done": END,
})
app = graph.compile()
```

## Release Notes
v0.1.1 (2025-04-08)

- Added ChatCloudflareWorkersAI integration
- Added CloudflareWorkersAIEmbeddings support
- Added CloudflareVectorize integration

v0.1.3 (2025-04-10)

- Added AI Gateway support for CloudflareWorkersAIEmbeddings
- Added Async support for CloudflareWorkersAIEmbeddings

v0.1.4 (2025-04-14)

- Added support for additional model parameters as explicit class attributes for ChatCloudflareWorkersAI

v0.1.6 (2025-05-01)

- Added Standalone D1 Metadata Filtering Methods
- Update Docs for more clarity around D1 Table/Vectorize Index Names

v0.1.8 (2025-05-11)

- Added support for environmental variables (embeddings, vectorstores)
